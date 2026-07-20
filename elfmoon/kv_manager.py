"""KV Cache 永続化マネージャ（ハイブリッドアーキテクチャ対応）。

Qwen3.6 系は full attention（KVCache）と linear attention（ArraysCache＝再帰状態）が
混在する。再帰状態は KV と違って途中位置に切り詰められないため、保存は
「全レイヤーが同一トークン数を処理した整合状態」でのみ行う。

運用フロー（api_server 側）:
  1. prefill 完了直後（プロンプト先頭 len-1 トークン処理時点）に snapshot() で
     状態への参照を捕捉する（この時点では重いコピーをしない）
  2. 生成完了後に save() で捕捉時点の状態をメモリ＋ディスクへ永続化する
  3. 次リクエストの lookup() はプロンプトとの最長プレフィックス一致を返す

ディスク形式は version=2（KV に加えて再帰状態も保存）。旧形式（v1）は
再帰状態を欠く不整合データのため、起動時に削除する。
"""

import hashlib
import json
import os
import struct
import sys
import threading
import time
from collections import OrderedDict
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

DISK_CACHE_DIR = os.environ.get("ELFMOON_KV_CACHE_DIR") or os.path.expanduser(
    "~/.cache/elfmoon/kv_cache"
)
MAX_DISK_ENTRIES = 4
MIN_SAVE_TOKENS = 20
FORMAT_VERSION = 2
# 情報ログ（hit/save 等）。対話 CLI ではプロンプト表示に割り込むため抑制できる。
# エラーログは本フラグに関わらず常に出す。
KVC_LOG = os.environ.get("ELFMOON_KVC_LOG", "1") != "0"


def _build_cache_objects(
    offset: int, layer_data: list[tuple[str, Any]], n_layers: int
) -> list[Any]:
    """保存データからキャッシュオブジェクトを再構築する。

    MLA (DeepSeek V4) 対応: tag == "mla" の場合は MLACacheState を返す。
    """
    cache: list[Any] = []
    for tag, data in layer_data:
        if tag == "kv":
            keys, vals = data
            kc = KVCache()
            kc.keys = keys
            kc.values = vals
            kc.offset = offset
            cache.append(kc)
        elif tag == "arr":
            ac = ArraysCache(size=2)
            if data is not None:
                ac.state = [mx.array(x) for x in data]
            cache.append(ac)
        elif tag == "mla":
            cache.append(_mla_state_from_dict(data))
    while len(cache) < n_layers:
        cache.append(ArraysCache(size=2))
    return cache


class KVCacheManager:
    """整合スナップショット方式の KV Cache ストア（メモリ＋ディスク）。"""

    def __init__(self, max_entries: int = 4, cache_dir: str = DISK_CACHE_DIR):
        self._caches: OrderedDict = OrderedDict()
        self._max_entries = max_entries
        self._dir = cache_dir
        self._disk_lock = threading.Lock()
        os.makedirs(self._dir, exist_ok=True)
        self._purge_old_format()

    # ---- hash ----

    def _hash_prefix(self, tokens: list[int], length: int) -> str:
        packed = b"".join(struct.pack("<i", t) for t in tokens[:length])
        return hashlib.sha256(packed).hexdigest()

    # ---- snapshot ----

    def snapshot(self, cache: list[Any]) -> list[tuple[str, Any]] | None:
        """prefill 直後のキャッシュ状態を捕捉する（save() で永続化する）。

        MLX 配列への添字代入は新しいバッキングを生成するため、ArraysCache の
        state は mx.array() でコピーすれば以後の生成に影響されない。
        KVCache のバッファは offset 以降にのみ追記されるので、参照を保持して
        save() 時に offset までスライスすれば捕捉時点の内容が得られる。
        非対応のキャッシュ型や未初期化の再帰状態がある場合は None（保存不可）。
        """
        snap: list[tuple[str, Any]] = []
        for c in cache:
            if isinstance(c, KVCache):
                snap.append(("kv", (c.keys, c.values, c.offset)))
            elif isinstance(c, ArraysCache):
                state = c.state
                if state is None or any(x is None for x in state):
                    return None
                # 上のガードで全要素 non-None を保証済み（if は型絞り込み用）
                snap.append(("arr", [mx.array(x) for x in state if x is not None]))
            else:
                return None
        return snap

    # ---- lookup（メモリ→ディスク、最長プレフィックス一致） ----

    def lookup(self, prompt_ids: list[int], model) -> tuple[list[Any] | None, int]:
        n_layers = len(getattr(model, "layers", None) or model.model.layers)

        # 1) メモリ: 一致する中で最長 offset のエントリ
        best_key = None
        best: tuple[int, Any] | None = None
        for key, (offset, layer_data) in self._caches.items():
            if (
                offset <= len(prompt_ids)
                and (best is None or offset > best[0])
                and self._hash_prefix(prompt_ids, offset) == key
            ):
                best_key, best = key, (offset, layer_data)
        if best is not None:
            self._caches.move_to_end(best_key)
            return _build_cache_objects(best[0], best[1], n_layers), best[0]

        # 2) ディスク: offset 降順に一致を試す
        candidates = [
            e
            for e in self._list_disk_entries()
            if e.get("offset", 0) <= len(prompt_ids)
            and self._hash_prefix(prompt_ids, e["offset"]) == e.get("hash", "")
        ]
        candidates.sort(key=lambda e: e["offset"], reverse=True)
        for entry in candidates:
            key, offset = entry["hash"], entry["offset"]
            try:
                layer_data = self._disk_load_arrays(key)
            except Exception as e:
                print(
                    f"[KVC] disk load error, removing corrupt entry: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                self._disk_delete(key)
                continue
            self._caches[key] = (offset, layer_data)
            self._caches.move_to_end(key)
            while len(self._caches) > self._max_entries:
                self._caches.popitem(last=False)
            if KVC_LOG:
                print(
                    f"[KVC] disk→memory: key={key[:12]} offset={offset}",
                    file=sys.stderr,
                    flush=True,
                )
            return _build_cache_objects(offset, layer_data, n_layers), offset

        return None, 0

    # ---- save（メモリ＋ディスク） ----

    def save(self, token_ids: list[int], snap: list[tuple[str, Any]] | None):
        """snapshot() の捕捉状態を token_ids（処理済み全トークン）キーで保存する。"""
        if snap is None:
            return
        offset = len(token_ids)
        if offset < MIN_SAVE_TOKENS:
            return
        key = self._hash_prefix(token_ids, offset)
        if key in self._caches and self._caches[key][0] == offset:
            # 同一内容が既にある → メモリ/ディスクとも書き直し不要
            self._caches.move_to_end(key)
            return

        layer_data: list[tuple[str, Any]] = []
        to_eval: list[mx.array] = []
        for tag, data in snap:
            if tag == "kv":
                keys, vals, kv_off = data
                end = min(kv_off, offset)
                if keys is not None and end > 0:
                    k = keys[..., :end, :].astype(mx.float16)
                    v = vals[..., :end, :].astype(mx.float16)
                else:
                    k = mx.zeros((1, 1, 0, 1), dtype=mx.float16)
                    v = mx.zeros((1, 1, 0, 1), dtype=mx.float16)
                layer_data.append(("kv", (k, v)))
                to_eval.extend([k, v])
            else:
                layer_data.append(("arr", data))
                to_eval.extend(data)
        mx.eval(to_eval)

        # メモリ
        self._caches[key] = (offset, layer_data)
        self._caches.move_to_end(key)
        while len(self._caches) > self._max_entries:
            self._caches.popitem(last=False)

        # ディスク（バックグラウンド書込み: 応答終端をブロックしない）
        threading.Thread(
            target=self._disk_save,
            args=(key, offset, layer_data, len(token_ids)),
            daemon=True,
        ).start()

    # ---- disk persistence ----

    def _disk_path(self, key: str) -> str:
        return os.path.join(self._dir, f"{key}.safetensors")

    def _meta_path(self, key: str) -> str:
        return os.path.join(self._dir, f"{key}.json")

    def _disk_save(
        self,
        key: str,
        offset: int,
        layer_data: list[tuple[str, Any]],
        prompt_length: int,
    ):
        try:
            with self._disk_lock:
                arrays: dict[str, mx.array] = {}
                kv_indices: list[int] = []
                arr_indices: dict[str, int] = {}
                for i, (tag, data) in enumerate(layer_data):
                    if tag == "kv":
                        k, v = data
                        arrays[f"l{i}_keys"] = k
                        arrays[f"l{i}_values"] = v
                        kv_indices.append(i)
                    elif data is not None:
                        for j, x in enumerate(data):
                            arrays[f"l{i}_arr{j}"] = x
                        arr_indices[str(i)] = len(data)

                if arrays:
                    mx.save_safetensors(self._disk_path(key), arrays)

                meta = {
                    "version": FORMAT_VERSION,
                    "hash": key,
                    "offset": offset,
                    "num_layers": len(layer_data),
                    "kv_indices": kv_indices,
                    "arr_indices": arr_indices,
                    "prompt_tokens": prompt_length,
                    "created_at": time.time(),
                }
                with open(self._meta_path(key), "w") as f:
                    json.dump(meta, f)

                self._cleanup_disk()

            if KVC_LOG:
                print(
                    f"[KVC] disk save: key={key[:12]} offset={offset} "
                    f"kv={len(kv_indices)} arr={len(arr_indices)}/{len(layer_data)}",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as e:
            print(f"[KVC] disk save error: {e}", file=sys.stderr, flush=True)

    def _disk_load_arrays(self, key: str) -> list[tuple[str, Any]]:
        with open(self._meta_path(key)) as f:
            meta = json.load(f)
        if meta.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported cache format: {meta.get('version')}")
        num_layers = meta["num_layers"]
        kv_indices = set(meta.get("kv_indices", []))
        arr_indices = {int(k): v for k, v in meta.get("arr_indices", {}).items()}

        arrays: dict[str, mx.array] = mx.load(self._disk_path(key))  # type: ignore[assignment]
        layer_data: list[tuple[str, Any]] = []
        for i in range(num_layers):
            if i in kv_indices:
                layer_data.append(
                    ("kv", (arrays[f"l{i}_keys"], arrays[f"l{i}_values"]))
                )
            elif i in arr_indices:
                layer_data.append(
                    ("arr", [arrays[f"l{i}_arr{j}"] for j in range(arr_indices[i])])
                )
            else:
                layer_data.append(("arr", None))
        if all(tag == "arr" and d is None for tag, d in layer_data):
            raise ValueError(f"No cache entries found in {self._disk_path(key)}")
        return layer_data

    def _list_disk_entries(self) -> list[dict]:
        if not os.path.isdir(self._dir):
            return []
        entries = []
        for fname in os.listdir(self._dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(self._dir, fname)) as f:
                        meta = json.load(f)
                    if meta.get("version") == FORMAT_VERSION:
                        entries.append(meta)
                except (OSError, json.JSONDecodeError):
                    pass
        return entries

    def _purge_old_format(self):
        """旧形式（再帰状態を欠く v1）のエントリを削除する。"""
        if not os.path.isdir(self._dir):
            return
        for fname in os.listdir(self._dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    meta = json.load(f)
                version = meta.get("version")
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                version = None
            if version != FORMAT_VERSION:
                key = fname[: -len(".json")]
                self._disk_delete(key)
                if KVC_LOG:
                    print(
                        f"[KVC] purged old-format entry: {key[:12]}",
                        file=sys.stderr,
                        flush=True,
                    )

    def _cleanup_disk(self):
        entries = self._list_disk_entries()
        if len(entries) <= MAX_DISK_ENTRIES:
            return
        entries.sort(key=lambda e: e.get("created_at", 0))
        for entry in entries[:-MAX_DISK_ENTRIES]:
            self._disk_delete(entry["hash"])

    def _disk_delete(self, key: str):
        for path in [self._disk_path(key), self._meta_path(key)]:
            if os.path.exists(path):
                os.remove(path)

    # ---- clear ----

    def clear(self):
        self._caches.clear()

    def clear_disk(self):
        """Remove all disk cache entries."""
        for entry in self._list_disk_entries():
            self._disk_delete(entry["hash"])


kv_manager = KVCacheManager()


# ======== MLA (DeepSeek V4) キャッシュ対応 ========


class MLACacheState:
    """1 層の DeepseekV4Attention の可変状態を保持する。"""

    __slots__ = (
        "ltype",
        "win",
        "comp_kvs",
        "buf_kv",
        "buf_gate",
        "buf_idx",
        "entry_count",
        "idx_comp_kvs",
        "idx_buf_kv",
        "idx_buf_gate",
        "idx_buf_pos",
    )

    def __init__(self, ltype: str):
        self.ltype = ltype
        self.win: mx.array | None = None
        self.comp_kvs: mx.array | None = None
        self.buf_kv: mx.array | None = None
        self.buf_gate: mx.array | None = None
        self.buf_idx: int = 0
        self.entry_count: int = 0
        self.idx_comp_kvs: mx.array | None = None
        self.idx_buf_kv: mx.array | None = None
        self.idx_buf_gate: mx.array | None = None
        self.idx_buf_pos: int = 0


def _mla_state_from_attn(attn) -> MLACacheState:
    s = MLACacheState(attn.ltype)
    s.win = attn.win
    s.comp_kvs = attn.comp_kvs
    s.buf_kv = getattr(attn, "buf_kv", None)
    s.buf_gate = getattr(attn, "buf_gate", None)
    s.buf_idx = attn.buf_idx
    s.entry_count = attn.entry_count
    s.idx_comp_kvs = getattr(attn, "idx_comp_kvs", None)
    s.idx_buf_kv = getattr(attn, "idx_buf_kv", None)
    s.idx_buf_gate = getattr(attn, "idx_buf_gate", None)
    s.idx_buf_pos = getattr(attn, "idx_buf_pos", 0)
    return s


def _mla_apply_to_attn(attn, state: MLACacheState):
    attn.win = state.win
    attn.comp_kvs = state.comp_kvs
    if state.buf_kv is not None:
        attn.buf_kv = state.buf_kv
        attn.buf_gate = state.buf_gate
    attn.buf_idx = state.buf_idx
    attn.entry_count = state.entry_count
    if state.idx_comp_kvs is not None:
        attn.idx_comp_kvs = state.idx_comp_kvs
        attn.idx_buf_kv = state.idx_buf_kv
        attn.idx_buf_gate = state.idx_buf_gate
        attn.idx_buf_pos = state.idx_buf_pos


def _mla_state_to_dict(state: MLACacheState) -> dict:
    d: dict = {"ltype": state.ltype}
    for k in (
        "win",
        "comp_kvs",
        "buf_kv",
        "buf_gate",
        "idx_comp_kvs",
        "idx_buf_kv",
        "idx_buf_gate",
    ):
        v = getattr(state, k)
        if v is not None:
            d[k] = v
    d["buf_idx"] = state.buf_idx
    d["entry_count"] = state.entry_count
    d["idx_buf_pos"] = state.idx_buf_pos
    return d


def _mla_state_from_dict(d: dict) -> MLACacheState:
    s = MLACacheState(d["ltype"])
    for k in (
        "win",
        "comp_kvs",
        "buf_kv",
        "buf_gate",
        "idx_comp_kvs",
        "idx_buf_kv",
        "idx_buf_gate",
    ):
        v = d.get(k)
        if v is not None:
            setattr(s, k, v)
    s.buf_idx = d.get("buf_idx", 0)
    s.entry_count = d.get("entry_count", 0)
    s.idx_buf_pos = d.get("idx_buf_pos", 0)
    return s


def snapshot_v4(model) -> list[tuple[str, Any]]:
    snap: list[tuple[str, Any]] = []
    for layer in model.layers:
        snap.append(("mla", _mla_state_to_dict(_mla_state_from_attn(layer.attn))))
    return snap


def restore_v4(model, layer_data: list[tuple[str, Any]]):
    for i, (tag, data) in enumerate(layer_data):
        if tag == "mla" and i < len(model.layers):
            _mla_apply_to_attn(model.layers[i].attn, _mla_state_from_dict(data))


# ---- KVCacheManager のディスク保存/読込を MLA 対応に拡張 ----


def _mla_disk_save(self, key, offset, layer_data, prompt_length):
    try:
        with self._disk_lock:
            arrays: dict[str, mx.array] = {}
            arr_map: dict[int, list[str]] = {}
            for i, (tag, data) in enumerate(layer_data):
                layer_arrs: list[str] = []
                for k, v in data.items():
                    if isinstance(v, mx.array):
                        aname = f"l{i}_{k}"
                        arrays[aname] = v
                        layer_arrs.append(k)
                arr_map[i] = layer_arrs
            if arrays:
                mx.save_safetensors(self._disk_path(key), arrays)
            meta = {
                "version": FORMAT_VERSION,
                "hash": key,
                "offset": offset,
                "num_layers": len(layer_data),
                "mla_indices": {str(i): arr_map[i] for i in range(len(layer_data))},
                "prompt_tokens": prompt_length,
                "created_at": time.time(),
            }
            with open(self._meta_path(key), "w") as f:
                json.dump(meta, f)
            self._cleanup_disk()
    except Exception as e:
        print(f"[KVC] mla disk save error: {e}", file=sys.stderr, flush=True)


def _mla_disk_load(self, key, meta):
    loaded = mx.load(self._disk_path(key))
    arrays = loaded if isinstance(loaded, dict) else {}
    mla_indices = {int(k): v for k, v in meta.get("mla_indices", {}).items()}
    layer_data: list[tuple[str, Any]] = []
    for i in range(meta["num_layers"]):
        keys = mla_indices.get(i, [])
        d: dict = {}
        for k in keys:
            aname = f"l{i}_{k}"
            if aname in arrays:
                d[k] = arrays[aname]
        d["ltype"] = "sliding"
        layer_data.append(("mla", d))
    return layer_data


_orig_disk_save = KVCacheManager._disk_save


def _patched_disk_save(self, key, offset, layer_data, prompt_length):
    if layer_data and layer_data[0][0] == "mla":
        _mla_disk_save(self, key, offset, layer_data, prompt_length)
    else:
        _orig_disk_save(self, key, offset, layer_data, prompt_length)


_orig_disk_load = KVCacheManager._disk_load_arrays


def _patched_disk_load(self, key):
    meta_path = self._meta_path(key)
    with open(meta_path) as f:
        meta = json.load(f)
    if "mla_indices" in meta:
        return _mla_disk_load(self, key, meta)
    return _orig_disk_load(self, key)


KVCacheManager._disk_save = _patched_disk_save
KVCacheManager._disk_load_arrays = _patched_disk_load
