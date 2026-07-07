import hashlib
import json
import os
import struct
import sys
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
from mlx_lm.models.cache import KVCache, ArraysCache

SAVE_RATIO = 0.95
DISK_CACHE_DIR = os.path.expanduser("~/.cache/elfmoon/kv_cache")
MAX_DISK_ENTRIES = 4


def _truncate_cache(cache: List[Any], new_offset: int) -> List[Tuple[str, Any]]:
    """Truncate cache and return per-layer (type_tag, data).

    - KVCache: tag='kv', data=(keys, vals) truncated to new_offset
    - ArraysCache: tag='arr', data=state (list of arrays, or None if empty)
    """
    truncated = []
    for c in cache:
        if isinstance(c, KVCache):
            if c.keys is not None and new_offset > 0:
                k = c.keys[..., :new_offset, :].astype(mx.float16)
                v = c.values[..., :new_offset, :].astype(mx.float16)
                mx.eval([k, v])
            else:
                k = mx.zeros((1, 1, 0, 1), dtype=mx.float16)
                v = mx.zeros((1, 1, 0, 1), dtype=mx.float16)
            truncated.append(("kv", (k, v)))
        elif isinstance(c, ArraysCache) and not c.empty():
            mx.eval([x for x in c.state if x is not None])
            truncated.append(("arr", c.state))
        else:
            truncated.append(("arr", None))
    return truncated


def _build_cache_objects(
    offset: int, layer_data: List[Tuple[str, Any]], n_layers: int
) -> List[Any]:
    """Reconstruct cache from saved data. SSM layers restore state if saved."""
    cache = []
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
    while len(cache) < n_layers:
        cache.append(ArraysCache(size=2))
    return cache


class KVCacheManager:
    """KV Cache store with hash-based lookup.

    Saves only the first SAVE_RATIO portion of the cache (system prompt part),
    so subsequent requests with different user messages still hit the cache.
    Also persists to disk so cache survives server restarts.
    """

    def __init__(self, max_entries: int = 4):
        self._caches: OrderedDict = OrderedDict()
        self._max_entries = max_entries
        os.makedirs(DISK_CACHE_DIR, exist_ok=True)

    # ---- hash ----

    def _hash_prefix(self, tokens: List[int], length: int) -> str:
        packed = b"".join(struct.pack("<i", t) for t in tokens[:length])
        return hashlib.sha256(packed).hexdigest()

    # ---- lookup (memory then disk) ----

    def lookup(self, prompt_ids: List[int], model) -> Tuple[Optional[List[Any]], int]:
        n_layers = len(getattr(model, "layers", None) or model.model.layers)
        # 1) Memory cache
        for key, (offset, layer_data) in reversed(list(self._caches.items())):
            if (
                len(prompt_ids) >= offset
                and self._hash_prefix(prompt_ids, offset) == key
            ):
                self._caches.move_to_end(key)
                return _build_cache_objects(offset, layer_data, n_layers), offset

        # 2) Disk cache
        for key, offset, layer_data in self._disk_search(prompt_ids):
            mem_entry = (offset, layer_data)
            self._caches[key] = mem_entry
            self._caches.move_to_end(key)
            while len(self._caches) > self._max_entries:
                self._caches.popitem(last=False)
            print(
                f"[KVC] disk→memory: key={key[:12]} offset={offset}",
                file=sys.stderr,
                flush=True,
            )
            return _build_cache_objects(offset, layer_data, n_layers), offset

        return None, 0

    # ---- save (memory + disk) ----

    def save(self, prompt_ids: List[int], cache: List[Any]):
        if cache is None or len(cache) == 0:
            return
        base_length = len(prompt_ids)
        if base_length < 20:
            return
        save_offset = max(20, int(base_length * SAVE_RATIO))
        key = self._hash_prefix(prompt_ids, save_offset)
        truncated = _truncate_cache(cache, save_offset)

        # Memory
        self._caches[key] = (save_offset, truncated)
        self._caches.move_to_end(key)
        while len(self._caches) > self._max_entries:
            self._caches.popitem(last=False)

        # Disk (fire-and-forget — slow I/O, don't block response)
        self._disk_save(key, save_offset, truncated, base_length)

    # ---- disk persistence ----

    def _disk_path(self, key: str) -> str:
        return os.path.join(DISK_CACHE_DIR, f"{key}.safetensors")

    def _meta_path(self, key: str) -> str:
        return os.path.join(DISK_CACHE_DIR, f"{key}.json")

    def _disk_save(
        self,
        key: str,
        offset: int,
        layer_data: List[Tuple[str, Any]],
        prompt_length: int,
    ):
        try:
            arrays: Dict[str, mx.array] = {}
            kv_indices = []
            for i, (tag, data) in enumerate(layer_data):
                if tag == "kv":
                    k, v = data
                    arrays[f"l{i}_keys"] = k
                    arrays[f"l{i}_values"] = v
                    kv_indices.append(i)

            if arrays:
                mx.save_safetensors(self._disk_path(key), arrays)

            meta = {
                "hash": key,
                "offset": offset,
                "num_layers": len(layer_data),
                "kv_indices": kv_indices,
                "prompt_tokens": prompt_length,
                "created_at": time.time(),
            }
            with open(self._meta_path(key), "w") as f:
                json.dump(meta, f)

            self._cleanup_disk()

            print(
                f"[KVC] disk save: key={key[:12]} offset={offset} kv_layers={len(kv_indices)}/{len(layer_data)}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as e:
            print(f"[KVC] disk save error: {e}", file=sys.stderr, flush=True)

    def _disk_load_arrays(self, key: str) -> List[Tuple[str, Any]]:
        meta_path = self._meta_path(key)
        with open(meta_path) as f:
            meta = json.load(f)
        num_layers = meta.get("num_layers", 48)
        kv_indices = set(meta.get("kv_indices", []))

        if not kv_indices:
            raise ValueError(f"No KVCache indices in {self._disk_path(key)}")

        arrays: Dict[str, mx.array] = mx.load(self._disk_path(key))  # type: ignore[assignment]
        layer_data: List[Tuple[str, Any]] = []
        for i in range(num_layers):
            if i in kv_indices:
                k = arrays.get(f"l{i}_keys")
                v = arrays.get(f"l{i}_values")
                if k is not None and v is not None:
                    layer_data.append(("kv", (k, v)))
                    continue
            layer_data.append(("arr", None))
        if all(tag == "arr" for tag, _ in layer_data):
            raise ValueError(f"No KVCache entries found in {self._disk_path(key)}")
        return layer_data

    def _disk_search(self, prompt_ids: List[int]):
        """Yield (key, offset, layer_data) for disk entries matching prompt_ids."""
        entries = self._list_disk_entries()
        for entry in entries:
            key = entry.get("hash", "")
            offset = entry.get("offset", 0)
            if (
                len(prompt_ids) >= offset
                and self._hash_prefix(prompt_ids, offset) == key
            ):
                try:
                    layer_data = self._disk_load_arrays(key)
                    yield key, offset, layer_data
                except Exception as e:
                    print(
                        f"[KVC] disk load error, removing corrupt entry: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    self._disk_delete(key)

    def _list_disk_entries(self) -> List[Dict]:
        if not os.path.isdir(DISK_CACHE_DIR):
            return []
        entries = []
        for fname in os.listdir(DISK_CACHE_DIR):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(DISK_CACHE_DIR, fname)) as f:
                        entries.append(json.load(f))
                except (json.JSONDecodeError, IOError):
                    pass
        return entries

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
