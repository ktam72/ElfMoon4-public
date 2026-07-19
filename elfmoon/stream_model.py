"""最終統合: mlx_lm の Qwen MoE 系モデルの各層MoEを ElfMoon ストリーミングMoEに差し替える。
融合 switch_mlp（Qwen3.6-35B で約17GB）を解放し、ExpertStore + ResidentCache から必要分だけ流す。
"""

import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
from expert_store import BITS, GROUP, ExpertStore
from resident_cache import ResidentCache
from slot_cache import GlobalSlotCache, _SENTINEL

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_NAME = "qwen3.6-35b-mlx"

# モデル置き場のルート。任意ディレクトリ（外部SSD等）を指せる唯一の結合点。
MODELS_ROOT = os.environ.get("ELFMOON_MODELS_ROOT", os.path.join(_HERE, "..", "models"))

# gather_qmm 高速プレフィルを使う最小チャンクトークン数。これ未満は per-expert 経路
# （ResidentCache が効き短チャンクで有利）。実測: 2048 で互角、4096 で fused 3.4倍。
FUSED_MIN_TOKENS = int(os.environ.get("ELFMOON_FUSED_MIN_TOKENS", "2048"))


def resolve_model(name=None):
    """モデル名 → (model_path, store_dir) を解決する。

    store はモデルディレクトリ直下の `store/` に必ず存在する規約（integrate.py が作る）。
    ELFMOON_MODEL_DIR/ELFMOON_STORE_DIR が明示されていれば旧方式として最優先する。
    """
    explicit_model = os.environ.get("ELFMOON_MODEL_DIR")
    if name is None and explicit_model:
        model_path = explicit_model
        store_dir = os.environ.get(
            "ELFMOON_STORE_DIR", os.path.join(model_path, "store")
        )
        return model_path, store_dir

    name = name or os.environ.get("ELFMOON_MODEL", DEFAULT_MODEL_NAME)
    model_path = os.path.join(MODELS_ROOT, name)
    store_dir = os.path.join(model_path, "store")
    return model_path, store_dir


def list_models():
    """MODELS_ROOT 直下で config.json を持つモデルを列挙する。
    戻り値: (name, has_store, is_native) のリスト
      - has_store: DeepSeek 系 MoE の store/ が存在するか
      - is_native: mlx_lm で直接動作するモデル（gemma4 等）か
    """
    if not os.path.isdir(MODELS_ROOT):
        return []
    names = []
    for entry in sorted(os.listdir(MODELS_ROOT)):
        d = os.path.join(MODELS_ROOT, entry)
        cfg_path = os.path.join(d, "config.json")
        if os.path.isfile(cfg_path):
            has_store = os.path.isdir(os.path.join(d, "store"))
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                mt = cfg.get("model_type", "")
                try:
                    from mlx_lm.utils import _get_classes

                    _get_classes({"model_type": mt})
                    is_native = not has_store
                except Exception:
                    is_native = False
            except Exception:
                is_native = False
            names.append((entry, has_store, is_native))
    return names


# 後方互換: モジュールレベル定数（--model 未指定・env var 未設定時は既定モデル）
MODEL_PATH, STORE_DIR = resolve_model()


# ---- A: Compiled MoE decode ----


@mx.compile
def _infer_qparams(w, s):
    """Infer (group_size, bits) from quantized weight and scales shapes."""
    n_out, n_packed = w.shape
    n_groups = s.shape[-1]
    for bits in (4, 8):
        effective = n_packed * 32 // bits
        if effective % n_groups == 0:
            gs = effective // n_groups
            if gs in {32, 64, 128}:
                return gs, bits
    return 64, 4


def _decode_moe(
    x: mx.array,
    w_gu: mx.array,
    s_gu: mx.array,
    b_gu: mx.array | None,
    w_dw: mx.array,
    s_dw: mx.array,
    b_dw: mx.array | None,
    weights: mx.array,
    top_k: int,
    shared=None,
    group_size=GROUP,
    bits=BITS,
    mode="affine",
):
    xb = mx.broadcast_to(x, (2 * top_k, 1, x.shape[-1]))
    gu = mx.quantized_matmul(
        xb,
        w_gu,
        s_gu,
        b_gu,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    g, u = gu[:top_k], gu[top_k:]
    h = (g * mx.sigmoid(g)) * u
    yo = mx.quantized_matmul(
        h, w_dw, s_dw, b_dw, transpose=True, group_size=group_size, bits=bits, mode=mode
    )
    result = (yo[:, 0, :] * weights[:, None]).sum(0)
    if shared is not None:
        gated = len(shared) == 11
        if gated:
            (
                sg_w,
                sg_s,
                sg_b,
                sg_bits,
                sg_gs,
                se_guw,
                se_gus,
                se_gub,
                se_dw,
                se_ds,
                se_db,
            ) = shared
        else:
            (
                se_guw,
                se_gus,
                se_gub,
                se_dw,
                se_ds,
                se_db,
            ) = shared
        se_gu = mx.quantized_matmul(
            x,
            se_guw,
            se_gus,
            se_gub,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        se_g, se_u = se_gu[:, : se_gu.shape[-1] // 2], se_gu[:, se_gu.shape[-1] // 2 :]
        se_h = (se_g * mx.sigmoid(se_g)) * se_u
        se_gs, se_bits = _infer_qparams(se_dw, se_ds)
        se_out = mx.quantized_matmul(
            se_h,
            se_dw,
            se_ds,
            se_db,
            transpose=True,
            group_size=se_gs,
            bits=se_bits,
            mode=mode,
        )
        if gated:
            sg = mx.quantized_matmul(
                x, sg_w, sg_s, sg_b, transpose=True, group_size=sg_gs, bits=sg_bits
            )
            result = result + mx.sigmoid(sg) * se_out
        else:
            result = result + se_out
    return result


def _decode_moe_gather(
    x,
    gsc,
    layer,
    slot_ids,
    weights,
    top_k,
    shared=None,
    group_size=GROUP,
    bits=BITS,
    mode="affine",
):
    """gather_qmm-based MoE decode using GlobalSlotCache 3D buffer.

    All experts must be resident (slot_ids all valid). Uses 3 gather_qmm
    calls (gate, up, down) — no explicit data copy, gather fused into matmul.

    NOTE: DEAD END (directive_deepseek_09.md).
    Real stream_generate path measured 0.55x vs baseline.
    Kept for reference only. Do not use for new development.
    """
    xb = mx.expand_dims(x, (-2, -3))
    rhs = slot_ids.astype(mx.uint32).reshape(1, -1)

    g = mx.gather_qmm(
        xb,
        gsc.gate_wq,
        gsc.gate_s,
        gsc.gate_b,
        rhs_indices=rhs,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    u = mx.gather_qmm(
        xb,
        gsc.up_wq,
        gsc.up_s,
        gsc.up_b,
        rhs_indices=rhs,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    h = (g * mx.sigmoid(g)) * u
    yo = mx.gather_qmm(
        h,
        gsc.down_wq,
        gsc.down_s,
        gsc.down_b,
        rhs_indices=rhs,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    wg = weights.astype(yo.dtype).reshape(1, -1, 1)
    result = (yo[:, :, 0, :] * wg).sum(1)
    if shared is not None:
        result = result + _shared_ffn(x, shared, group_size, bits, mode)
    return result


def _shared_ffn(x, shared, group_size, bits, mode):
    gated = len(shared) == 11
    if gated:
        (
            sg_w,
            sg_s,
            sg_b,
            sg_bits,
            sg_gs,
            se_guw,
            se_gus,
            se_gub,
            se_dw,
            se_ds,
            se_db,
        ) = shared
    else:
        se_guw, se_gus, se_gub, se_dw, se_ds, se_db = shared
    se_gu = mx.quantized_matmul(
        x,
        se_guw,
        se_gus,
        se_gub,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    se_g, se_u = se_gu[:, : se_gu.shape[-1] // 2], se_gu[:, se_gu.shape[-1] // 2 :]
    se_h = (se_g * mx.sigmoid(se_g)) * se_u
    se_gs, se_bits = _infer_qparams(se_dw, se_ds)
    se_out = mx.quantized_matmul(
        se_h,
        se_dw,
        se_ds,
        se_db,
        transpose=True,
        group_size=se_gs,
        bits=se_bits,
        mode=mode,
    )
    if gated:
        sg = mx.quantized_matmul(
            x, sg_w, sg_s, sg_b, transpose=True, group_size=sg_gs, bits=sg_bits
        )
        return mx.sigmoid(sg) * se_out
    return se_out


# ---- B: プレフィル(N>1) 用 融合テンソル mmap + gather_qmm ----


class FusedPrefillStore:
    """元モデル safetensors の融合 expert テンソル（[n_experts,...]）への mmap アクセス。

    integrate.py が分解する前の `switch_mlp.{gate,up,down}_proj.{weight,scales,biases}`
    をプレフィル専用に直接読む。per-expert の stack を不要にし gather_qmm へ直行できる。
    層の dict は保持しない（materialize された融合テンソル ~450MB/層 が40層で
    18GB 蓄積するのを防ぐ）。warm 時の再読込は OS のページキャッシュが担保する。
    """

    def __init__(self, model_path):
        idx_path = os.path.join(model_path, "model.safetensors.index.json")
        wm = json.load(open(idx_path))["weight_map"]
        self._model_path = model_path
        self._layer_keys = {}  # layer -> {"gate.wq": full_key, ...}
        for k in wm:
            if ".mlp.switch_mlp.gate_proj.weight" not in k:
                continue
            l = int(k.split(".layers.")[1].split(".")[0])
            base = k.rsplit(".gate_proj.weight", 1)[0]
            keys = {}
            for p in ("gate", "up", "down"):
                for t, suf in (("wq", "weight"), ("s", "scales"), ("b", "biases")):
                    keys[f"{p}.{t}"] = f"{base}.{p}_proj.{suf}"
            self._layer_keys[l] = keys
        self._wm = wm
        if not self._layer_keys:
            raise ValueError(f"融合 switch_mlp キーが見つからない: {idx_path}")

    def __contains__(self, layer):
        return layer in self._layer_keys

    def load(self, layer):
        """層の融合テンソル dict（lazy mmap 参照）を返す。呼び出し側は保持しないこと。"""
        keys = self._layer_keys[layer]
        shards = {self._wm[v] for v in keys.values()}
        out = {}
        for sh in shards:
            data = mx.load(os.path.join(self._model_path, sh))
            out.update({k: data[v] for k, v in keys.items() if self._wm[v] == sh})
        return out


def _prefill_moe_gather(xf, idx, w, fused, group_size, bits, mode):
    """融合テンソル（全 expert stack 済み）に対して gather_qmm で MoE を一括計算。

    per-expert の Python ループ（expert_groups 構築・top_k×n_experts 個の小 matmul
    dispatch・tolist 同期）を、expert 順ソート付きの 3 カーネルに置き換える。
    数値は per-expert 経路と bf16 丸め差（~1e-3）で一致。
    """
    x = mx.expand_dims(xf, (-2, -3))  # [N,1,1,DIM]
    do_sort = idx.size >= 64
    if do_sort:
        xs, ii, inv = _gather_sort(x, idx)  # expert 連続アクセスのためソート
    else:
        xs, ii, inv = x, idx, None

    def gq(xin, p):
        return mx.gather_qmm(
            xin,
            fused[f"{p}.wq"],
            fused[f"{p}.s"],
            fused[f"{p}.b"],
            rhs_indices=ii,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
            sorted_indices=do_sort,
        )

    g = gq(xs, "gate")
    u = gq(xs, "up")
    h = (g * mx.sigmoid(g)) * u
    yo = gq(h, "down")
    if do_sort:
        yo = _scatter_unsort(yo, inv, idx.shape)  # [N, top_k, 1, DIM]
    yo = yo.squeeze(-2)  # [N, top_k, DIM]
    return (yo * w[:, :, None].astype(yo.dtype)).sum(axis=1)  # [N, DIM]


# ---- Streaming MoE（MoE 層差し替え） ----


class StreamingMoE(nn.Module):
    """層の融合MoEを置換。routerは元の量子化gateを流用、expertはストア/キャッシュから。"""

    def __init__(
        self,
        layer_idx,
        gate,
        n_experts,
        top_k,
        store,
        cache,
        shared_exp=None,
        shared_gate=None,
        norm=True,
        activation="softmax",
        correction_bias=None,
        routing_scale=1.0,
        group_size=GROUP,
        bits=BITS,
        mode="affine",
        fused_store=None,
        is_last_moe=False,
        gsc=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.gate = gate
        self.n_experts = n_experts
        self.top_k = top_k
        self._store = store
        self._cache = cache
        self._gsc = gsc
        self._fused_store = fused_store
        self._is_last_moe = is_last_moe
        self.norm = norm
        self.activation = activation
        self.correction_bias = correction_bias
        self.routing_scale = routing_scale
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        if shared_exp is not None:
            se = shared_exp
            # gate+up の量子化重みを結合して1回の quantized_matmul に統合する
            se_gu_w = mx.concatenate([se.gate_proj.weight, se.up_proj.weight], axis=0)
            se_gu_s = mx.concatenate([se.gate_proj.scales, se.up_proj.scales], axis=0)
            se_gu_b = mx.concatenate([se.gate_proj.biases, se.up_proj.biases], axis=0)
            se_tuple = (
                se_gu_w,
                se_gu_s,
                se_gu_b,
                se.down_proj.weight,
                se.down_proj.scales,
                se.down_proj.biases,
            )
            if shared_gate is not None:
                sg = shared_gate
                self._shared = (
                    sg.weight,
                    sg.scales,
                    sg.biases,
                    sg.bits,
                    sg.group_size,
                ) + se_tuple
            else:
                self._shared = se_tuple
            # Validate: gate+up intermediate size must match down_proj input dim
            gu_inter = se_gu_w.shape[0] // 2
            dw_gs, dw_bits = _infer_qparams(se.down_proj.weight, se.down_proj.scales)
            dw_in = se.down_proj.weight.shape[-1] * 32 // dw_bits
            if gu_inter != dw_in:
                print(
                    f"  layer {self.layer_idx}: shared expert dim mismatch (gate+up={gu_inter}, down={dw_in}) — disabling",
                    flush=True,
                )
                self._shared = None
        else:
            self._shared = None

    def __call__(self, x):
        shp = x.shape
        xf = x.reshape(-1, shp[-1])
        logits = self.gate(xf)
        if self.activation == "sigmoid":
            probs = mx.sigmoid(logits)
        elif self.activation == "sqrtsoftplus":
            probs = mx.sqrt(mx.log1p(mx.exp(-mx.abs(logits))) + mx.maximum(logits, 0))
        else:
            probs = mx.softmax(logits, axis=-1)
        # 選択(top-k)には補正バイアス込みのスコアを使うが、重みには補正前の
        # probs を使う（DeepSeek/Kimi式 aux-loss-free ルーティングの規約）。
        sel_probs = (
            probs + self.correction_bias if self.correction_bias is not None else probs
        )
        idx = mx.argpartition(-sel_probs, self.top_k - 1, axis=-1)[:, : self.top_k]
        w = mx.take_along_axis(probs, idx, axis=-1)
        if self.norm:
            w = w / mx.sum(w, axis=-1, keepdims=True)
        if self.routing_scale != 1.0:
            w = w * self.routing_scale
        N = xf.shape[0]

        def load(e):
            return self._cache.get(
                (self.layer_idx, e), lambda e=e: self._store.load(self.layer_idx, e)
            )

        if N == 1:
            gsc = self._gsc
            if gsc is not None:
                # M2 path: fill misses, then ALWAYS run gather_qmm
                mx.eval(idx, w)
                idx_l = idx.tolist()[0]

                miss_ids = [e for e in idx_l if (self.layer_idx, e) not in gsc._lru]
                if miss_ids:
                    gsc.get_slots(self.layer_idx, miss_ids)

                # GPU slot_ids (all valid after fill)
                slot_ids = mx.take(
                    gsc.slot_map[self.layer_idx], idx[0].astype(mx.uint32)
                ).astype(mx.uint32)
                weights_gpu = w[0].astype(mx.float16)
                result = _decode_moe_gather(
                    xf[0:1],
                    gsc,
                    self.layer_idx,
                    slot_ids,
                    weights_gpu,
                    self.top_k,
                    shared=self._shared,
                    group_size=self.group_size,
                    bits=self.bits,
                    mode=self.mode,
                ).reshape(shp)
                return result.astype(x.dtype)

            # Non-GSC path: existing stack + _decode_moe
            mx.eval(idx, w)
            idx_l = idx.tolist()
            experts = [load(e) for e in idx_l[0]]

            w_gu = mx.stack(
                [e["gate.wq"] for e in experts] + [e["up.wq"] for e in experts]
            )
            s_gu = mx.stack(
                [e["gate.s"] for e in experts] + [e["up.s"] for e in experts]
            )
            b_gu = (
                mx.stack(
                    [e.get("gate.b") for e in experts]
                    + [e.get("up.b") for e in experts]
                )
                if any(e.get("gate.b") is not None for e in experts)
                else None
            )
            w_dw = mx.stack([e["down.wq"] for e in experts])
            s_dw = mx.stack([e["down.s"] for e in experts])
            b_dw = (
                mx.stack([e.get("down.b") for e in experts])
                if any(e.get("down.b") is not None for e in experts)
                else None
            )

            weights = w[0].astype(mx.float16)
            result = _decode_moe(
                xf[0:1],
                w_gu,
                s_gu,
                b_gu,
                w_dw,
                s_dw,
                b_dw,
                weights,
                self.top_k,
                shared=self._shared,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            ).reshape(shp)
            return result.astype(x.dtype)

        # --- プレフィル(N>1) 高速経路: 融合テンソル mmap + gather_qmm ---
        # per-expert ループの Python dispatch（プレフィル時間の~78%）を3カーネルに
        # 集約する。融合テンソル読込は ~450MB/層 の固定費のため、長チャンクのみ有効
        # （短チャンクは ResidentCache が効く per-expert 経路が有利。実測の損益分岐:
        # N=2048 でほぼ互角、N=4096 で fused が3.4倍）。読めないモデルはフォールバック。
        fs = self._fused_store
        if fs is not None and N >= FUSED_MIN_TOKENS and self.layer_idx in fs:
            fused = fs.load(self.layer_idx)
            out = _prefill_moe_gather(
                xf, idx, w, fused, self.group_size, self.bits, self.mode
            )
            out = out.reshape(shp)
            out = self._add_shared(out, xf)
            # 融合テンソル(~450MB/層)の materialize を定期的に確定して解放する。
            # 全40層を遅延すると18GB蓄積する。async_eval は CPU をブロックせず
            # GPU 実行を開始させるので、次層のロードと計算がオーバーラップする。
            if self.layer_idx % 4 == 3:
                mx.async_eval(out)
            if self._is_last_moe:
                # プレフィルで使った融合テンソルのバッファ(~18GB)をプールに残すと
                # decode 時のメモリ圧で速度が落ちる（実測 23→17 t/s）ため即返却する
                mx.eval(out)
                mx.clear_cache()
            return out.astype(x.dtype)

        # --- プレフィル(N>1) フォールバック: Expert単位でトークンをバッチ処理 ---
        mx.eval(idx, w)
        idx_l = idx.tolist()
        w_l = w.tolist()
        expert_groups = {}
        for t in range(N):
            for j in range(self.top_k):
                e = int(idx_l[t][j])
                if e not in expert_groups:
                    expert_groups[e] = []
                expert_groups[e].append((t, w_l[t][j]))

        # scatter-add はベクトル化（mx.zeros().at[].add()）。トークン単位のPython
        # ループを廃し、expert毎に1回のscatterで集約する。8000超の逐次addノードが
        # 消え、プリフィルが35Bで約1.4倍/80Bで約1.86倍に高速化（数値パリティ一致）。
        out = mx.zeros((N, xf.shape[-1]), dtype=xf.dtype)
        for e, items in expert_groups.items():
            exp = load(e)
            indices = mx.array([it[0] for it in items])
            weights = mx.array([it[1] for it in items])
            xb = xf[indices]
            mode = self.mode
            g = mx.quantized_matmul(
                xb,
                exp["gate.wq"],
                exp["gate.s"],
                exp.get("gate.b"),
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=mode,
            )
            u = mx.quantized_matmul(
                xb,
                exp["up.wq"],
                exp["up.s"],
                exp.get("up.b"),
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=mode,
            )
            h = (g * mx.sigmoid(g)) * u
            yo = mx.quantized_matmul(
                h,
                exp["down.wq"],
                exp["down.s"],
                exp.get("down.b"),
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=mode,
            )
            contrib = yo * weights[:, None].astype(yo.dtype)
            out = out.at[indices].add(contrib)
        out = out.reshape(shp)
        out = self._add_shared(out, xf)
        return out.astype(x.dtype)

    def _add_shared(self, out, xf):
        """shared expert の寄与を out に加算する（N>1 の両経路で共用）。"""
        if self._shared is None:
            return out
        gated = len(self._shared) == 11
        if gated:
            (
                sg_w,
                sg_s,
                sg_b,
                sg_bits,
                sg_gs,
                se_gu_w,
                se_gu_s,
                se_gu_b,
                se_dw,
                se_ds,
                se_db,
            ) = self._shared
        else:
            (
                se_gu_w,
                se_gu_s,
                se_gu_b,
                se_dw,
                se_ds,
                se_db,
            ) = self._shared
        se_h = mx.quantized_matmul(
            xf,
            se_gu_w,
            se_gu_s,
            se_gu_b,
            transpose=True,
            group_size=GROUP,
            bits=BITS,
        )
        k2 = se_h.shape[-1] // 2
        se_g, se_u = se_h[..., :k2], se_h[..., k2:]
        se_gated = se_g * mx.sigmoid(se_g)
        se_act = se_gated * se_u
        se_dw_gs, se_dw_bits = _infer_qparams(se_dw, se_ds)
        se_out = mx.quantized_matmul(
            se_act,
            se_dw,
            se_ds,
            se_db,
            transpose=True,
            group_size=se_dw_gs,
            bits=se_dw_bits,
        )
        se_out = se_out.reshape(out.shape)
        if gated:
            sg = mx.quantized_matmul(
                xf, sg_w, sg_s, sg_b, transpose=True, group_size=sg_gs, bits=sg_bits
            )
            return out + mx.sigmoid(sg).reshape(*out.shape[:-1], 1) * se_out
        return out + se_out


# ---- Wiring ----


def _read_top_k(model_path=None):
    """config.json から num_experts_per_tok を読み取る。
    35B は text_config 入れ子、80B はフラット。両方対応。
    """
    try:
        cfg = json.load(open(os.path.join(model_path or MODEL_PATH, "config.json")))
        for key in ("num_experts_per_tok",):
            v = cfg.get(key) or cfg.get("text_config", {}).get(key)
            if v is not None:
                return v
    except Exception:
        pass
    return 8


def _read_routing_config(model_path=None):
    """config.json からルーティング方式を読み取る（Qwen系はsoftmax決め打ちの既定値）。

    moe_router_activation_func: "softmax"(既定) または "sigmoid"（Kimi/GLM/ERNIE等）
    routed_scaling_factor: ルーティング重みへの追加スケール（既定1.0）
    """
    try:
        cfg = json.load(open(os.path.join(model_path or MODEL_PATH, "config.json")))
        tc = cfg.get("text_config", cfg)
        activation = tc.get("moe_router_activation_func", "softmax") or "softmax"
        scale = tc.get("routed_scaling_factor", 1.0) or 1.0
        return activation, float(scale)
    except Exception:
        return "softmax", 1.0


def wire_streaming(
    model,
    capacity,
    top_k=None,
    perf=False,
    store_dir=None,
    model_path=None,
    model_type=None,
):
    """全層の mlp を StreamingMoE に差し替え、融合expertを解放。

    top_k=None の場合、config.json の num_experts_per_tok を自動検出。
    perf=True の場合、実効容量を 8000（≈13.5GB）に引き上げ。
    store_dir/model_path 未指定時はモジュール既定（resolve_model()の結果）を使う。
    model_type が "deepseek_v4" の場合は V4 専用パスを使用。
    """
    if model_type is None and model_path is not None:
        try:
            cfg = json.load(open(os.path.join(model_path, "config.json")))
            model_type = cfg.get("model_type")
        except Exception:
            pass
    if model_type == "deepseek_v4":
        return _wire_deepseek_v4(
            model,
            capacity,
            top_k=top_k,
            perf=perf,
            store_dir=store_dir,
            model_path=model_path,
        )
    if top_k is None:
        top_k = _read_top_k(model_path)
    # ELFMOON_TOP_K: 推論時 top_k 削減（opt-in 高速化・実測 80B top_k=4 で ~1.6x）。
    # 学習時 top_k を超える指定や 0 以下は無効。品質トレードオフがあるため既定は不変。
    _tk_env = os.environ.get("ELFMOON_TOP_K")
    if _tk_env:
        try:
            _tk = int(_tk_env)
            if 1 <= _tk < top_k:
                print(f"  top_k override: {top_k} → {_tk}（ELFMOON_TOP_K, 品質注意）")
                top_k = _tk
            elif _tk != top_k:
                print(f"  ELFMOON_TOP_K={_tk} は無効（有効範囲 1〜{top_k - 1}）: 無視")
        except ValueError:
            print(f"  ELFMOON_TOP_K={_tk_env!r} は数値でない: 無視")
    activation, routing_scale = _read_routing_config(model_path)
    if activation not in ("softmax", "sigmoid", "sqrtsoftplus"):
        raise ValueError(
            f"未対応のmoe_router_activation_func: {activation!r}（softmax/sigmoid/sqrtsoftplusのみ対応）"
        )
    store = ExpertStore(store_dir or STORE_DIR)
    gsc = None
    gsc_ne = int(os.environ.get("SSC", "0"))
    if gsc_ne > 0:
        init_layers = list(
            getattr(model, "layers", None)
            or getattr(model.model, "layers", None)
            or getattr(model.language_model, "layers", None)
            or []
        )
        try:
            cfg = json.load(open(os.path.join(model_path or MODEL_PATH, "config.json")))
            tc = cfg.get("text_config", cfg)
            d = tc.get("hidden_size", 2048)
            i = tc.get("intermediate_size", 512)
        except Exception:
            d, i = 2048, 512
        gsc = GlobalSlotCache(gsc_ne, store, n_layers=len(init_layers), dim=d, inter=i)
        pe = store.per_expert_bytes() / (1024 * 1024)
        print(
            f"  GlobalSlotCache: {gsc_ne}slots x {len(init_layers)}layers (~{gsc_ne * pe / 1024:.1f}GB)"
        )
    if perf:
        eff_cap = max(capacity, 8000)
        cache = ResidentCache(eff_cap)
        s = cache.stats()
        print(
            f"  性能モード: 実効容量 {s['capacity']}（{s['capacity'] * 1.69 / 1000:.1f}GB）"
        )
    else:
        cache = ResidentCache(capacity)
        s = cache.stats()
        print(
            f"  省メモリモード: 実効容量 {s['capacity']}（{s['capacity'] * 1.69 / 1000:.1f}GB）"
        )
    # プレフィル高速化: 元モデルの融合テンソルを mmap 直読みする（読めなければ従来経路）
    fused_store = None
    try:
        fused_store = FusedPrefillStore(model_path or MODEL_PATH)
        print(
            f"  プレフィル: gather_qmm 高速経路（融合テンソル {len(fused_store._layer_keys)}層）"
        )
    except Exception as e:
        print(f"  プレフィル: per-expert 経路（融合テンソル読込不可: {e}）")
    layers = (
        getattr(model, "layers", None)
        or getattr(model.model, "layers", None)
        or getattr(model.language_model, "layers", None)
    )
    n_dense = 0
    last_moe = None
    for l, layer in enumerate(layers):
        mlp = layer.mlp
        # Qwen MoE: mlp が直接 num_experts と gate を持つ（switch_mlp なし）
        if hasattr(mlp, "num_experts") and hasattr(mlp, "gate"):
            n_exp = mlp.num_experts
            gate = mlp.gate
            shared_exp = getattr(mlp, "shared_expert", None)
            shared_gate = getattr(mlp, "shared_expert_gate", None)
            correction_bias = getattr(mlp, "e_score_correction_bias", None)
            layer.mlp = StreamingMoE(
                l,
                gate,
                n_exp,
                top_k or mlp.top_k,
                store,
                cache,
                shared_exp=shared_exp,
                shared_gate=shared_gate,
                activation=activation,
                correction_bias=correction_bias,
                routing_scale=routing_scale,
                norm=mlp.norm_topk_prob,
                fused_store=fused_store,
                gsc=gsc,
            )
        elif hasattr(mlp, "switch_mlp"):
            n_exp = mlp.switch_mlp.gate_proj.weight.shape[0]
            gate = mlp.gate
            shared_exp = getattr(mlp, "shared_expert", None) or getattr(
                mlp, "shared_experts", None
            )
            shared_gate = getattr(mlp, "shared_expert_gate", None)
            correction_bias = getattr(mlp, "e_score_correction_bias", None)
            layer.mlp = StreamingMoE(
                l,
                gate,
                n_exp,
                top_k,
                store,
                cache,
                shared_exp=shared_exp,
                shared_gate=shared_gate,
                activation=activation,
                correction_bias=correction_bias,
                routing_scale=routing_scale,
                fused_store=fused_store,
                gsc=gsc,
            )
        else:
            n_dense += 1
            continue
        last_moe = layer.mlp
    if last_moe is not None:
        # プレフィル終了地点（最終MoE層）で融合テンソルのバッファを返却させる
        last_moe._is_last_moe = True
    if n_dense:
        print(f"  dense層{n_dense}個はストリーミング対象外のまま常駐（通常のMLP）")
    mx.clear_cache()

    # 起動時ウォームスタート: 前回セッション終了時の常駐セットを SSD から先読みし、
    # コールドスタート直後の命中率を前回定常値から開始する。
    # ELFMOON_PRIME: 0=無効 / 1=キャッシュ容量まで（既定） / N>1=上限N個
    _hotset_path = os.path.join(store_dir or STORE_DIR, "hotset.json")
    _prime = os.environ.get("ELFMOON_PRIME", "1")
    if _prime != "0" and os.path.exists(_hotset_path):
        try:
            import time as _time

            _t0 = _time.time()
            _keys = json.load(open(_hotset_path))
            _cap = cache.capacity if _prime == "1" else min(int(_prime), cache.capacity)
            _keys = _keys[-_cap:]  # 保存順は LRU→MRU。直近使用分を優先
            _pending = []
            for _le in _keys:
                _w = store.load(int(_le[0]), int(_le[1]))
                cache.prime((int(_le[0]), int(_le[1])), _w)
                _pending.extend(_w.values())
                # 遅延ロードを溜めすぎると fd 枯渇するため、こまめに実体化して閉じる
                if len(_pending) >= 384:
                    mx.eval(_pending)
                    _pending = []
            if _pending:
                mx.eval(_pending)
            print(
                f"  ウォームスタート: {len(_keys)} experts プライム（{_time.time() - _t0:.0f}秒）"
            )
        except Exception as _e:
            print(f"  ウォームスタート失敗（無視して続行）: {_e}")

    import atexit

    def _save_hotset(_cache=cache, _path=_hotset_path):
        try:
            _ks = [[int(k[0]), int(k[1])] for k in _cache._d.keys()]
            with open(_path, "w") as _f:
                json.dump(_ks, _f)
        except Exception:
            pass

    atexit.register(_save_hotset)
    return cache, store


def _wire_deepseek_v4(
    model, capacity, top_k=None, perf=False, store_dir=None, model_path=None
):
    if top_k is None:
        top_k = 6
    activation = "sqrtsoftplus"
    routing_scale = 1.5
    store = ExpertStore(store_dir or STORE_DIR)
    if perf:
        eff_cap = max(capacity, 8000)
        cache = ResidentCache(eff_cap)
        s = cache.stats()
        print(
            f"  性能モード: 実効容量 {s['capacity']}（{s['capacity'] * 1.69 / 1000:.1f}GB）"
        )
    else:
        cache = ResidentCache(capacity)
        s = cache.stats()
        print(
            f"  省メモリモード: 実効容量 {s['capacity']}（{s['capacity'] * 1.69 / 1000:.1f}GB）"
        )
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = getattr(model, "model", model).layers
    n_moe = 0
    for l, layer in enumerate(layers):
        moe = StreamingMoE(
            l,
            layer.gate,
            256,
            top_k,
            store,
            cache,
            shared_exp=None,
            shared_gate=None,
            activation=activation,
            correction_bias=None,
            routing_scale=routing_scale,
            group_size=32,
            mode="mxfp4",
        )
        layer.set_streaming_moe(moe)
        n_moe += 1
    mx.clear_cache()
    return cache, store


# ---- CLI ----

if __name__ == "__main__":
    import sys

    from mlx_lm import generate, load

    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 6144
    perf = "--perf" in sys.argv
    mode = "性能" if perf else "省メモリ"
    print(f"常駐容量={cap} experts（{mode}モード）")
    model, tok = load(MODEL_PATH, lazy=True)
    print("元モデル ロード完了（lazy）。ストリーミング化中...")
    cache, store = wire_streaming(model, cap, perf=perf)
    print(f"差し替え完了。常駐メモリ={mx.get_active_memory() / 1e9:.2f}GB")

    plen = sys.argv[2] if len(sys.argv) > 2 else "short"
    if plen == "long":
        ctx = "\n".join(
            f"func f{i}(_ x: Int) -> Int {{ return x * {i} + {i * i} }}"
            for i in range(40)
        )
        prompt = (
            ctx + "\n// 上記を踏まえ、Swiftで最大公約数gcd(_:_:)を書いて。コードのみ。"
        )
    else:
        prompt = "Write a Swift function gcd(_ a: Int, _ b: Int) -> Int. Code only."
    t = time.perf_counter()
    out = generate(model, tok, prompt=prompt, max_tokens=80, verbose=True)
    dt = time.perf_counter() - t
    print("=== 生成 ===")
    print(out)
    s = cache.stats()
    print(
        f"命中率={s['hit_rate'] * 100:.1f}% (hit={s['hits']} miss={s['misses']} 常駐={s['resident']})"
    )
    print(f"時間={dt:.1f}s")
