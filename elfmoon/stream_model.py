"""最終統合: mlx_lm の Qwen MoE 系モデルの各層MoEを ElfMoon ストリーミングMoEに差し替える。
融合 switch_mlp（Qwen3.6-35B で約17GB）を解放し、ExpertStore + ResidentCache から必要分だけ流す。
"""

import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
from expert_store import BITS, GROUP, ExpertStore
from resident_cache import ResidentCache

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_NAME = "qwen3.6-35b-mlx"

# モデル置き場のルート。任意ディレクトリ（外部SSD等）を指せる唯一の結合点。
MODELS_ROOT = os.environ.get("ELFMOON_MODELS_ROOT", os.path.join(_HERE, "..", "models"))


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
    """MODELS_ROOT 直下で config.json を持つディレクトリをモデルとして列挙する。"""
    if not os.path.isdir(MODELS_ROOT):
        return []
    names = []
    for entry in sorted(os.listdir(MODELS_ROOT)):
        d = os.path.join(MODELS_ROOT, entry)
        if os.path.isfile(os.path.join(d, "config.json")):
            has_store = os.path.isdir(os.path.join(d, "store"))
            names.append((entry, has_store))
    return names


# 後方互換: モジュールレベル定数（--model 未指定・env var 未設定時は既定モデル）
MODEL_PATH, STORE_DIR = resolve_model()


# ---- A: Compiled MoE decode ----


@mx.compile
def _decode_moe(
    x: mx.array,
    w_gu: mx.array,
    s_gu: mx.array,
    b_gu: mx.array,
    w_dw: mx.array,
    s_dw: mx.array,
    b_dw: mx.array,
    weights: mx.array,
    top_k: int,
    shared=None,
):
    xb = mx.broadcast_to(x, (2 * top_k, 1, x.shape[-1]))
    gu = mx.quantized_matmul(
        xb, w_gu, s_gu, b_gu, transpose=True, group_size=GROUP, bits=BITS
    )
    g, u = gu[:top_k], gu[top_k:]
    h = (g * mx.sigmoid(g)) * u
    yo = mx.quantized_matmul(
        h, w_dw, s_dw, b_dw, transpose=True, group_size=GROUP, bits=BITS
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
            x, se_guw, se_gus, se_gub, transpose=True, group_size=GROUP, bits=BITS
        )
        se_g, se_u = se_gu[:, : se_gu.shape[-1] // 2], se_gu[:, se_gu.shape[-1] // 2 :]
        se_h = (se_g * mx.sigmoid(se_g)) * se_u
        se_out = mx.quantized_matmul(
            se_h, se_dw, se_ds, se_db, transpose=True, group_size=GROUP, bits=BITS
        )
        if gated:
            sg = mx.quantized_matmul(
                x, sg_w, sg_s, sg_b, transpose=True, group_size=sg_gs, bits=sg_bits
            )
            result = result + mx.sigmoid(sg) * se_out
        else:
            result = result + se_out
    return result


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
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.gate = gate
        self.n_experts = n_experts
        self.top_k = top_k
        self._store = store
        self._cache = cache
        self.norm = norm
        self.activation = activation
        self.correction_bias = correction_bias
        self.routing_scale = routing_scale

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
        else:
            self._shared = None

    def __call__(self, x):
        shp = x.shape
        xf = x.reshape(-1, shp[-1])
        logits = self.gate(xf)
        if self.activation == "sigmoid":
            probs = mx.sigmoid(logits)
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
        mx.eval(idx, w)
        idx_l = idx.tolist()
        N = xf.shape[0]

        def load(e):
            return self._cache.get(
                (self.layer_idx, e), lambda e=e: self._store.load(self.layer_idx, e)
            )

        if N == 1:
            experts = [load(e) for e in idx_l[0]]

            w_gu = mx.stack(
                [e["gate.wq"] for e in experts] + [e["up.wq"] for e in experts]
            )
            s_gu = mx.stack(
                [e["gate.s"] for e in experts] + [e["up.s"] for e in experts]
            )
            b_gu = mx.stack(
                [e["gate.b"] for e in experts] + [e["up.b"] for e in experts]
            )
            w_dw = mx.stack([e["down.wq"] for e in experts])
            s_dw = mx.stack([e["down.s"] for e in experts])
            b_dw = mx.stack([e["down.b"] for e in experts])

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
            ).reshape(shp)
            return result.astype(x.dtype)

        # --- プレフィル(N>1): Expert単位でトークンをバッチ処理 ---
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
            g = mx.quantized_matmul(
                xb,
                exp["gate.wq"],
                exp["gate.s"],
                exp["gate.b"],
                transpose=True,
                group_size=GROUP,
                bits=BITS,
            )
            u = mx.quantized_matmul(
                xb,
                exp["up.wq"],
                exp["up.s"],
                exp["up.b"],
                transpose=True,
                group_size=GROUP,
                bits=BITS,
            )
            h = (g * mx.sigmoid(g)) * u
            yo = mx.quantized_matmul(
                h,
                exp["down.wq"],
                exp["down.s"],
                exp["down.b"],
                transpose=True,
                group_size=GROUP,
                bits=BITS,
            )
            contrib = yo * weights[:, None].astype(yo.dtype)
            out = out.at[indices].add(contrib)
        out = out.reshape(shp)
        if self._shared is not None:
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
            se_out = mx.quantized_matmul(
                se_act, se_dw, se_ds, se_db, transpose=True, group_size=GROUP, bits=BITS
            )
            if gated:
                sg = mx.quantized_matmul(
                    xf, sg_w, sg_s, sg_b, transpose=True, group_size=sg_gs, bits=sg_bits
                )
                out = out + mx.sigmoid(sg) * se_out
            else:
                out = out + se_out
        return out.astype(x.dtype)


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
    model, capacity, top_k=None, perf=False, store_dir=None, model_path=None
):
    """全層の mlp を StreamingMoE に差し替え、融合expertを解放。

    top_k=None の場合、config.json の num_experts_per_tok を自動検出。
    perf=True の場合、実効容量を 8000（≈13.5GB）に引き上げ。
    store_dir/model_path 未指定時はモジュール既定（resolve_model()の結果）を使う。
    """
    if top_k is None:
        top_k = _read_top_k(model_path)
    activation, routing_scale = _read_routing_config(model_path)
    if activation not in ("softmax", "sigmoid"):
        raise ValueError(
            f"未対応のmoe_router_activation_func: {activation!r}（softmax/sigmoidのみ対応）"
        )
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
    layers = getattr(model, "layers", None) or model.model.layers
    n_dense = 0
    for l, layer in enumerate(layers):
        mlp = layer.mlp
        if not hasattr(mlp, "switch_mlp"):
            # first_k_dense_replace 等でdense層(MoE不使用)が混在するモデル向け。
            # 通常のMLPのまま常駐させ、ストリーミング化はスキップする。
            n_dense += 1
            continue
        n_exp = mlp.switch_mlp.gate_proj.weight.shape[0]
        gate = mlp.gate
        # 属性名はモデルにより単数/複数が異なる（Qwen: shared_expert, GLM/Kimi: shared_experts）
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
        )
    if n_dense:
        print(f"  dense層{n_dense}個はストリーミング対象外のまま常駐（通常のMLP）")
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
