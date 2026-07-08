"""最終統合: mlx_lm の Qwen MoE 系モデルの各層MoEを ElfMoon ストリーミングMoEに差し替える。
融合 switch_mlp（Qwen3.6-35B で約17GB）を解放し、ExpertStore + ResidentCache から必要分だけ流す。
"""

import os
import time

import mlx.core as mx
import mlx.nn as nn

from expert_store import ExpertStore, GROUP, BITS
from resident_cache import ResidentCache

_HERE = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(_HERE, "spike/real_store")
GATE_DIR = os.path.join(_HERE, "spike/real_gates")
MODEL_PATH = os.path.join(_HERE, "..", "models", "qwen3.6-35b-mlx")


# ---- A: Compiled MoE decode ----


@mx.compile
def _decode_moe(
    x: mx.array,
    w_gw: mx.array,
    s_gw: mx.array,
    b_gw: mx.array,
    w_up: mx.array,
    s_up: mx.array,
    b_up: mx.array,
    w_dw: mx.array,
    s_dw: mx.array,
    b_dw: mx.array,
    weights: mx.array,
    top_k: int,
    shared=None,
):
    xb = mx.broadcast_to(x, (top_k, 1, x.shape[-1]))
    g = mx.quantized_matmul(
        xb, w_gw, s_gw, b_gw, transpose=True, group_size=GROUP, bits=BITS
    )
    u = mx.quantized_matmul(
        xb, w_up, s_up, b_up, transpose=True, group_size=GROUP, bits=BITS
    )
    h = (g * mx.sigmoid(g)) * u
    yo = mx.quantized_matmul(
        h, w_dw, s_dw, b_dw, transpose=True, group_size=GROUP, bits=BITS
    )
    result = (yo[:, 0, :] * weights[:, None]).sum(0)
    if shared is not None:
        (
            sg_w,
            sg_s,
            sg_b,
            sg_bits,
            sg_gs,
            se_gw,
            se_gs,
            se_gb,
            se_uw,
            se_us,
            se_ub,
            se_dw,
            se_ds,
            se_db,
        ) = shared
        sg = mx.quantized_matmul(
            x, sg_w, sg_s, sg_b, transpose=True, group_size=sg_gs, bits=sg_bits
        )
        se_g = mx.quantized_matmul(
            x, se_gw, se_gs, se_gb, transpose=True, group_size=GROUP, bits=BITS
        )
        se_u = mx.quantized_matmul(
            x, se_uw, se_us, se_ub, transpose=True, group_size=GROUP, bits=BITS
        )
        se_h = (se_g * mx.sigmoid(se_g)) * se_u
        se_out = mx.quantized_matmul(
            se_h, se_dw, se_ds, se_db, transpose=True, group_size=GROUP, bits=BITS
        )
        result = result + mx.sigmoid(sg) * se_out
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
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.gate = gate
        self.n_experts = n_experts
        self.top_k = top_k
        self._store = store
        self._cache = cache
        self.norm = norm

        if shared_exp is not None:
            sg = shared_gate
            se = shared_exp
            self._shared = (
                sg.weight,
                sg.scales,
                sg.biases,
                sg.bits,
                sg.group_size,
                se.gate_proj.weight,
                se.gate_proj.scales,
                se.gate_proj.biases,
                se.up_proj.weight,
                se.up_proj.scales,
                se.up_proj.biases,
                se.down_proj.weight,
                se.down_proj.scales,
                se.down_proj.biases,
            )
        else:
            self._shared = None

    def __call__(self, x):
        shp = x.shape
        xf = x.reshape(-1, shp[-1])
        logits = self.gate(xf).astype(mx.float32)
        probs = mx.softmax(logits, axis=-1)
        idx = mx.argpartition(-probs, self.top_k - 1, axis=-1)[:, : self.top_k]
        w = mx.take_along_axis(probs, idx, axis=-1)
        if self.norm:
            w = w / mx.sum(w, axis=-1, keepdims=True)
        idx_l = idx.tolist()
        N = xf.shape[0]

        def load(e):
            return self._cache.get(
                (self.layer_idx, e), lambda e=e: self._store.load(self.layer_idx, e)
            )

        if N == 1:
            experts = [load(e) for e in idx_l[0]]

            w_gw = mx.stack([e["gate.wq"] for e in experts])
            s_gw = mx.stack([e["gate.s"] for e in experts])
            b_gw = mx.stack([e["gate.b"] for e in experts])
            w_up = mx.stack([e["up.wq"] for e in experts])
            s_up = mx.stack([e["up.s"] for e in experts])
            b_up = mx.stack([e["up.b"] for e in experts])
            w_dw = mx.stack([e["down.wq"] for e in experts])
            s_dw = mx.stack([e["down.s"] for e in experts])
            b_dw = mx.stack([e["down.b"] for e in experts])

            weights = w[0].astype(mx.float16)
            result = _decode_moe(
                xf[0:1],
                w_gw,
                s_gw,
                b_gw,
                w_up,
                s_up,
                b_up,
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

        token_buf = [None] * N
        for e, items in expert_groups.items():
            exp = load(e)
            indices = [it[0] for it in items]
            weights = [it[1] for it in items]
            xb = xf[mx.array(indices)]
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
            wv = mx.array(weights).astype(yo.dtype)
            contrib = yo * wv[:, None]
            for i, t_idx in enumerate(indices):
                if token_buf[t_idx] is None:
                    token_buf[t_idx] = contrib[i]
                else:
                    token_buf[t_idx] = token_buf[t_idx] + contrib[i]
        out = mx.stack(token_buf).reshape(shp)
        if self._shared is not None:
            (
                sg_w,
                sg_s,
                sg_b,
                sg_bits,
                sg_gs,
                se_gw,
                se_gs,
                se_gb,
                se_uw,
                se_us,
                se_ub,
                se_dw,
                se_ds,
                se_db,
            ) = self._shared
            sg = mx.quantized_matmul(
                xf, sg_w, sg_s, sg_b, transpose=True, group_size=sg_gs, bits=sg_bits
            )
            se_g = mx.quantized_matmul(
                xf, se_gw, se_gs, se_gb, transpose=True, group_size=GROUP, bits=BITS
            )
            se_u = mx.quantized_matmul(
                xf, se_uw, se_us, se_ub, transpose=True, group_size=GROUP, bits=BITS
            )
            se_h = (se_g * mx.sigmoid(se_g)) * se_u
            se_out = mx.quantized_matmul(
                se_h, se_dw, se_ds, se_db, transpose=True, group_size=GROUP, bits=BITS
            )
            out = out + mx.sigmoid(sg) * se_out
        return out.astype(x.dtype)


# ---- Wiring ----


def wire_streaming(model, capacity, top_k=8, perf=False):
    """全層の mlp を StreamingMoE に差し替え、融合expertを解放。

    perf=True の場合、実効容量を 8000（≈13.5GB）に引き上げ。
    """
    store = ExpertStore(STORE_DIR)
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
    for l, layer in enumerate(layers):
        mlp = layer.mlp
        n_exp = mlp.switch_mlp.gate_proj.weight.shape[0]
        gate = mlp.gate
        shared_exp = getattr(mlp, "shared_expert", None)
        shared_gate = getattr(mlp, "shared_expert_gate", None)
        layer.mlp = StreamingMoE(
            l,
            gate,
            n_exp,
            top_k,
            store,
            cache,
            shared_exp=shared_exp,
            shared_gate=shared_gate,
        )
    mx.clear_cache()
    return cache, store


# ---- CLI ----

if __name__ == "__main__":
    import sys
    from mlx_lm import load, generate

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
