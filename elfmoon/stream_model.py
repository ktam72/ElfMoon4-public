"""最終統合: mlx_lm の Qwen3-Coder の各層MoEを ElfMoon ストリーミングMoEに差し替える。
融合 switch_mlp(16GB) を解放し、ExpertStore + ResidentCache から必要分だけ流す。
"""
import time
import mlx.core as mx
import mlx.nn as nn
from expert_store import ExpertStore, expert_ffn
from resident_cache import ResidentCache

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(_HERE, "spike/real_store")
GATE_DIR = os.path.join(_HERE, "spike/real_gates")
MODEL_PATH = os.path.join(_HERE, "..", "models", "qwen3-coder-mlx")


class StreamingMoE(nn.Module):
    """層の融合MoEを置換。routerは元の量子化gateを流用、expertはストア/キャッシュから。"""

    def __init__(self, layer_idx, gate, n_experts, top_k, store, cache, norm=True):
        super().__init__()
        self.layer_idx = layer_idx
        self.gate = gate            # 元の router（8bit量子化Linear）を流用
        self.n_experts = n_experts
        self.top_k = top_k
        self._store = store         # _始まりでMLXのparam走査から除外
        self._cache = cache
        self.norm = norm

    def __call__(self, x):
        shp = x.shape
        xf = x.reshape(-1, shp[-1])                 # [N, D]
        logits = self.gate(xf).astype(mx.float32)   # [N, E]
        probs = mx.softmax(logits, axis=-1)
        idx = mx.argpartition(-probs, self.top_k - 1, axis=-1)[:, :self.top_k]
        w = mx.take_along_axis(probs, idx, axis=-1)  # [N, k]
        if self.norm:
            w = w / mx.sum(w, axis=-1, keepdims=True)
        idx_l = idx.tolist()
        w_l = w.tolist()
        outs = []
        for n in range(xf.shape[0]):
            acc = mx.zeros((shp[-1],))
            xn = xf[n][None]
            for j in range(self.top_k):
                e = int(idx_l[n][j])
                wt = self._cache.get((self.layer_idx, e),
                                     lambda l=self.layer_idx, e=e: self._store.load(l, e))
                acc = acc + w_l[n][j] * expert_ffn(xn, wt)[0]
            outs.append(acc)
        return mx.stack(outs).reshape(shp).astype(x.dtype)


def wire_streaming(model, capacity, top_k=8):
    """全層の mlp を StreamingMoE に差し替え、融合expertを解放。"""
    store = ExpertStore(STORE_DIR)
    cache = ResidentCache(capacity)
    layers = model.model.layers
    for l, layer in enumerate(layers):
        mlp = layer.mlp
        n_exp = mlp.switch_mlp.gate_proj.weight.shape[0]
        gate = mlp.gate
        layer.mlp = StreamingMoE(l, gate, n_exp, top_k, store, cache)
    mx.clear_cache()      # 解放された融合expertのメモリを回収
    return cache, store


if __name__ == "__main__":
    import sys
    from mlx_lm import load, generate
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 1200   # 常駐expert数
    print(f"常駐容量={cap} experts (~{cap*2.65/1000:.1f}GB)")
    model, tok = load(MODEL_PATH)
    print("元モデル ロード完了。ストリーミング化中...")
    cache, store = wire_streaming(model, cap)
    print(f"差し替え完了。常駐メモリ={mx.get_active_memory()/1e9:.2f}GB")

    prompt = "Write a Swift function gcd(_ a: Int, _ b: Int) -> Int. Code only."
    t = time.perf_counter()
    out = generate(model, tok, prompt=prompt, max_tokens=80, verbose=False)
    dt = time.perf_counter() - t
    print("=== 生成 ===")
    print(out)
    s = cache.stats()
    print(f"命中率={s['hit_rate']*100:.1f}% (hit={s['hits']} miss={s['misses']} 常駐={s['resident']})")
    print(f"時間={dt:.1f}s")
