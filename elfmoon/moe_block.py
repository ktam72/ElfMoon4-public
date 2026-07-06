"""モジュール③: ハイブリッド MoE ブロック（mlx-lm の SwitchGLU を置換）。

デコード（batch=1）を主対象に:
  router(top-k) → 各activeexpertを常駐キャッシュ/ストアから取得 → FFN → スコア加重和。
ホット=キャッシュ命中（将来は融合バッチ計算）、コールド=ストアから個別ロード。
v1 は正しさ優先で per-expert ループ（k=8程度なので実害小）。
"""
import mlx.core as mx
try:
    from .expert_store import expert_ffn
except ImportError:
    from expert_store import expert_ffn


def route(x, gate_w, top_k, norm=True):
    """x:[dim], gate_w:[n_experts, dim] → (idx[top_k], weight[top_k])。"""
    logits = gate_w @ x                      # [n_experts]
    probs = mx.softmax(logits, axis=-1)
    idx = mx.argpartition(-probs, top_k - 1)[:top_k]
    w = probs[idx]
    if norm:
        w = w / mx.sum(w)
    return idx, w


class MoEBlock:
    def __init__(self, layer_idx, gate_w, n_experts, top_k, store, cache):
        self.layer = layer_idx
        self.gate_w = gate_w
        self.n_experts = n_experts
        self.top_k = top_k
        self.store = store
        self.cache = cache

    def __call__(self, x):
        """x:[dim] → y:[dim]。"""
        idx, w = route(x, self.gate_w, self.top_k)
        idx = [int(i) for i in idx.tolist()]
        xr = x[None]                          # [1, dim]
        y = mx.zeros_like(x)
        for e, weight in zip(idx, w):
            wt = self.cache.get((self.layer, e),
                                lambda e=e: self.store.load(self.layer, e))
            y = y + weight * expert_ffn(xr, wt)[0]
        return y


def reference_moe(x, gate_w, experts, top_k):
    """全expertメモリ常駐の素朴実装（正しさ比較用）。experts: {e: weights}。"""
    idx, w = route(x, gate_w, top_k)
    idx = [int(i) for i in idx.tolist()]
    xr = x[None]
    y = mx.zeros_like(x)
    for e, weight in zip(idx, w):
        y = y + weight * expert_ffn(xr, experts[e])[0]
    return y
