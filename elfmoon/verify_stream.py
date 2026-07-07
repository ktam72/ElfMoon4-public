"""ストリーミングMoEが元のmlpと一致するか、実モデルで層単位検証する。
ずれていれば我々の実装バグ。一致すれば言語問題は別要因。
"""
import mlx.core as mx
from mlx_lm import load
from stream_model import StreamingMoE, MODEL_PATH
from expert_store import ExpertStore
from resident_cache import ResidentCache

model, tok = load(MODEL_PATH)
layers = model.model.layers
store = ExpertStore("spike/real_store")
cache = ResidentCache(200000)   # 全部載る大容量（キャッシュ影響を排除）

D = model.model.layers[0].mlp.gate.weight.shape[1] if hasattr(
    model.model.layers[0].mlp.gate, "weight") else 2048

for l in (0, 1, 24, 47):
    orig = layers[l].mlp                       # 元の融合MoE
    mine = StreamingMoE(l, orig.gate,
                        orig.switch_mlp.gate_proj.weight.shape[0],
                        8, store, cache)
    x = mx.random.normal((1, 3, 2048)).astype(mx.float16)   # [B,T,D]
    yo = orig(x)
    ym = mine(x)
    err = float(mx.max(mx.abs(yo.astype(mx.float32) - ym.astype(mx.float32))))
    rel = err / (float(mx.max(mx.abs(yo.astype(mx.float32)))) + 1e-9)
    print(f"layer {l:2d}: 最大誤差={err:.4e} 相対={rel:.4e} "
          f"{'OK' if rel < 1e-2 else 'NG(ずれ)'}")
