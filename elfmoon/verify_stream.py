"""ストリーミングMoEが元のmlpと一致するか、実モデルで層単位検証する。
ずれていれば我々の実装バグ。一致すれば出力品質の問題は別要因。

先に integrate.py split_all <model_dir> でモデル直下の store/ を生成しておくこと。
対象モデルは ELFMOON_MODEL（既定 qwen3.6-35b-mlx）/ ELFMOON_MODELS_ROOT で指定。
"""

import json, os, mlx.core as mx
from mlx_lm import load
from stream_model import StreamingMoE, MODEL_PATH, STORE_DIR
from expert_store import ExpertStore
from resident_cache import ResidentCache

model, tok = load(MODEL_PATH, lazy=True)
# config.json から top_k を自動検出（未指定時は8＝35B互換）
cfg = json.load(open(os.path.join(MODEL_PATH, "config.json")))
top_k = cfg.get("num_experts_per_tok", 8)
layers = getattr(model, "layers", None) or model.model.layers
store = ExpertStore(STORE_DIR)
cache = ResidentCache(10240)

g0 = layers[0].mlp.gate
D = g0.weight.shape[1] * (32 // g0.bits) if hasattr(g0, "bits") else g0.weight.shape[1]
n_layers = len(layers)
print(f"layers={n_layers}, hidden={D}, top_k={top_k}")

for l in sorted({0, 1, n_layers // 2, n_layers - 1}):
    orig = layers[l].mlp
    mine = StreamingMoE(
        l,
        orig.gate,
        orig.switch_mlp.gate_proj.weight.shape[0],
        top_k,
        store,
        cache,
        shared_exp=getattr(orig, "shared_expert", None),
        shared_gate=getattr(orig, "shared_expert_gate", None),
    )
    x = mx.random.normal((1, 3, D)).astype(mx.float16)
    yo = orig(x)
    ym = mine(x)
    err = float(mx.max(mx.abs(yo.astype(mx.float32) - ym.astype(mx.float32))))
    rel = err / (float(mx.max(mx.abs(yo.astype(mx.float32)))) + 1e-9)
    print(
        f"layer {l:2d}: 最大誤差={err:.4e} 相対={rel:.4e} "
        f"{'OK' if rel < 1e-2 else 'NG(ずれ)'}"
    )
