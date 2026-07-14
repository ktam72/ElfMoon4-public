"""実重み統合（mlx_lm非依存）: MLX版 Qwen MoE 系の safetensors を直接読み、
融合 switch_mlp を per-expert に分解して ExpertStore 形式で保存する。

キー構造（prefix は _detect_prefix で自動検出。例: 'model', 'language_model.model'）:
  {prefix}.layers.{l}.mlp.gate.{weight,scales,biases}                 ルーター(8bit)
  {prefix}.layers.{l}.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}  融合expert(4bit,[n_experts,...])
量子化パラメータ: expert=group64/bit4（ExpertStoreと一致→スライスのみ）, gate=group64/bit8。
"""

import json
import os
import sys

import mlx.core as mx

GROUP = 64
GATE_BITS = 8


PREFIX: list[str] = [""]  # mutable container for prefix, set in split_all


def _detect_prefix(W):
    """Detect key prefix before '.layers.N' (e.g. '' or 'language_model.model')."""
    for k in W:
        if ".mlp.gate." in k:
            return k.split(".layers")[0].rstrip(".")
    return "model"


def _base(l):
    p = PREFIX[0]
    return f"{p + '.' if p else ''}layers.{l}.mlp"


def load_shards(path):
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    W = {}
    for shard in sorted(set(idx["weight_map"].values())):
        W.update(mx.load(os.path.join(path, shard)))  # mmap
    return W


def router_gate_float(W, l):
    p = f"{_base(l)}.gate"
    return mx.dequantize(
        W[f"{p}.weight"],
        W[f"{p}.s" if f"{p}.s" in W else f"{p}.scales"],
        W[f"{p}.biases"],
        group_size=GROUP,
        bits=GATE_BITS,
    )


def split_layer(W, l, store_dir):
    os.makedirs(store_dir, exist_ok=True)
    b = _base(l)
    # 融合expertを取得
    projs = {
        name: (
            W[f"{b}.switch_mlp.{name}_proj.weight"],
            W[f"{b}.switch_mlp.{name}_proj.scales"],
            W[f"{b}.switch_mlp.{name}_proj.biases"],
        )
        for name in ("gate", "up", "down")
    }
    n_exp = projs["gate"][0].shape[0]
    for e in range(n_exp):
        d = {}
        for name, (w, s, b) in projs.items():
            d[f"{name}.wq"], d[f"{name}.s"], d[f"{name}.b"] = w[e], s[e], b[e]
        mx.save_safetensors(os.path.join(store_dir, f"l{l}_e{e}.safetensors"), d)
    mx.eval()
    return n_exp


def verify_layer0(path, store_dir):
    """分解の往復が量子化を保つか検証: 保存前スライス == 保存→ロード。

    store_dir は必須（デフォルト値を持たせない）。過去に危険なデフォルト値
    （他モデルのstoreを指す固定パス）が原因で本番storeを誤って上書きした事故があるため。
    """
    from expert_store import ExpertStore, expert_ffn

    W = load_shards(path)
    PREFIX[0] = _detect_prefix(W)  # split_layer が参照する prefix を設定
    n = split_layer(W, 0, store_dir)
    store = ExpertStore(store_dir)
    base = f"{_base(0)}.switch_mlp"
    hidden = router_gate_float(W, 0).shape[-1]
    x = mx.random.normal((1, hidden))
    max_err = 0.0
    for e in (0, 7, 63, 127):
        # 直接（融合テンソルからスライス）した参照
        ref = {}
        for k in ("gate", "up", "down"):
            ref[f"{k}.wq"] = W[f"{base}.{k}_proj.weight"][e]
            ref[f"{k}.s"] = W[f"{base}.{k}_proj.scales"][e]
            ref[f"{k}.b"] = W[f"{base}.{k}_proj.biases"][e]
        y_ref = expert_ffn(x, ref)
        y_store = expert_ffn(x, store.load(0, e))
        err = float(mx.max(mx.abs(y_ref - y_store)))
        max_err = max(max_err, err)
    print(
        f"[分解往復検証] layer0 {n}experts, 誤差={max_err:.2e} "
        f"{'OK' if max_err == 0 else ('OK(≈0)' if max_err < 1e-5 else 'NG')}"
    )
    print(f"expert1個: {store.per_expert_bytes() / 1e6:.2f} MB")


if __name__ == "__main__":
    path = sys.argv[2] if len(sys.argv) > 2 else "../models/qwen3.6-35b-mlx"
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    # オプション: 第3引数 = store_dir（未指定時はモデル直下の store/ … 規約）
    store_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(path, "store")
    if cmd == "verify":
        verify_layer0(path, store_dir)
    elif cmd == "split_all":
        W = load_shards(path)
        PREFIX[0] = _detect_prefix(W)
        pfx = PREFIX[0]
        print(f"検出: prefix={pfx!r}, ", end="", flush=True)
        n_layers = 1 + max(int(k.split(".layers.")[1].split(".")[0]) for k in W if f"{pfx}.layers." in k)
        print(f"layers={n_layers}", flush=True)
        n_dense = 0
        for l in range(n_layers):
            base = f"{pfx + '.' if pfx else ''}layers.{l}.mlp"
            if f"{base}.switch_mlp.gate_proj.weight" not in W:
                # first_k_dense_replace 等で dense層(switch_mlpなし)が混在するモデル向け
                n_dense += 1
                continue
            ne = split_layer(W, l, store_dir)
            if l % 8 == 0:
                print(f"  layer {l}/{n_layers} 分解済(experts={ne})", flush=True)
        print(f"完了: {n_layers}層（dense層{n_dense}個はスキップ）")
