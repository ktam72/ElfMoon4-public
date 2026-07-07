"""実重み統合（mlx_lm非依存）: MLX版Qwen3-Coder の safetensors を直接読み、
融合 switch_mlp を per-expert に分解して ExpertStore 形式で保存する。

キー構造（確認済）:
  model.layers.{l}.mlp.gate.{weight,scales,biases}                 ルーター(8bit)
  model.layers.{l}.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}  融合expert(4bit,[128,...])
量子化パラメータ: expert=group64/bit4（ExpertStoreと一致→スライスのみ）, gate=group64/bit8。
"""

import sys, os, json
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


def _base(l):
    p = PREFIX[0]
    return f"{p + '.' if p else ''}layers.{l}.mlp"


def router_gate_float(W, l):
    p = f"{_base(l)}.gate"
    return mx.dequantize(
        W[f"{p}.weight"],
        W[f"{p}.s" if f"{p}.s" in W else f"{p}.scales"],
        W[f"{p}.biases"],
        group_size=GROUP,
        bits=GATE_BITS,
    )


def split_layer(W, l, store_dir, gate_dir):
    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(gate_dir, exist_ok=True)
    b = _base(l)
    # ルーターgate（floatに復元）
    g = mx.dequantize(
        W[f"{b}.gate.weight"],
        W[f"{b}.gate.scales"],
        W[f"{b}.gate.biases"],
        group_size=GROUP,
        bits=GATE_BITS,
    )
    mx.save_safetensors(os.path.join(gate_dir, f"gate_l{l}.safetensors"), {"w": g})
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


def verify_layer0(path):
    """分解の往復が量子化を保つか検証: 保存前スライス == 保存→ロード。"""
    from expert_store import ExpertStore, expert_ffn

    W = load_shards(path)
    store_dir, gate_dir = "spike/real_store", "spike/real_gates"
    pfx = _detect_prefix(W)
    n = split_layer(W, 0, store_dir, gate_dir)
    store = ExpertStore(store_dir)
    base = f"{pfx}.layers.0.mlp.switch_mlp"
    x = mx.random.normal((1, 2048))
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
    path = sys.argv[2] if len(sys.argv) > 2 else "../models/qwen3-30b-instruct-mlx"
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "verify":
        verify_layer0(path)
    elif cmd == "split_all":
        W = load_shards(path)
        PREFIX[0] = _detect_prefix(W)
        pfx = PREFIX[0]
        print(f"検出: prefix={pfx!r}, ", end="", flush=True)
        n_layers = 1 + max(
            int(k.split(".layers.")[1].split(".")[0])
            for k in W
            if f"{pfx}.layers." in k
        )
        print(f"layers={n_layers}", flush=True)
        for l in range(n_layers):
            ne = split_layer(W, l, "spike/real_store", "spike/real_gates")
            if l % 8 == 0:
                print(f"  layer {l}/{n_layers} 分解済(experts={ne})", flush=True)
        print(f"完了: {n_layers}層")
