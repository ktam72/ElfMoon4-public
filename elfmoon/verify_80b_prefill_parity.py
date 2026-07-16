"""80B prefill gather_qmm logits パリティ検証。

同一プロンプト（~1000tok）を (a) fused 経路と (b) per-expert 経路で
プレフィル → decode 先頭数 token の logits を比較する。

実行: ELFMOON_MODEL=qwen3-next-80b-mlx python3 elfmoon/verify_80b_prefill_parity.py
"""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx
from mlx_lm import load
from stream_model import MODEL_PATH, wire_streaming

PROMPT = (
    "\n".join(
        f"func f{i}(_ x: Int) -> Int {{ return x * {i} + {i * i} }}" for i in range(50)
    )
    + "\n// Write gcd in Swift. Code only."
)


def run(mode, disable_fused=False):
    os.environ["SSC"] = "0"
    mx.clear_cache()
    model, tok = load(MODEL_PATH, lazy=True)
    cache, store = wire_streaming(model, 6144)
    mx.eval()

    if disable_fused:
        layers = (
            getattr(model, "layers", None)
            or getattr(model.model, "layers", None)
            or getattr(model.language_model, "layers", None)
        )
        n = 0
        for l in layers:
            mlp = getattr(l, "mlp", None)
            if mlp and hasattr(mlp, "_fused_store") and mlp._fused_store is not None:
                mlp._fused_store = None
                n += 1
        print(f"  {mode}: disabled fused store for {n} MoE layers", flush=True)

    ids = tok.encode(PROMPT)
    print(f"  {mode}: prompt={len(ids)} tokens", flush=True)

    # Prefill (run the model once to process all prompt tokens)
    t0 = time.perf_counter()
    logits = model(mx.array([ids]))
    mx.eval(logits)
    dt = time.perf_counter() - t0
    print(f"  {mode}: prefill {dt:.2f}s ({len(ids) / dt:.0f} tok/s)", flush=True)

    # Decode a few steps
    all_ids = list(ids)
    first_logits = None
    for step in range(8):
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        all_ids.append(int(nxt))
        if step == 0:
            first_logits = logits[:, -1, :] + 0
        logits = model(mx.array([[all_ids[-1]]]))
    mx.eval(logits)

    mem = mx.get_active_memory() / 1e9
    print(f"  {mode}: mem={mem:.2f}GB", flush=True)
    return first_logits


if __name__ == "__main__":
    print("=== 80B prefill logits parity: fused vs per-expert ===", flush=True)

    # (a) Fused path (default)
    fused_logits = run("fused", disable_fused=False)

    print()

    # (b) Per-expert path (fused disabled)
    per_exp_logits = run("per-expert", disable_fused=True)

    # Compare
    print()
    print("=== Comparison ===", flush=True)
    fused_argmax = mx.argmax(fused_logits, axis=-1)
    per_exp_argmax = mx.argmax(per_exp_logits, axis=-1)
    mx.eval(fused_argmax, per_exp_argmax)

    argmax_match = int(mx.sum((fused_argmax == per_exp_argmax).astype(mx.int32)))
    total = fused_argmax.size
    print(f"  argmax match: {argmax_match}/{total} ({argmax_match / total * 100:.1f}%)")

    max_err = float(
        mx.max(
            mx.abs(fused_logits.astype(mx.float32) - per_exp_logits.astype(mx.float32))
        )
    )
    print(f"  max logits error: {max_err:.2e}")

    if argmax_match == total and max_err < 1e-3:
        print("  ✅ PARITY PASS: fused == per-expert (argmax一致, err<1e-3)")
    elif argmax_match == total:
        print(f"  ⚠️ argmax一致だが誤差{max_err:.2e} > 1e-3 — 要確認")
    else:
        print(f"  ❌ PARITY FAIL: {total - argmax_match}個のargmax不一致")
