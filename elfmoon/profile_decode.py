"""Per-token decode breakdown profiler for ElfMoon4.

Measures time distribution within each decode step:
  - Routing (gate -> softmax -> argpartition -> tolist)
  - Expert stacking (cache.get -> mx.stack)
  - MoE matmul (_decode_moe compiled call, includes shared expert)
  - Attention + norms + residuals + extra (everything outside MoE)

Usage:
  cd elfmoon && python3 profile_decode.py [capacity] [--perf]
"""

import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from stream_model import (
    StreamingMoE,
    _decode_moe,
    resolve_model,
    wire_streaming,
)


class PerfCounters:
    def reset(self):
        for k in (
            "routing_s",
            "stacking_s",
            "moe_s",
            "full_token_s",
            "n_routing",
            "n_stacking",
            "n_moe",
            "n_tokens",
        ):
            setattr(self, k, 0.0)

    def __init__(self):
        self.reset()

    def record_routing(self, dt):
        self.routing_s += dt
        self.n_routing += 1

    def record_stacking(self, dt):
        self.stacking_s += dt
        self.n_stacking += 1

    def record_moe(self, dt):
        self.moe_s += dt
        self.n_moe += 1

    def token_done(self, dt):
        self.full_token_s += dt
        self.n_tokens += 1

    def report(self):
        nt = max(self.n_tokens, 1)
        total_ms = self.full_token_s / nt * 1000
        moe_ms = (self.routing_s + self.stacking_s + self.moe_s) / nt * 1000

        lines = [
            f"\n{'=' * 65}",
            f"Per-Token Decode Breakdown  ({int(self.n_tokens)} tokens, {1000 / total_ms:.1f} t/s)",
            f"{'=' * 65}",
            f"  {'Component':<30s} {'ms/tok':>9s} {'% of tok':>9s} {'% of MoE':>9s}",
            f"  {'-' * 57}",
        ]

        def add(label, val_ms, of_moe=False):
            pct_tok = val_ms / total_ms * 100 if total_ms > 0 else 0
            pct_moe = (
                f"{val_ms / moe_ms * 100:>7.1f}%" if of_moe and moe_ms > 0 else " " * 9
            )
            lines.append(f"  {label:<30s} {val_ms:>8.2f}ms {pct_tok:>7.1f}%{pct_moe}")

        attn_overhead_ms = total_ms - moe_ms
        add("MoE total (40 layers)", moe_ms)
        add("  Routing", self.routing_s / nt * 1000, of_moe=True)
        add("  Expert stacking", self.stacking_s / nt * 1000, of_moe=True)
        add("  MoE matmul (compiled)", self.moe_s / nt * 1000, of_moe=True)
        add("Attention + norms + etc", attn_overhead_ms)

        n_per_layer = self.n_moe / max(nt, 1)
        if n_per_layer > 0:
            lines.append(
                f"\n  Per-call averages: "
                f"routing={self.routing_s / self.n_routing * 1000:.2f}ms, "
                f"stack={self.stacking_s / self.n_stacking * 1000:.2f}ms, "
                f"matmul={self.moe_s / self.n_moe * 1000:.2f}ms"
            )

        lines.append(f"\n  Raw ({int(nt)} tokens):")
        for k, label in [
            ("routing_s", "Routing"),
            ("stacking_s", "Stacking"),
            ("moe_s", "MoE matmul"),
            ("full_token_s", "Full token"),
        ]:
            v = getattr(self, k)
            dem = k.replace("_s", "").replace("full_token", "n_token") + "s"
            denom = getattr(self, dem, nt)
            lines.append(f"    {label:<15s} {v:.4f}s  ({v / denom * 1000:.2f}ms avg)")

        print("\n".join(lines) + "\n")


COUNTERS = PerfCounters()
_PROFILING = False


def _patch_streaming_moe():
    original_call = StreamingMoE.__call__

    def profiled_call(self, x):
        if not _PROFILING:
            return original_call(self, x)
        shp = x.shape
        xf = x.reshape(-1, shp[-1])
        N = xf.shape[0]
        if N != 1:
            return original_call(self, x)

        t0 = time.perf_counter()
        logits = self.gate(xf)
        if self.activation == "sigmoid":
            probs = mx.sigmoid(logits)
        else:
            probs = mx.softmax(logits, axis=-1)
        sel_probs = (
            probs + self.correction_bias if self.correction_bias is not None else probs
        )
        idx = mx.argpartition(-sel_probs, self.top_k - 1, axis=-1)[:, : self.top_k]
        w = mx.take_along_axis(probs, idx, axis=-1)
        if self.norm:
            w = w / mx.sum(w, axis=-1, keepdims=True)
        if self.routing_scale != 1.0:
            w = w * self.routing_scale
        idx_l = idx.tolist()
        COUNTERS.record_routing(time.perf_counter() - t0)

        t0 = time.perf_counter()
        experts = [
            self._cache.get(
                (self.layer_idx, int(e)),
                lambda e=e: self._store.load(self.layer_idx, int(e)),
            )
            for e in idx_l[0]
        ]
        k = len(experts)
        w_gu = mx.stack([e["gate.wq"] for e in experts] + [e["up.wq"] for e in experts])
        s_gu = mx.stack([e["gate.s"] for e in experts] + [e["up.s"] for e in experts])
        b_gu = mx.stack([e["gate.b"] for e in experts] + [e["up.b"] for e in experts])
        w_dw = mx.stack([e["down.wq"] for e in experts])
        s_dw = mx.stack([e["down.s"] for e in experts])
        b_dw = mx.stack([e["down.b"] for e in experts])
        weights = w[0].astype(mx.float16)
        mx.eval(w_gu, s_gu, b_gu, w_dw, s_dw, b_dw, weights)
        COUNTERS.record_stacking(time.perf_counter() - t0)

        t0 = time.perf_counter()
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
        )
        mx.eval(result)
        COUNTERS.record_moe(time.perf_counter() - t0)

        return result.reshape(shp).astype(x.dtype)

    StreamingMoE.__call__ = profiled_call


def run_profile(model, tokenizer, n_profile=30):
    global _PROFILING
    _patch_streaming_moe()

    prompt = "Write a Swift function gcd. Code only."
    prompt_tokens = tokenizer.encode(prompt)
    print(f"Prompt: {len(prompt_tokens)} tokens", flush=True)

    # Warmup via mlx_lm.generate (compiles _decode_moe, warms cache)
    print("Warmup...", end=" ", flush=True)
    from mlx_lm import generate as mlx_generate

    _ = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=20,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )
    print("done.", flush=True)

    # Profiled loop: use generate_step for correct cache handling
    _PROFILING = True
    COUNTERS.reset()
    mx.clear_cache()

    gen = generate_step(
        mx.array(prompt_tokens),
        model,
        max_tokens=n_profile + 2,
        sampler=make_sampler(temp=0.0),
    )

    # Skip first 2 tokens (compilation residue + cache settling)
    for _ in range(2):
        token_id, logprobs = next(gen)
        mx.eval(token_id)

    # Time the rest
    for _ in range(n_profile):
        t0 = time.perf_counter()
        token_id, logprobs = next(gen)
        mx.eval(token_id, logprobs)
        COUNTERS.token_done(time.perf_counter() - t0)

    _PROFILING = False
    COUNTERS.report()


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 6144
    perf = "--perf" in sys.argv
    model_path, store_dir = resolve_model()

    print(f"Model: {model_path}")
    print(f"Mode: {'perf' if perf else 'eco'} capacity={cap}")

    model, tokenizer = load(model_path, lazy=True)
    print(f"Loaded  mem={mx.get_active_memory() / 1e9:.2f}GB")
    wire_streaming(model, cap, perf=perf, store_dir=store_dir, model_path=model_path)
    print(f"Wired   mem={mx.get_active_memory() / 1e9:.2f}GB")

    run_profile(model, tokenizer)


if __name__ == "__main__":
    main()
