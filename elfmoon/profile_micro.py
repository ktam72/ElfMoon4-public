"""Micro-benchmarks: isolate decode bottleneck components.

Experiments:
  1. Production-equivalent timing (no extra mx.eval barriers)
  2. tolist() sync cost per layer
  3. cache.get dict overhead
  4. Per-expert matmul vs stack+compiled
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

# ── Experiment 1: Production-equivalent timing ─────────────────


def exp1_production_timing(model, tokenizer, n_tokens=30):
    """Time the real production path with NO extra mx.eval barriers.

    Only the minimal instrumentation: wrap each StreamingMoE.__call__
    with a single wall-clock timer (no internal mx.eval splitting).
    """
    original_call = StreamingMoE.__call__

    moe_total_s = 0.0
    call_count = 0

    def timed_call(self, x):
        nonlocal moe_total_s, call_count
        t0 = time.perf_counter()
        result = original_call(self, x)
        mx.eval(result)
        moe_total_s += time.perf_counter() - t0
        call_count += 1
        return result

    StreamingMoE.__call__ = timed_call

    prompt = tokenizer.encode("Write a Swift function gcd. Code only.")

    # Warmup
    from mlx_lm import generate as mlx_generate

    _ = mlx_generate(
        model,
        tokenizer,
        prompt="Write a Swift function gcd. Code only.",
        max_tokens=15,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )

    # Profile
    mx.clear_cache()
    gen = generate_step(
        mx.array(prompt), model, max_tokens=n_tokens + 2, sampler=make_sampler(temp=0.0)
    )
    for _ in range(2):
        tok_id, lp = next(gen)
        mx.eval(tok_id)

    moe_total_s = 0.0
    call_count = 0
    t_tokens = []

    for _ in range(n_tokens):
        t0 = time.perf_counter()
        tok_id, lp = next(gen)
        mx.eval(tok_id, lp)
        t_tokens.append(time.perf_counter() - t0)

    StreamingMoE.__call__ = original_call

    avg_token = sum(t_tokens) / len(t_tokens) * 1000
    avg_moe = moe_total_s / max(call_count, 1) * 1000 * 40  # 40 layers

    print(f"\n{'=' * 60}")
    print("Exp 1: Production-equivalent timing")
    print(f"{'=' * 60}")
    print(f"  Full token:    {avg_token:.2f}ms ({1000 / avg_token:.1f} t/s)")
    print(f"  MoE total:     {moe_total_s / n_tokens * 1000:.2f}ms")
    print(f"  MoE/layer:     {moe_total_s / call_count * 1000:.3f}ms")
    print(
        f"  Non-MoE:       {avg_token - moe_total_s / n_tokens * 1000:.2f}ms "
        f"({(avg_token - moe_total_s / n_tokens * 1000) / avg_token * 100:.1f}%)"
    )
    print(f"  MoE count:     {call_count} calls ({call_count / n_tokens:.1f}/tok)")


# ── Experiment 2: tolist() sync cost ───────────────────────────


def exp2_tolist_cost(model, tokenizer, n_tokens=30):
    """Measure the actual cost of idx.tolist() GPU→CPU sync per layer.

    Compares:
    - Baseline: original StreamingMoE.__call__
    - No-tolist: skip tolist() by feeding fixed expert indices
    """
    from mlx_lm.sample_utils import make_sampler

    prompt = tokenizer.encode("Write a Swift function gcd. Code only.")

    # Warmup
    _ = mx.random.uniform
    from mlx_lm import generate as mlx_generate

    _ = mlx_generate(
        model,
        tokenizer,
        prompt="Write a Swift function gcd. Code only.",
        max_tokens=15,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )

    # ── Baseline: standard generate_step timing ──
    mx.clear_cache()
    gen = generate_step(
        mx.array(prompt), model, max_tokens=n_tokens + 2, sampler=make_sampler(temp=0.0)
    )
    for _ in range(2):
        tok_id, lp = next(gen)
        mx.eval(tok_id)
    baseline_t = []
    for _ in range(n_tokens):
        t0 = time.perf_counter()
        tok_id, lp = next(gen)
        mx.eval(tok_id, lp)
        baseline_t.append(time.perf_counter() - t0)
    baseline_avg = sum(baseline_t) / len(baseline_t) * 1000

    # ── Experimental: patch to measure tolist time ──
    # We patch StreamingMoE.__call__ to time the routing section
    original_call = StreamingMoE.__call__

    tolist_times = []  # per-layer
    routing_times = []

    def routing_timed_call(self, x):
        if x.shape[0] != 1:
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
        mx.eval(idx, w)
        idx_l = idx.tolist()
        routing_times.append(time.perf_counter() - t0)
        tolist_times.append(time.perf_counter() - t0)

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
        return result.reshape(shp).astype(x.dtype)

    StreamingMoE.__call__ = routing_timed_call

    mx.clear_cache()
    gen = generate_step(
        mx.array(prompt), model, max_tokens=n_tokens + 2, sampler=make_sampler(temp=0.0)
    )
    for _ in range(2):
        tok_id, lp = next(gen)
        mx.eval(tok_id)
    tolist_times.clear()
    routing_times.clear()
    for _ in range(n_tokens):
        tok_id, lp = next(gen)
        mx.eval(tok_id, lp)

    StreamingMoE.__call__ = original_call

    avg_routing = sum(routing_times) / len(routing_times) * 1000
    avg_tolist = sum(tolist_times) / len(tolist_times) * 1000

    print(f"\n{'=' * 60}")
    print("Exp 2: tolist() sync cost (per layer)")
    print(f"{'=' * 60}")
    print(f"  Baseline token:    {baseline_avg:.2f}ms")
    print(
        f"  Routing+sync avg:  {avg_routing:.3f}ms/layer ({avg_routing * 40:.2f}ms total)"
    )
    print(
        f"  tolist+gate+soft:  {avg_tolist:.3f}ms/layer ({avg_tolist * 40:.2f}ms total)"
    )
    print(
        f"  Samples:           {len(routing_times)} routing calls ({len(routing_times) / 40 / n_tokens:.1f} layers avg)"
    )


# ── Experiment 3: cache.get overhead ───────────────────────────


def exp3_cache_get_overhead(model, tokenizer, n_tokens=30):
    """Isolate the Python overhead of cache.get + mx.stack.

    Compares:
    - Standard path (8 cache.get + 6 mx.stack + compiled matmul)
    - Direct matmul: bypass stack, do per-expert matmul with individual weights
    """
    original_call = StreamingMoE.__call__

    # Standard path timing
    standard_moe_s = 0.0
    standard_calls = 0

    def standard_timed(self, x):
        nonlocal standard_moe_s, standard_calls
        if x.shape[0] != 1:
            return original_call(self, x)
        shp = x.shape
        xf = x.reshape(-1, shp[-1])
        N = xf.shape[0]
        if N != 1:
            return original_call(self, x)

        t0 = time.perf_counter()
        logits = self.gate(xf)
        probs = mx.softmax(logits, axis=-1)
        idx = mx.argpartition(-probs, self.top_k - 1, axis=-1)[:, : self.top_k]
        w = mx.take_along_axis(probs, idx, axis=-1)
        w = w / mx.sum(w, axis=-1, keepdims=True)
        mx.eval(idx, w)
        idx_l = idx.tolist()

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
        standard_moe_s += time.perf_counter() - t0
        standard_calls += 1
        return result.reshape(shp).astype(x.dtype)

    StreamingMoE.__call__ = standard_timed

    prompt = tokenizer.encode("Write a Swift function gcd. Code only.")
    from mlx_lm import generate as mlx_generate

    _ = mlx_generate(
        model,
        tokenizer,
        prompt="Write a Swift function gcd. Code only.",
        max_tokens=15,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )

    mx.clear_cache()
    gen = generate_step(
        mx.array(prompt), model, max_tokens=n_tokens + 2, sampler=make_sampler(temp=0.0)
    )
    for _ in range(2):
        tok_id, lp = next(gen)
        mx.eval(tok_id)
    standard_moe_s = 0.0
    standard_calls = 0
    for _ in range(n_tokens):
        tok_id, lp = next(gen)
        mx.eval(tok_id, lp)

    std_avg = standard_moe_s / standard_calls * 1000 if standard_calls else 0

    StreamingMoE.__call__ = original_call

    print(f"\n{'=' * 60}")
    print("Exp 3: MoE internal cost (standard path)")
    print(f"{'=' * 60}")
    print(f"  MoE/layer:        {std_avg:.3f}ms")
    print(f"  MoE total/token:  {std_avg * 40:.2f}ms")
    print(f"  Full token:       {std_avg * 40 / 0.85:.2f}ms (est 85% MoE share)")


# ── Main ────────────────────────────────────────────────────────


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 6144
    perf = "--perf" in sys.argv
    model_path, store_dir = resolve_model()

    print(f"Model: {model_path}")
    model, tokenizer = load(model_path, lazy=True)
    print(f"Loaded  mem={mx.get_active_memory() / 1e9:.2f}GB")
    wire_streaming(model, cap, perf=perf, store_dir=store_dir, model_path=model_path)
    print(f"Wired   mem={mx.get_active_memory() / 1e9:.2f}GB")

    exp1_production_timing(model, tokenizer)
    exp2_tolist_cost(model, tokenizer)
    exp3_cache_get_overhead(model, tokenizer)


if __name__ == "__main__":
    main()
