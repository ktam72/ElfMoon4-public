"""Routing pattern analysis: inter-layer expert similarity.

Measures how often adjacent layers select the same expert indices.
If correlation is high, we can skip tolist() in some layers and
reuse the previous layer's indices.
"""

import math
import sys
from collections import Counter

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from stream_model import StreamingMoE, _decode_moe, resolve_model, wire_streaming


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 6144
    n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    model_path, store_dir = resolve_model()

    print(f"Model: {model_path}", flush=True)
    model, tokenizer = load(model_path, lazy=True)
    wire_streaming(model, cap, store_dir=store_dir, model_path=model_path)
    print(f"Wired  mem={mx.get_active_memory() / 1e9:.2f}GB", flush=True)

    prompt = "Write Swift code for FizzBuzz, merge sort, and JSON parsing."
    prompt_tokens = tokenizer.encode(prompt)
    print(
        f"Prompt: {len(prompt_tokens)} tokens, profile: {n_tokens} tokens", flush=True
    )

    original_call = StreamingMoE.__call__

    routing_data = {}
    cur_token = [-1]

    def collect_routing(self, x):
        shp = x.shape
        xf = x.reshape(-1, shp[-1])
        N = xf.shape[0]
        if N != 1:
            return original_call(self, x)

        logits = self.gate(xf)
        probs = mx.softmax(logits, axis=-1)
        sel_probs = (
            probs + self.correction_bias if self.correction_bias is not None else probs
        )
        idx = mx.argpartition(-sel_probs, self.top_k - 1, axis=-1)[:, : self.top_k]
        w = mx.take_along_axis(probs, idx, axis=-1)
        if self.norm:
            w = w / mx.sum(w, axis=-1, keepdims=True)
        mx.eval(idx, w)
        idx_l = idx.tolist()

        if cur_token[0] >= 0:
            routing_data[(cur_token[0], self.layer_idx)] = set(idx_l[0])

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

    StreamingMoE.__call__ = collect_routing

    # Warmup
    from mlx_lm import generate as mlx_generate

    _ = mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=10,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )

    # Collect routing data
    mx.clear_cache()
    gen = generate_step(
        mx.array(prompt_tokens),
        model,
        max_tokens=n_tokens + 2,
        sampler=make_sampler(temp=0.0),
    )
    for _ in range(2):
        tok, lp = next(gen)
        mx.eval(tok)

    cur_token[0] = 0
    for step in range(n_tokens):
        cur_token[0] = step
        tok, lp = next(gen)
        mx.eval(tok, lp)

    cur_token[0] = -1
    StreamingMoE.__call__ = original_call

    # Analyze
    n_layers = max(k[1] for k in routing_data) + 1 if routing_data else 40
    n_tok_actual = max(k[0] for k in routing_data) + 1 if routing_data else 0
    print(
        f"\nCollected {len(routing_data)} routing decisions ({n_tok_actual} tokens × {n_layers} layers)",
        flush=True,
    )

    # 1. Adjacent layer overlap (same token, L vs L+1)
    total_adj = 0
    overlap_adj = 0
    jaccard_adj = []
    for tok in range(n_tok_actual):
        for l in range(n_layers - 1):
            s1 = routing_data.get((tok, l))
            s2 = routing_data.get((tok, l + 1))
            if s1 is not None and s2 is not None:
                total_adj += 1
                overlap_adj += len(s1 & s2)
                union = s1 | s2
                if union:
                    jaccard_adj.append(len(s1 & s2) / len(union))

    # 2. Same layer, adjacent tokens
    total_same = 0
    overlap_same = 0
    jaccard_same = []
    for l in range(n_layers):
        for t in range(n_tok_actual - 1):
            s1 = routing_data.get((t, l))
            s2 = routing_data.get((t + 1, l))
            if s1 is not None and s2 is not None:
                total_same += 1
                overlap_same += len(s1 & s2)
                union = s1 | s2
                if union:
                    jaccard_same.append(len(s1 & s2) / len(union))

    # 3. Per-layer entropy
    layer_choices = {l: [] for l in range(n_layers)}
    for (tok, l), experts in routing_data.items():
        layer_choices[l].append(experts)

    layer_stats = {}
    for l in range(n_layers):
        choices = layer_choices.get(l, [])
        if choices:
            flat = [e for s in choices for e in s]
            counter = Counter(flat)
            total_counts = sum(counter.values())
            entropy = -sum(
                (c / total_counts) * math.log2(c / total_counts)
                for c in counter.values()
            )
            max_entropy = math.log2(len(counter)) if counter else 1
            layer_stats[l] = {
                "unique": len(counter),
                "entropy": entropy,
                "norm": entropy / max_entropy if max_entropy > 0 else 0,
            }

    # 4. Window-based adjacency (L vs L+2, L vs L+3)
    def window_overlap(layers_apart, data):
        total = 0
        overlap = 0
        for tok in range(n_tok_actual):
            for l in range(n_layers - layers_apart):
                s1 = data.get((tok, l))
                s2 = data.get((tok, l + layers_apart))
                if s1 is not None and s2 is not None:
                    total += 1
                    overlap += len(s1 & s2)
        return overlap, total, overlap / max(total, 1) / 8 * 100

    # 5. Consecutive layer overlap frequency distribution
    adj_overlap_dist = Counter()
    for tok in range(n_tok_actual):
        for l in range(n_layers - 1):
            s1 = routing_data.get((tok, l))
            s2 = routing_data.get((tok, l + 1))
            if s1 is not None and s2 is not None:
                adj_overlap_dist[len(s1 & s2)] += 1

    # Print report
    print(f"\n{'=' * 65}")
    print(f"Routing Pattern Analysis  ({n_tok_actual} tokens × {n_layers} layers)")
    print(f"{'=' * 65}")

    print("\n── Adjacent Layer, Same Token (L vs L+1) ──")
    if total_adj > 0:
        pct = overlap_adj / (total_adj * 8) * 100
        print(f"  Comparisons:     {total_adj}")
        print(
            f"  Avg overlap:     {overlap_adj / total_adj:.2f}/8 experts ({pct:.1f}%)"
        )
        avg_j = sum(jaccard_adj) / len(jaccard_adj)
        print(f"  Avg Jaccard:      {avg_j:.3f}")
        print(f"\n  Overlap distribution ({n_tok_actual} tokens):")
        print(f"    {'Overlap':>8s}  {'Count':>6s}  {'%':>6s}")
        for o in range(9):
            c = adj_overlap_dist.get(o, 0)
            if c:
                print(f"    {o:>6d}/8  {c:>6d}  {c / total_adj * 100:>5.1f}%")
    else:
        print("  No data")

    print("\n── Same Layer, Adjacent Tokens (T vs T+1) ──")
    if total_same > 0:
        pct_s = overlap_same / (total_same * 8) * 100
        print(f"  Comparisons:     {total_same}")
        print(
            f"  Avg overlap:     {overlap_same / total_same:.2f}/8 experts ({pct_s:.1f}%)"
        )
        if jaccard_same:
            avg_js = sum(jaccard_same) / len(jaccard_same)
            print(f"  Avg Jaccard:      {avg_js:.3f}")
    else:
        print("  No data")

    print("\n── Windowed Adjacent Layer Overlap (same token) ──")
    for gap in [2, 4, 8]:
        ov, tc, p = window_overlap(gap, routing_data)
        if tc:
            print(f"  Layer L vs L+{gap:<2d}:  {ov / tc:.2f}/8 ({p:.1f}%)  ({tc} comp)")

    print(f"\n── Per-Layer Expert Diversity ({n_tok_actual} tokens) ──")
    print(f"  {'Layer':>6s} {'Unique':>7s} {'Entropy':>8s} {'Norm':>6s}")
    for l in range(min(10, n_layers)):
        s = layer_stats.get(l)
        if s:
            print(
                f"  {l:>6d}  {s['unique']:>5d}  {s['entropy']:>8.2f}  {s['norm']:>5.1%}"
            )
    if n_layers > 10:
        avgu = sum(s["unique"] for s in layer_stats.values()) / len(layer_stats)
        avge = sum(s["entropy"] for s in layer_stats.values()) / len(layer_stats)
        print(f"  {'average':>6s}  {avgu:>5.0f}  {avge:>8.2f}")

    print("\n── Top-1 Most Frequent Expert Per Layer ──")
    for l in range(min(10, n_layers)):
        choices = layer_choices.get(l, [])
        if choices:
            flat = [e for s in choices for e in s]
            top_expert, top_cnt = Counter(flat).most_common(1)[0]
            pct_top = top_cnt / len(flat) * 100
            print(
                f"  Layer {l:>2d}: expert {top_expert:>3d} ({top_cnt}/{len(flat)} = {pct_top:.0f}%)"
            )


if __name__ == "__main__":
    main()
