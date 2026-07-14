"""Phase 0: 切り分け実験 — idx.tolist() 同期コスト / ミスロードI/Oコスト の真値計測。

C案 HANDOFF_DECODE_SPEEDUP_C.md §2 に基づく。
- Exp-A: idx.tolist() 同期コスト（固定expert + tolist破棄 の2変種）
- Exp-B: ミスロード I/O コスト（SSD読み捨てバイパス）

プロトコル（§0 絶対ルール）:
  - python3 stream_model.py 6144 long を基準コマンドとする
  - 交互に2回ずつ実行（1回目は捨てる・ページキャッシュ温め）
  - 報告: gen tokens-per-sec（デコード単体）、命中率、ピークメモリ、品質先頭200字
"""

import time

import mlx.core as mx
from expert_store import BITS, GROUP, ExpertStore
from mlx_lm import generate, load
from resident_cache import ResidentCache
from stream_model import MODEL_PATH, STORE_DIR, StreamingMoE, _decode_moe

LONG_PROMPT = (
    "\n".join(
        f"func f{i}(_ x: Int) -> Int {{ return x * {i} + {i * i} }}" for i in range(40)
    )
    + "\n// Swiftで最大公約数gcd(_:_:)を書いて。コードのみ。"
)
FIXED_EIDS = tuple(range(8))
TOP_K = len(FIXED_EIDS)

# Save originals for restore
_ORIGINAL_MOE_CALL = StreamingMoE.__call__
_ORIGINAL_STORE_LOAD = ExpertStore.load


def _wire(model, store, cache):
    """モデルの全層を StreamingMoE に差し替え。"""
    layers = getattr(model, "layers", None) or model.model.layers
    for l, layer in enumerate(layers):
        mlp = layer.mlp
        n_exp = mlp.switch_mlp.gate_proj.weight.shape[0]
        gate = mlp.gate
        shared_exp = getattr(mlp, "shared_expert", None)
        shared_gate = getattr(mlp, "shared_expert_gate", None)
        layer.mlp = StreamingMoE(
            l,
            gate,
            n_exp,
            8,
            store,
            cache,
            shared_exp=shared_exp,
            shared_gate=shared_gate,
        )
    mx.clear_cache()


def _run_once(store, cache, patch_fn=None):
    """warmup(1回目) → cache stats リセット → measured(2回目) を同一モデルロードで実行。

    Metal のメモリ解放問題を回避するため、モデルは1回だけロードする。
    warmup 後に cache.hits/misses をゼロリセットし、measured の命中率は
    コールドスタート相当として記録する。
    """
    model, tok = load(MODEL_PATH, lazy=True)
    _wire(model, store, cache)

    if patch_fn:
        patch_fn()

    # --- warmup ---
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    out = generate(model, tok, prompt=LONG_PROMPT, max_tokens=80, verbose=False)
    dt = time.perf_counter() - t0
    s = cache.stats()
    peak = mx.get_peak_memory() / 1e9
    gen_tokens = len(tok.encode(out)) if out.strip() else 0
    gen_speed = gen_tokens / dt if dt > 0 and gen_tokens > 0 else 0.0
    print(
        f"    warmup: gen={gen_speed:.1f} t/s ({gen_tokens}t)  hit={s['hit_rate'] * 100:.1f}%  peak={peak:.1f}GB"
    )

    # --- reset cache stats so measured reflects cold-start miss rate ---
    cache.hits = 0
    cache.misses = 0
    mx.reset_peak_memory()

    # --- measured ---
    t0 = time.perf_counter()
    out = generate(model, tok, prompt=LONG_PROMPT, max_tokens=80, verbose=False)
    dt = time.perf_counter() - t0
    s = cache.stats()
    peak = mx.get_peak_memory() / 1e9
    gen_tokens = len(tok.encode(out)) if out.strip() else 0
    gen_speed = gen_tokens / dt if dt > 0 and gen_tokens > 0 else 0.0
    print(
        f"    measured: gen={gen_speed:.1f} t/s ({gen_tokens}t)  hit={s['hit_rate'] * 100:.1f}%  peak={peak:.1f}GB"
    )

    result = {
        "gen_speed": gen_speed,
        "gen_tokens": gen_tokens,
        "total_time": dt,
        "hit_rate": s["hit_rate"],
        "hits": s["hits"],
        "misses": s["misses"],
        "peak_gb": peak,
        "output_preview": out[:200] if out else "",
    }

    if patch_fn:
        StreamingMoE.__call__ = _ORIGINAL_MOE_CALL
        ExpertStore.load = _ORIGINAL_STORE_LOAD

    del model, tok
    mx.clear_cache()

    return result


# ---- Patch functions (save + replace at class level) ----


def _patch_fixed_experts():
    """ルーター不使用・固定expert・tolist同期なし。"""
    original = StreamingMoE.__call__

    def fixed_call(self, x):
        shp = x.shape
        xf = x.reshape(-1, shp[-1])
        N = xf.shape[0]
        if N == 1:
            bufs = []
            for e in FIXED_EIDS:
                w = self._cache.get(
                    (self.layer_idx, e), lambda e=e: self._store.load(self.layer_idx, e)
                )
                bufs.append(w)
            w_gu = mx.stack([b["gate.wq"] for b in bufs] + [b["up.wq"] for b in bufs])
            s_gu = mx.stack([b["gate.s"] for b in bufs] + [b["up.s"] for b in bufs])
            b_gu = mx.stack([b["gate.b"] for b in bufs] + [b["up.b"] for b in bufs])
            w_dw = mx.stack([b["down.wq"] for b in bufs])
            s_dw = mx.stack([b["down.s"] for b in bufs])
            b_dw = mx.stack([b["down.b"] for b in bufs])
            weights = mx.array([1.0 / TOP_K] * TOP_K, dtype=mx.float16)
            result = _decode_moe(
                xf[0:1],
                w_gu,
                s_gu,
                b_gu,
                w_dw,
                s_dw,
                b_dw,
                weights,
                TOP_K,
                shared=self._shared,
            )
            return result.reshape(shp).astype(x.dtype)
        return original(self, x)

    StreamingMoE.__call__ = fixed_call


def _patch_tolist_discard():
    """tolist() は呼ぶが結果破棄、固定expertを使用。"""
    original = StreamingMoE.__call__

    def tolist_discard_call(self, x):
        shp = x.shape
        xf = x.reshape(-1, shp[-1])
        N = xf.shape[0]
        if N == 1:
            logits = self.gate(xf)
            probs = mx.softmax(logits, axis=-1)
            idx = mx.argpartition(-probs, self.top_k - 1, axis=-1)[:, : self.top_k]
            _ = idx.tolist()
            bufs = []
            for e in FIXED_EIDS:
                w = self._cache.get(
                    (self.layer_idx, e), lambda e=e: self._store.load(self.layer_idx, e)
                )
                bufs.append(w)
            w_gu = mx.stack([b["gate.wq"] for b in bufs] + [b["up.wq"] for b in bufs])
            s_gu = mx.stack([b["gate.s"] for b in bufs] + [b["up.s"] for b in bufs])
            b_gu = mx.stack([b["gate.b"] for b in bufs] + [b["up.b"] for b in bufs])
            w_dw = mx.stack([b["down.wq"] for b in bufs])
            s_dw = mx.stack([b["down.s"] for b in bufs])
            b_dw = mx.stack([b["down.b"] for b in bufs])
            weights = mx.array([1.0 / TOP_K] * TOP_K, dtype=mx.float16)
            result = _decode_moe(
                xf[0:1],
                w_gu,
                s_gu,
                b_gu,
                w_dw,
                s_dw,
                b_dw,
                weights,
                TOP_K,
                shared=self._shared,
            )
            return result.reshape(shp).astype(x.dtype)
        return original(self, x)

    StreamingMoE.__call__ = tolist_discard_call


def _patch_noio_load():
    """ExpertStore.load をゼロ埋めダミー dict で置換（SSD I/O 完全ゼロ）。"""
    dim = 2048
    inter = 512  # Qwen3.6-35B 実測値 (DEFAULT_INTER=768 は本モデルと不一致)

    def _make_dummy(M, N):
        wq = mx.zeros((M, N // (32 // BITS)), dtype=mx.uint32)
        s = mx.zeros((M, N // GROUP), dtype=mx.bfloat16)
        b = mx.zeros((M, N // GROUP), dtype=mx.bfloat16)
        return wq, s, b

    gwq, gs, gb = _make_dummy(inter, dim)
    uwq, us, ub = _make_dummy(inter, dim)
    dwq, ds, db = _make_dummy(dim, inter)

    dummy = {
        "gate.wq": gwq,
        "gate.s": gs,
        "gate.b": gb,
        "up.wq": uwq,
        "up.s": us,
        "up.b": ub,
        "down.wq": dwq,
        "down.s": ds,
        "down.b": db,
    }

    def noio_load(self, layer, expert):
        return dummy

    ExpertStore.load = noio_load


# ---- Main ----


def main():
    print("=" * 60)
    print("Phase 0: 切り分け実験")
    print("=" * 60)
    print()

    results = {}

    # ---- Exp-A ----
    print("--- Exp-A: idx.tolist() 同期コスト ---")

    try:
        store = ExpertStore(STORE_DIR)
        cache = ResidentCache(6144)
        r = _run_once(store, cache)
        results["baseline"] = r
    except Exception as e:
        print(f"    baseline FAILED: {e}")

    try:
        store2 = ExpertStore(STORE_DIR)
        cache2 = ResidentCache(6144)
        r = _run_once(store2, cache2, patch_fn=_patch_fixed_experts)
        results["fixed_experts"] = r
    except Exception as e:
        print(f"    fixed_experts FAILED: {e}")

    try:
        store3 = ExpertStore(STORE_DIR)
        cache3 = ResidentCache(6144)
        r = _run_once(store3, cache3, patch_fn=_patch_tolist_discard)
        results["tolist_discard"] = r
    except Exception as e:
        print(f"    tolist_discard FAILED: {e}")

    # ---- Exp-B ----
    print("\n--- Exp-B: ミスロード I/O コスト ---")

    try:
        store4 = ExpertStore(STORE_DIR)
        cache4 = ResidentCache(6144)
        r = _run_once(store4, cache4)
        results["baseline_b"] = r
    except Exception as e:
        print(f"    baseline_b FAILED: {e}")

    try:
        store5 = ExpertStore(STORE_DIR)
        cache5 = ResidentCache(6144)
        r = _run_once(store5, cache5, patch_fn=_patch_noio_load)
        results["no_io"] = r
    except Exception as e:
        print(f"    no_io FAILED: {e}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("結果サマリ")
    print("=" * 60)
    print()
    print(
        f"{'Variant':<30s} {'gen t/s':>8s} {'gen tok':>7s} {'hit%':>6s} {'peak GB':>8s}"
    )
    print("-" * 61)
    labels = {
        "baseline": "Baseline (normal)",
        "fixed_experts": "Fixed experts (no sync, no miss)",
        "tolist_discard": "tolist() + discard (sync only)",
        "baseline_b": "Baseline B (normal load)",
        "no_io": "No-I/O load (SSD bypass)",
    }
    for k, v in results.items():
        print(
            f"  {labels.get(k, k):<28s} {v['gen_speed']:>8.1f} {v['gen_tokens']:>7d}"
            f" {v['hit_rate'] * 100:>5.1f}% {v['peak_gb']:>7.1f}GB"
        )

    print()
    b = results.get("baseline", {})
    f = results.get("fixed_experts", {})
    t = results.get("tolist_discard", {})
    n = results.get("no_io", {})

    if b and f and t:
        print("--- Exp-A 分析 ---")
        print(
            f"  同期＋ミス排除:          {f['gen_speed']:.1f} t/s (+{(f['gen_speed'] / b['gen_speed'] - 1) * 100:.0f}%)"
        )
        print(
            f"  同期のみ排除:            {t['gen_speed']:.1f} t/s (+{(t['gen_speed'] / b['gen_speed'] - 1) * 100:.0f}%)"
        )
        print(
            f"  ミス＋同期コスト合計:    {b['gen_speed'] - f['gen_speed']:.1f} t/s 差"
        )
        print(
            f"  同期コスト単体:          {b['gen_speed'] - t['gen_speed']:.1f} t/s 差"
        )
        print(
            f"  ミスコスト（同期以外）:  {t['gen_speed'] - f['gen_speed']:.1f} t/s 差"
        )
        print()
        print(
            f"  Baseline コールド:       {b['gen_speed']:.1f} t/s (hit {b['hit_rate'] * 100:.0f}%)"
        )
        print(
            f"  固定expert 上限:         {f['gen_speed']:.1f} t/s (hit {f['hit_rate'] * 100:.0f}%)"
        )
        print(
            f"  同期のみ:                {t['gen_speed']:.1f} t/s (hit {t['hit_rate'] * 100:.0f}%)"
        )

    if n and b:
        print()
        print("--- Exp-B 分析 ---")
        print(
            f"  Baseline:                {b['gen_speed']:.1f} t/s (hit {b['hit_rate'] * 100:.0f}%)"
        )
        print(
            f"  No-I/O load:             {n['gen_speed']:.1f} t/s (hit {n['hit_rate'] * 100:.0f}%)"
        )
        print(
            f"  ミスI/Oコスト:           {b['gen_speed'] - n['gen_speed']:.1f} t/s 差"
        )


if __name__ == "__main__":
    main()
