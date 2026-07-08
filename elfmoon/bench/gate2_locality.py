"""§8.4 Gate-2: 時間的局所性の実測

デコード中の各層の top-8 expert id をログし分析:
1. 連続トークン間 per-layer 一致率
2. 直近 W=8 トークン窓の頻度上位で次トークンミス予測カバー率

注意: StreamingMoE.__call__ を丸ごと置換してログ挿入。二重計算なし。
教訓: タイマー不要（ログ収集のみ）だが実体化済み idx.tolist() を使う。
"""

import time
from collections import Counter

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from stream_model import StreamingMoE, MODEL_PATH, STORE_DIR, _decode_moe
from expert_store import ExpertStore, GROUP, BITS
from resident_cache import ResidentCache

LONG_PROMPT = (
    "\n".join(
        f"func f{i}(_ x: Int) -> Int {{ return x * {i} + {i * i} }}" for i in range(40)
    )
    + "\n// Swiftで最大公約数gcd(_:_:)を書いて。コードのみ。"
)

MAX_TOKENS = 80
W = 8  # window size

EXPERT_LOG = []  # [(token_idx, layer_idx, [8 eids])]

_ORIGINAL_CALL = StreamingMoE.__call__


def _make_logging_call():
    """オリジナルの __call__ にログ収集だけ挿入した版を返す。"""

    def logging_call(self, x):
        shp = x.shape
        xf = x.reshape(-1, shp[-1])
        logits = self.gate(xf).astype(mx.float32)
        probs = mx.softmax(logits, axis=-1)
        idx = mx.argpartition(-probs, self.top_k - 1, axis=-1)[:, : self.top_k]
        w = mx.take_along_axis(probs, idx, axis=-1)
        if self.norm:
            w = w / mx.sum(w, axis=-1, keepdims=True)
        idx_l = idx.tolist()
        N = xf.shape[0]

        def load(e):
            return self._cache.get(
                (self.layer_idx, e), lambda e=e: self._store.load(self.layer_idx, e)
            )

        if N == 1:
            # ★ ログ収集: idx_l は tolist() で実体化済み
            EXPERT_LOG.append((self.layer_idx, idx_l[0]))
            experts = [load(e) for e in idx_l[0]]

            w_gw = mx.stack([e["gate.wq"] for e in experts])
            s_gw = mx.stack([e["gate.s"] for e in experts])
            b_gw = mx.stack([e["gate.b"] for e in experts])
            w_up = mx.stack([e["up.wq"] for e in experts])
            s_up = mx.stack([e["up.s"] for e in experts])
            b_up = mx.stack([e["up.b"] for e in experts])
            w_dw = mx.stack([e["down.wq"] for e in experts])
            s_dw = mx.stack([e["down.s"] for e in experts])
            b_dw = mx.stack([e["down.b"] for e in experts])

            weights = w[0].astype(mx.float16)
            result = _decode_moe(
                xf[0:1],
                w_gw,
                s_gw,
                b_gw,
                w_up,
                s_up,
                b_up,
                w_dw,
                s_dw,
                b_dw,
                weights,
                self.top_k,
                shared=self._shared,
            ).reshape(shp)
            return result.astype(x.dtype)

        # プレフィル(N>1): 通常処理
        w_l = w.tolist()
        expert_groups = {}
        for t_idx in range(N):
            for j in range(self.top_k):
                e = int(idx_l[t_idx][j])
                if e not in expert_groups:
                    expert_groups[e] = []
                expert_groups[e].append((t_idx, w_l[t_idx][j]))

        token_buf = [None] * N
        for e, items in expert_groups.items():
            exp = load(e)
            indices = [it[0] for it in items]
            weights_list = [it[1] for it in items]
            xb = xf[mx.array(indices)]
            g = mx.quantized_matmul(
                xb,
                exp["gate.wq"],
                exp["gate.s"],
                exp["gate.b"],
                transpose=True,
                group_size=GROUP,
                bits=BITS,
            )
            u = mx.quantized_matmul(
                xb,
                exp["up.wq"],
                exp["up.s"],
                exp["up.b"],
                transpose=True,
                group_size=GROUP,
                bits=BITS,
            )
            h = (g * mx.sigmoid(g)) * u
            yo = mx.quantized_matmul(
                h,
                exp["down.wq"],
                exp["down.s"],
                exp["down.b"],
                transpose=True,
                group_size=GROUP,
                bits=BITS,
            )
            wv = mx.array(weights_list).astype(yo.dtype)
            contrib = yo * wv[:, None]
            for i, t_idx in enumerate(indices):
                if token_buf[t_idx] is None:
                    token_buf[t_idx] = contrib[i]
                else:
                    token_buf[t_idx] = token_buf[t_idx] + contrib[i]
        out = mx.stack(token_buf).reshape(shp)
        if self._shared is not None:
            (
                sg_w,
                sg_s,
                sg_b,
                sg_bits,
                sg_gs,
                se_gw,
                se_gs,
                se_gb,
                se_uw,
                se_us,
                se_ub,
                se_dw,
                se_ds,
                se_db,
            ) = self._shared
            sg = mx.quantized_matmul(
                xf, sg_w, sg_s, sg_b, transpose=True, group_size=sg_gs, bits=sg_bits
            )
            se_g = mx.quantized_matmul(
                xf, se_gw, se_gs, se_gb, transpose=True, group_size=GROUP, bits=BITS
            )
            se_u = mx.quantized_matmul(
                xf, se_uw, se_us, se_ub, transpose=True, group_size=GROUP, bits=BITS
            )
            se_h = (se_g * mx.sigmoid(se_g)) * se_u
            se_out = mx.quantized_matmul(
                se_h, se_dw, se_ds, se_db, transpose=True, group_size=GROUP, bits=BITS
            )
            out = out + mx.sigmoid(sg) * se_out
        return out.astype(x.dtype)

    return logging_call


def _wire(model, store, cache):
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


def main():
    global EXPERT_LOG
    EXPERT_LOG = []

    print("モデルロード中...")
    model, tok = load(MODEL_PATH, lazy=True)
    store = ExpertStore(STORE_DIR)
    cache = ResidentCache(6144)
    _wire(model, store, cache)

    # パッチ適用
    StreamingMoE.__call__ = _make_logging_call()

    print("生成中...")
    prompt_tokens = tok.encode(LONG_PROMPT)
    prompt_arr = mx.array(prompt_tokens)

    t0 = time.perf_counter()
    gen_tokens = []
    for i, (token, _) in enumerate(
        generate_step(prompt_arr, model, max_tokens=MAX_TOKENS)
    ):
        if i == 0:
            prompt_time = time.perf_counter() - t0
            t_decode = time.perf_counter()
        gen_tokens.append(int(token))
    decode_time = time.perf_counter() - t_decode
    gen_speed = len(gen_tokens) / decode_time if decode_time > 0 else 0.0

    print(f"生成: {len(gen_tokens)} tokens, {gen_speed:.3f} t/s")
    print(f"ヒット率: {cache.hit_rate * 100:.1f}%")
    print(f"ログエントリ数: {len(EXPERT_LOG)}")

    # ---- ログ構造化: layer==0 でトークン境界 ----
    token_log = []
    cur = []
    for entry in EXPERT_LOG:
        layer, experts = entry
        if layer == 0 and cur:
            token_log.append(cur)
            cur = []
        cur.append((layer, experts))
    if cur:
        token_log.append(cur)

    n_tok = len(token_log)
    n_layers = len(token_log[0]) if token_log else 0
    print(f"分析トークン数: {n_tok}, 層数: {n_layers}")

    # ---- 1. 連続トークン一致率 ----
    pair_agreements = []
    for t in range(1, n_tok):
        layer_scores = []
        for l_idx in range(n_layers):
            prev = set(token_log[t - 1][l_idx][1])
            curr = set(token_log[t][l_idx][1])
            layer_scores.append(len(prev & curr) / 8.0)
        pair_agreements.append(sum(layer_scores) / len(layer_scores))

    avg_agree = sum(pair_agreements) / len(pair_agreements) if pair_agreements else 0
    print(f"\n=== 連続トークン一致率 ===")
    print(f"  全体平均: {avg_agree * 100:.1f}%")

    per_layer_agree = [0.0] * n_layers
    per_layer_cnt = [0] * n_layers
    for t in range(1, n_tok):
        for l_idx in range(n_layers):
            prev = set(token_log[t - 1][l_idx][1])
            curr = set(token_log[t][l_idx][1])
            per_layer_agree[l_idx] += len(prev & curr)
            per_layer_cnt[l_idx] += 1
    per_layer_agree = [
        per_layer_agree[i] / (per_layer_cnt[i] * 8) * 100
        for i in range(n_layers)
        if per_layer_cnt[i] > 0
    ]
    print(f"  層ごと平均 (全層): {sum(per_layer_agree) / len(per_layer_agree):.1f}%")
    print(f"  最大層: {max(per_layer_agree):.1f}%")
    print(f"  最小層: {min(per_layer_agree):.1f}%")

    # ---- 2. 窓ベース予測カバー率 ----
    K_list = [8, 16, 24, 32]
    print(f"\n=== 窓ベース予測カバー率 (W={W}) ===")
    for K in K_list:
        coverages = []
        for t in range(W, n_tok):
            layer_covers = []
            for l_idx in range(n_layers):
                window = []
                for wt in range(t - W, t):
                    window.extend(token_log[wt][l_idx][1])
                freq = Counter(window)
                predicted = {e for e, _ in freq.most_common(K)}
                actual = set(token_log[t][l_idx][1])
                if actual:
                    layer_covers.append(len(actual & predicted) / len(actual))
            if layer_covers:
                coverages.append(sum(layer_covers) / len(layer_covers))
        avg_cov = sum(coverages) / len(coverages) if coverages else 0
        print(f"  K={K:2d}: {avg_cov * 100:.1f}%")

    # ---- 3. Per-layer K=8 カバー率 ----
    K = 8
    per_layer_cover = [0.0] * n_layers
    per_layer_covcnt = [0] * n_layers
    for t in range(W, n_tok):
        for l_idx in range(n_layers):
            window = []
            for wt in range(t - W, t):
                window.extend(token_log[wt][l_idx][1])
            freq = Counter(window)
            predicted = {e for e, _ in freq.most_common(K)}
            actual = set(token_log[t][l_idx][1])
            if actual:
                per_layer_cover[l_idx] += len(actual & predicted) / len(actual)
                per_layer_covcnt[l_idx] += 1
    per_layer_cover_pct = [
        per_layer_cover[i] / per_layer_covcnt[i] * 100 if per_layer_covcnt[i] > 0 else 0
        for i in range(n_layers)
    ]
    print(f"\n=== Per-layer カバー率 (K={K}, W={W}) ===")
    print(f"  全体平均: {sum(per_layer_cover_pct) / len(per_layer_cover_pct):.1f}%")
    best_idx = per_layer_cover_pct.index(max(per_layer_cover_pct))
    worst_idx = per_layer_cover_pct.index(min(per_layer_cover_pct))
    print(f"  最大: {max(per_layer_cover_pct):.1f}% (layer {best_idx})")
    print(f"  最小: {min(per_layer_cover_pct):.1f}% (layer {worst_idx})")

    # ---- 4. 入れ替わり数 (churn) ----
    total_churn = 0
    total_pairs = 0
    for t in range(1, n_tok):
        for l_idx in range(n_layers):
            prev_set = set(token_log[t - 1][l_idx][1])
            curr_set = set(token_log[t][l_idx][1])
            total_churn += 8 - len(prev_set & curr_set)
            total_pairs += 1
    avg_churn = total_churn / total_pairs if total_pairs else 0
    print(f"\n=== 入れ替わり ===")
    print(f"  平均 churn / 層: {avg_churn:.2f} / 8")

    # ---- 5. 出力確認 ----
    out = tok.decode(gen_tokens)
    print(f"\n--- 出力先頭200字 ---")
    print(out[:200] if out else "(empty)")

    # 復元
    StreamingMoE.__call__ = _ORIGINAL_CALL


if __name__ == "__main__":
    main()
