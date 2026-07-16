# V4-Flash Feasibility Gate（指示 #12b §3）

- モデル: DeepSeek-V4-Flash (284B total / 13B active)
- ターゲット: Mac (Apple Silicon), eco 24GB unified memory
- 一次情報: config.json（HuggingFace）、arXiv 2606.19348、directive #12b

## 1. モデル諸元（config.json 実値）

| 項目 | 値 |
|------|-----|
| hidden_size | 4096 |
| moe_intermediate_size | 2048 |
| n_routed_experts | 256 |
| n_shared_experts | 1 |
| num_experts_per_tok | 6 |
| num_hidden_layers | 43（先頭2層dense + 残41sparse） |
| num_attention_heads | 64 |
| num_key_value_heads | 1（MLA） |
| head_dim | 512 |
| q_lora_rank / o_lora_rank | 1024 / 1024 |
| qk_rope_head_dim | 64 |
| index_n_heads / index_head_dim / index_topk | 64 / 128 / 512 |
| vocab_size | 129280 |
| disk (FP8) | 282GB |
| expert 量子化（MLX） | mxfp4(group32, bits4) |
| 非expert 量子化（MLX） | affine int4(group64, bits4) |

## 2. メモリ見積り（MLX 4-bit quantized）

### Expert サイズ
- 1 expert ≈ 12.6MB（gate/up/down × hidden4096 × inter2048 × mxfp4 + scales）
- sparse 層: 41 × 256 = 10,496 experts
- 全 expert 保存: 10,496 × 12.6MB = **132GB**

### 非expert 常駐（全層必須）
- Attention(MLA)・norm・mHC・gate・shared expert・embed・lm_head
- 43層 × hidden=4096 の非expert params ≈ 13B active params の非expert分
- 非expert int4 換算: 約 **6-8GB**（13B - ~7B expert active = ~6B non-expert）

### ResidentCache 可能容量
| 項目 | 見積り |
|------|--------|
| 非expert 常駐 | ~7GB |
| KV cache（MLA, 1M context） | ~2GB（kv_heads=1 のため小）|
| 活性化メモリ | ~1GB |
| OS/他 | ~2GB |
| **expert cache 利用可能** | **~12GB** |

12GB ÷ 12.6MB/expert ≈ **950 experts**

### 被覆率
- 全 expert: 10,496
- 常駐可能: ~950
- **被覆率: ~9%**（80B の 25% より大幅低）
- トークン当たりの読み込み: top6 × 41層 = 246 experts × 12.6MB = **3.1GB/token**

## 3. Decode 速度見積り

### 前提
- SSD 帯域: ~4.4GB/s（実測, 990Pro）
- mxfp4 compute: 80B の ~13B/3.5B ≈ **3.7× 多い active params**

### シナリオ別

| シナリオ | 1層のmiss | miss load | compute | 合計/token | t/s |
|----------|-----------|-----------|---------|-----------|-----|
| **全 miss（cold start）** | 6 | 6×2.9ms=17.4ms | ~3ms | 41×20.4ms=837ms | **~1.2** |
| **定常状態（~9% hit）** | 5.5 | 5.5×2.9ms=16ms | ~3ms | 41×19ms=779ms | **~1.3** |
| **楽観（30% hit）** | 4.2 | 4.2×2.9ms=12.2ms | ~3ms | 41×15.2ms=623ms | **~1.6** |
| **理想（全 resident）** | 0 | 0 | ~3ms | 41×3ms=123ms | **~8.1** |

**実用見込み: 1-2 t/s**。SSD 帯域が律速（3.1GB/token ÷ 4.4GB/s = 704ms/token の純読み込み時間）。被覆率 9% では命中率が上がらず SSD I/O が支配的。

### 80B との比較

| | 80B A3B | V4-Flash |
|--|---------|----------|
| active params | 3.5B | **13B (3.7×)** |
| expert サイズ | 1.77MB | **12.6MB (7.1×)** |
| total experts | 24,576 | 10,496 |
| 被覆率 (eco) | 25% | **~9%** |
| /token read | 0.27GB | **3.1GB (11.5×)** |
| 実測 t/s | ~10 | **推定 1-2** |

## 4. Prefill 速度見積り

gather_qmm 経路でも expert 7倍・non-expert 3.7倍のため:
- 80B prefill: ~109 tok/s（1193 token, fused path）
- V4-Flash 推定: 109 ÷ 3.7 ≈ **~29 tok/s**（compute scaling、bandwidth similar）
- 1M context prefill: 1M ÷ 29 ≈ **~9.6時間**

## 5. 判定

**V4-Flash は 24GB eco の実用圏外。**

| 基準 | 結果 | 判定 |
|------|------|------|
| 常駐メモリ < 24GB | 非expert ~7GB + cache ~12GB = 19GB。収まるが expert cache が極小 | ⚠️ 境界 |
| decode ≥ 5 t/s | 推定 1-2 t/s。SSD 帯域律速で目標の 1/3〜1/5 | ❌ |
| prefill 実用的 | 1M context で ~10時間 | ❌ |
| 80B との改善 | 7倍大きい expert + 3.7倍 active = 実効性能 1/10〜1/20 | ❌ |

### 選択肢（PM 判断）

1. **中止**: V4-Flash は 24GB Apple Silicon の実用圏外。リソースを他に振る。
2. **perf モード前提**: 64GB/128GB 以上のメモリを前提に再設計。expert cache 容量を増やせるが、ExpertStore の 132GB 全展開は不可のまま。
3. **V4-Pro は論外**: 1.6T total / 49B active。eco 24GB では全く収まらない。
4. **80B の prefill/pipeline 最適化に集中**: 既に効くと分かっている方向（postmortem の方針）。
