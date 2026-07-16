"""モジュール④(前半): ホットリスト・プロファイルと起動時プライム。

MoEのexpert使用は強く偏る。実走(キャリブレーション)で頻度を数え、
頻出expertを起動時にキャッシュへ常駐させて命中率の底上げをする。
DS4 の ds4_streaming_hotlist.inc（hits/weight順の静的リスト）に対応。

真の層先読みプリフェッチ（層Lの計算中に層L+1のexpertをSSDから先読み）は
毎層routerが前層出力に依存する逐次性があるため後段で扱う。まずは最も効く静的プライム。
"""

from collections import Counter

import mlx.core as mx
from moe_block import route


def profile(gates, top_k, token_stream):
    """キャリブレーション列を流し、(layer,expert)の使用頻度を数える。
    gates: [layer]->[n_experts,dim], token_stream: iterable of x:[dim]。"""
    counter = Counter()
    for x in token_stream:
        h = x
        for l, gw in enumerate(gates):
            idx, _ = route(h, gw, top_k)
            for e in idx.tolist():
                counter[(l, int(e))] += 1
            h = h + mx.random.normal(h.shape) * 0.01  # 層で少し変化
    return counter


def build_hotlist(counter):
    """頻度降順の [(layer,expert), ...]。"""
    return [k for k, _ in counter.most_common()]


def prime_cache(cache, store, hotlist, max_experts):
    """ホットリスト上位を起動時にキャッシュへ常駐（命中/ミス統計を汚さない）。"""
    n = 0
    for l, e in hotlist[:max_experts]:
        cache.prime((l, e), store.load(l, e))
        n += 1
    mx.eval()
    return n
