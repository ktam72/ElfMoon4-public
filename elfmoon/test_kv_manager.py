"""kv_manager の整合スナップショット方式の検証テスト。

実行: cd elfmoon && python3 test_kv_manager.py
（pytest 不要。実際の mlx_lm KVCache / ArraysCache を使用）
"""

import json
import os
import shutil
import tempfile
import time

import mlx.core as mx
from kv_manager import MIN_SAVE_TOKENS, KVCacheManager
from mlx_lm.models.cache import ArraysCache, KVCache


class FakeModel:
    """lookup が層数を数えるためだけの最小モデル。"""

    def __init__(self, n_layers):
        self.layers = list(range(n_layers))


def make_cache(offset, seed=0):
    """KVCache(履歴 offset トークン) + ArraysCache(再帰状態) のペアを作る。"""
    mx.random.seed(seed)
    kv = KVCache()
    keys = mx.random.normal((1, 2, offset, 4)).astype(mx.float16)
    vals = mx.random.normal((1, 2, offset, 4)).astype(mx.float16)
    kv.update_and_fetch(keys, vals)
    arr = ArraysCache(size=2)
    arr.state = [
        mx.random.normal((1, 3)).astype(mx.float16),
        mx.random.normal((1, 2)).astype(mx.float16),
    ]
    mx.eval([kv.keys, kv.values] + arr.state)
    return [kv, arr]


def wait_disk(mgr, key, timeout=5.0):
    """バックグラウンドのディスク書込み完了を待つ。"""
    deadline = time.time() + timeout
    path = mgr._meta_path(key)
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.05)
    return False


def test_roundtrip_and_immutability(tmp):
    """スナップショット→（生成による変異）→保存→復元が捕捉時点と一致すること。"""
    mgr = KVCacheManager(cache_dir=tmp)
    model = FakeModel(2)
    offset = 30
    tokens = list(range(offset))
    cache = make_cache(offset, seed=1)

    snap = mgr.snapshot(cache)
    assert snap is not None, "snapshot が None"

    # 捕捉時点の期待値を控える
    exp_k = mx.array(cache[0].keys[..., :offset, :])
    exp_arr0 = mx.array(cache[1].state[0])
    mx.eval(exp_k, exp_arr0)

    # --- 生成をシミュレート: KV追記 + 再帰状態の上書き ---
    cache[0].update_and_fetch(mx.ones((1, 2, 5, 4), dtype=mx.float16), mx.ones((1, 2, 5, 4), dtype=mx.float16))
    cache[1].state = [mx.zeros((1, 3)), mx.zeros((1, 2))]

    mgr.save(tokens, snap)

    restored, cached_len = mgr.lookup(tokens + [999], model)
    assert cached_len == offset, f"offset 不一致: {cached_len}"
    assert restored is not None
    assert isinstance(restored[0], KVCache) and restored[0].offset == offset
    assert isinstance(restored[1], ArraysCache)
    # 生成による変異が混入していないこと（＝捕捉時点と一致）
    assert bool(mx.all(restored[0].keys[..., :offset, :] == exp_k)), "KV が変異している"
    assert bool(mx.all(restored[1].state[0] == exp_arr0)), "再帰状態が変異している"
    print("OK: roundtrip + snapshot 不変性")
    return mgr, tokens


def test_disk_restore(tmp, tokens):
    """新インスタンス（メモリ空）でディスクから復元でき、再帰状態も戻ること。"""
    mgr = KVCacheManager(cache_dir=tmp)  # メモリは空
    model = FakeModel(2)
    restored, cached_len = mgr.lookup(list(tokens) + [999], model)
    assert cached_len == len(tokens), f"disk hit 失敗: {cached_len}"
    assert isinstance(restored[1], ArraysCache) and restored[1].state[0] is not None, (
        "再帰状態がディスクから復元されていない"
    )
    assert restored[1].state[0].shape == (1, 3)
    print("OK: ディスク復元（再帰状態含む）")


def test_longest_prefix(tmp):
    """複数エントリ一致時に最長 offset を選ぶこと。"""
    mgr = KVCacheManager(cache_dir=tmp)
    model = FakeModel(2)
    base = list(range(100, 160))  # 60 tokens
    for off in (25, 40):
        cache = make_cache(off, seed=off)
        mgr.save(base[:off], mgr.snapshot(cache))
    _, cached_len = mgr.lookup(base, model)
    assert cached_len == 40, f"最長一致でない: {cached_len}"
    print("OK: 最長プレフィックス一致")


def test_dedup_and_min_tokens(tmp):
    """同一キー再保存のスキップと、短すぎるプロンプトの保存拒否。"""
    mgr = KVCacheManager(cache_dir=tmp)
    tokens = list(range(200, 230))
    cache = make_cache(len(tokens), seed=9)
    snap = mgr.snapshot(cache)
    mgr.save(tokens, snap)
    key = mgr._hash_prefix(tokens, len(tokens))
    assert wait_disk(mgr, key), "ディスク書込みが完了しない"
    mtime = os.path.getmtime(mgr._meta_path(key))
    time.sleep(0.05)
    mgr.save(tokens, snap)  # 同一 → ディスク書き直しなし
    assert os.path.getmtime(mgr._meta_path(key)) == mtime, "dedup が効いていない"

    short = list(range(MIN_SAVE_TOKENS - 1))
    mgr.save(short, mgr.snapshot(make_cache(len(short))))
    assert mgr._hash_prefix(short, len(short)) not in mgr._caches, "短小保存が通った"
    print("OK: dedup + 最小トークン数ガード")


def test_v1_purge(tmp):
    """旧形式（version なし）エントリが初期化時に削除されること。"""
    key = "deadbeef" * 8
    with open(os.path.join(tmp, f"{key}.json"), "w") as f:
        json.dump({"hash": key, "offset": 100}, f)  # v1: version キーなし
    KVCacheManager(cache_dir=tmp)
    assert not os.path.exists(os.path.join(tmp, f"{key}.json")), "v1 が残っている"
    print("OK: 旧形式パージ")


def main():
    tmp = tempfile.mkdtemp(prefix="elfmoon_kvtest_")
    try:
        mgr, tokens = test_roundtrip_and_immutability(tmp)
        key = mgr._hash_prefix(tokens, len(tokens))
        assert wait_disk(mgr, key), "ディスク書込みが完了しない"
        test_disk_restore(tmp, tokens)
        test_longest_prefix(tmp)
        test_dedup_and_min_tokens(tmp)
        test_v1_purge(tmp)
        print("\n全テスト成功 ✅")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
