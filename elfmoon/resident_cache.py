"""モジュール②: バイト予算つき LRU 常駐キャッシュ。

ホットexpertをGPU/ユニファイドメモリに常駐させ、予算超過時はLRUで退避。
DS4 の ds4_ssd_auto_cache_plan（予算×4/5、非routed差引、expert数算出）に対応。
命中率がElfMoonの速度を決める中核指標なので hit/miss を記録する。
"""
from collections import OrderedDict


def plan_cache_experts(budget_bytes, non_expert_bytes, per_expert_bytes,
                       max_experts=None, headroom=0.8):
    """常駐予算からホットexpert数を算出（DS4方式）。"""
    target = int(budget_bytes * headroom)
    cache_bytes = max(0, target - non_expert_bytes)
    n = cache_bytes // per_expert_bytes if per_expert_bytes else 0
    if max_experts is not None:
        n = min(n, max_experts)
    return max(1, int(n))


class ResidentCache:
    """(layer, expert) → 重み dict の LRU キャッシュ。

    capacity は「常駐expert数」。get(key, loader) で命中/ミスを扱う。
    ミス時に loader() で読み込み、満杯なら最古を退避（参照を落として解放）。
    """

    def __init__(self, capacity):
        self.capacity = max(1, int(capacity))
        self._d = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __contains__(self, key):
        return key in self._d

    def prime(self, key, weights):
        """ホットリスト起動時プライム用: 命中/ミス統計を汚さず投入。"""
        self._d[key] = weights
        self._d.move_to_end(key)
        self._evict()

    def get(self, key, loader):
        if key in self._d:
            self.hits += 1
            self._d.move_to_end(key)
            return self._d[key]
        self.misses += 1
        w = loader()
        self._d[key] = w
        self._d.move_to_end(key)
        self._evict()
        return w

    def _evict(self):
        while len(self._d) > self.capacity:
            self._d.popitem(last=False)  # 最古(LRU)を退避

    @property
    def hit_rate(self):
        t = self.hits + self.misses
        return self.hits / t if t else 0.0

    def stats(self):
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hit_rate, "resident": len(self._d),
                "capacity": self.capacity}
