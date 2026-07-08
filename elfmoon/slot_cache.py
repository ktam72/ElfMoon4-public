"""SlotResidentCache: 融合バッファ + グローバルLRU + 層別スロット管理。

層別3D配列 [per_layer, ...] × 40層で gather_qmm 性能を確保しつつ、
グローバル LRU で全層の容量を共有する。
"""

from collections import OrderedDict

import mlx.core as mx

from expert_store import GROUP, BITS

INTER = 512
DIM = 2048
EXPERT_SHAPES = {
    "gate.wq": (INTER, DIM // 8),
    "gate.s": (INTER, DIM // GROUP),
    "gate.b": (INTER, DIM // GROUP),
    "up.wq": (INTER, DIM // 8),
    "up.s": (INTER, DIM // GROUP),
    "up.b": (INTER, DIM // GROUP),
    "down.wq": (DIM, INTER // 8),
    "down.s": (DIM, INTER // GROUP),
    "down.b": (DIM, INTER // GROUP),
}


def _buf_name(key):
    return key.replace(".", "_")


def _make_buffers(per_layer, n_layers):
    bufs = {name: [] for name in EXPERT_SHAPES}
    for _ in range(n_layers):
        for name, shape in EXPERT_SHAPES.items():
            dtype = mx.uint32 if "wq" in name else mx.bfloat16
            bufs[name].append(mx.zeros((per_layer, *shape), dtype=dtype))
    return bufs


class SlotResidentCache:
    """融合スロットバッファ + グローバルLRU + 層別3D配列。

    容量 S を層数で均等分割（S//40）し、各層に独立した
    3D バッファを割り当てる。退避は同一層内でのみ行う。
    """

    def __init__(self, capacity, store, n_layers=40, min_per_layer=0):
        self.n_layers = n_layers
        self._store = store
        per_layer = capacity // n_layers
        if min_per_layer and per_layer < min_per_layer:
            per_layer = min_per_layer
        self.per_layer = per_layer

        _bufs = _make_buffers(self.per_layer, n_layers)
        for name, buflist in _bufs.items():
            setattr(self, _buf_name(name), buflist)

        self._lru = OrderedDict()
        self._layer_maps = [{} for _ in range(n_layers)]
        self._free = [list(range(self.per_layer - 1, -1, -1)) for _ in range(n_layers)]
        self.hits = 0
        self.misses = 0

    def get_layer_bufs(self, layer):
        return tuple(getattr(self, _buf_name(name))[layer] for name in EXPERT_SHAPES)

    def _write_slot(self, layer, slot, w):
        self.gate_wq[layer][slot] = w["gate.wq"]
        self.gate_s[layer][slot] = w["gate.s"]
        self.gate_b[layer][slot] = w["gate.b"]
        self.up_wq[layer][slot] = w["up.wq"]
        self.up_s[layer][slot] = w["up.s"]
        self.up_b[layer][slot] = w["up.b"]
        self.down_wq[layer][slot] = w["down.wq"]
        self.down_s[layer][slot] = w["down.s"]
        self.down_b[layer][slot] = w["down.b"]

    def _evict_one(self, layer, in_use):
        """同一層内の LRU 最古（かつ in_use 外）を1件退避。"""
        for key in list(self._lru.keys()):
            if key[0] == layer and key not in in_use:
                l, e = key
                slot = self._lru.pop(key)
                del self._layer_maps[l][e]
                return slot
        raise RuntimeError(
            f"layer {layer}: 退避可能スロットなし"
            f"（per_layer={self.per_layer}, in_use={len(in_use)}）"
        )

    def get_slots(self, layer, expert_ids):
        """expert_ids (list[int]) に対応するスロット番号のリストを返す。"""
        lm = self._layer_maps[layer]
        result = []
        miss = []
        for eid in expert_ids:
            key = (layer, eid)
            if key in self._lru:
                self.hits += 1
                self._lru.move_to_end(key)
                result.append(lm[eid])
            else:
                self.misses += 1
                miss.append(eid)
                result.append(None)

        if not miss:
            return result

        in_use = {(layer, eid) for eid in expert_ids}
        for eid in miss:
            free = self._free[layer]
            if free:
                slot = free.pop()
            else:
                slot = self._evict_one(layer, in_use)
            key = (layer, eid)
            w = self._store.load(layer, eid)
            self._write_slot(layer, slot, w)
            self._lru[key] = slot
            lm[eid] = slot
            idx = expert_ids.index(eid)
            result[idx] = slot

        bufs = [getattr(self, _buf_name(name))[layer] for name in EXPERT_SHAPES]
        mx.eval(*bufs)
        return result

    def get(self, key, loader=None):
        layer, expert = key
        slots = self.get_slots(layer, [expert])
        slot = slots[0]
        return {
            name: getattr(self, _buf_name(name))[layer][slot] for name in EXPERT_SHAPES
        }

    def prime(self, layer, expert):
        key = (layer, expert)
        if key in self._lru:
            self._lru.move_to_end(key)
            return
        free = self._free[layer]
        if free:
            slot = free.pop()
        else:
            slot = self._evict_one(layer, set())
        w = self._store.load(layer, expert)
        self._write_slot(layer, slot, w)
        self._lru[key] = slot
        self._layer_maps[layer][expert] = slot
        bufs = [getattr(self, _buf_name(name))[layer] for name in EXPERT_SHAPES]
        mx.eval(*bufs)

    @property
    def hit_rate(self):
        t = self.hits + self.misses
        return self.hits / t if t else 0.0

    def stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses)
            if (self.hits + self.misses)
            else 0.0,
            "resident": len(self._lru),
            "capacity": self.n_layers * self.per_layer,
        }
