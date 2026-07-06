"""モジュール①: on-disk per-expert 量子化重みストア。

融合テンソルではなく (layer, expert) 単位で個別に保存し、
mmap経由で必要な1個だけを高速ロードできるようにする。
これが「コールドexpertをSSDからストリーム」する土台。
"""
import os
import mlx.core as mx

# Qwen3-Coder-30B-A3B 近似の既定形状
DEFAULT_DIM = 2048
DEFAULT_INTER = 768
BITS = 4
GROUP = 64


def _quant(w):
    wq, s, b = mx.quantize(w, group_size=GROUP, bits=BITS)
    return wq, s, b


class ExpertStore:
    """(layer, expert) → gate/up/down の量子化重みを扱う。

    v1 は 1 expert = 1 safetensors ファイル。mx.load は mmap 相当で読み、
    ページキャッシュに乗っていれば温、無ければSSDコールド。
    """

    def __init__(self, path, dim=DEFAULT_DIM, inter=DEFAULT_INTER):
        self.path = path
        self.dim = dim
        self.inter = inter
        os.makedirs(path, exist_ok=True)

    def _file(self, layer, expert):
        return os.path.join(self.path, f"l{layer}_e{expert}.safetensors")

    def per_expert_bytes(self):
        """1 expert の概算バイト数（キャッシュ予算計算用）。"""
        f = self._file(0, 0)
        return os.path.getsize(f) if os.path.exists(f) else 0

    def load(self, layer, expert):
        """1 expert の重みを dict で返す（mx.array 群）。"""
        w = mx.load(self._file(layer, expert))
        return w if isinstance(w, dict) else {"_": w}

    # --- テスト用: 合成expertを生成 ---
    def generate_synthetic(self, n_layers, n_experts, seed=0):
        mx.random.seed(seed)
        shapes = {
            "gate": (self.inter, self.dim),
            "up": (self.inter, self.dim),
            "down": (self.dim, self.inter),
        }
        for l in range(n_layers):
            for e in range(n_experts):
                f = self._file(l, e)
                if os.path.exists(f):
                    continue
                d = {}
                for name, (o, inp) in shapes.items():
                    wq, s, b = _quant(mx.random.normal((o, inp)))
                    d[f"{name}.wq"], d[f"{name}.s"], d[f"{name}.b"] = wq, s, b
                mx.save_safetensors(f, d)
        mx.eval()


def expert_ffn(x, w):
    """SwiGLU FFN を量子化重みで計算。x:[T, dim] → [T, dim]。"""
    g = mx.quantized_matmul(x, w["gate.wq"], w["gate.s"], w["gate.b"],
                            transpose=True, group_size=GROUP, bits=BITS)
    u = mx.quantized_matmul(x, w["up.wq"], w["up.s"], w["up.b"],
                            transpose=True, group_size=GROUP, bits=BITS)
    h = (g * mx.sigmoid(g)) * u
    return mx.quantized_matmul(h, w["down.wq"], w["down.s"], w["down.b"],
                              transpose=True, group_size=GROUP, bits=BITS)
