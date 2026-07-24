"""ElfMoon 対話型チャットCLI。

モデルを一度だけロード＆ストリーミング化し、以降は対話ループで
何度でもプロンプトを投げられる。会話履歴を保持する。

使い方:
    cd elfmoon
    python3 chat.py                       # 常駐 6144 (既定モデル)
    python3 chat.py --model 80b           # ELFMOON_MODELS_ROOT/80b を使用
    python3 chat.py --model 80b 1200      # モデル指定 + 省メモリ
    python3 chat.py --no-think            # 思考プロセスを非表示
    python3 chat.py --fast                # 高速モード（top_k=6、実測~1.4-1.6x、品質トレードオフ）
    python3 chat.py --list                # 利用可能なモデル一覧

環境変数 ELFMOON_TOP_K=N でも指定可（--fast より優先、ストリーミング MoE のみ有効）。
"""

import logging
import fcntl
import os
import select
import sys
import termios
import time
import tty

# プロジェクトルートをパスに追加（model_v4.py の elfmoon. インポート用）
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

# 一部モデルのカスタムtokenizer実装が動作に無関係なWARNINGログを出すため抑制する
# （例: Kimi-Linearの tokenization_kimi.py が encode() 呼び出しごとに警告ログを出す）。
logging.disable(logging.WARNING)

from mlx_lm import load, stream_generate
from mlx_lm.utils import load_model
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache
from stream_model import MODELS_ROOT, list_models, resolve_model, wire_streaming
from pathlib import Path

SYSTEM = "You are an expert coding assistant. Write clean, correct, concise code."
MAX_TOKENS = 16384
MAX_HISTORY = 8
TEMP = 0.4
# プレフィルのチャンク幅。stream_generate 既定(512)は融合gather経路の閾値未満で
# 高速化されないため大きくする。api_server.py と同じ環境変数で連動。
PREFILL_STEP = int(os.environ.get("ELFMOON_PREFILL_STEP", "4096"))
# KV キャッシュ永続化（api_server と同じ kv_manager を利用）。ELFMOON_KVC=0 で無効。
KVC = os.environ.get("ELFMOON_KVC", "1") != "0"
# 対話 CLI では KVC の情報ログがプロンプト表示に割り込むため既定で抑制（エラーは出る）
os.environ.setdefault("ELFMOON_KVC_LOG", "0")


def _read_utf8_char(fd):
    b0 = os.read(fd, 1)
    if not b0:
        return None
    c0 = b0[0]
    if c0 < 0x80:
        return chr(c0)
    if c0 < 0xC0:
        return "\ufffd"
    need = 2 if c0 < 0xE0 else 3 if c0 < 0xF0 else 4
    raw = b0
    for _ in range(need - 1):
        more = os.read(fd, 1)
        if not more:
            break
        raw += more
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return "\ufffd"


def _read_esc_seq(fd):
    r, _, _ = select.select([fd], [], [], 0.05)
    if not r:
        return None
    first = os.read(fd, 1)
    if not first:
        return None
    if first == b"[":  # CSI: read until final byte (0x40-0x7E)
        seq = first
        for _ in range(15):
            r, _, _ = select.select([fd], [], [], 0.05)
            if not r:
                break
            cb = os.read(fd, 1)
            if not cb:
                break
            seq += cb
            if cb[0] in range(0x40, 0x7F):
                break
        return seq
    if first == b"O":  # SS3: read one more byte
        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            cb = os.read(fd, 1)
            if cb:
                return first + cb
        return first
    if first[0] in range(0x40, 0x7F):  # Two-char sequence
        return first
    return first


def _read_until_paste_end(fd):
    chars = []
    while True:
        ch = _read_utf8_char(fd)
        if ch is None:
            raise EOFError
        if ch == "\x1b":
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                seq = _read_esc_seq(fd)
                if seq == b"[201~":
                    return "".join(chars)
            chars.append(ch)
            continue
        if ch == "\x7f":
            if chars:
                chars.pop()
            continue
        chars.append(ch)
        sys.stdout.write(ch)
        sys.stdout.flush()


def _read_line_raw(fd):
    """Read one line in raw mode. Returns the line (without newline)."""
    chars = []
    while True:
        ch = _read_utf8_char(fd)
        if ch is None:
            raise EOFError
        if ch in "\n\r":
            print()
            return "".join(chars)
        if ch == "\x7f":
            if chars:
                chars.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        if ch == "\x1b":
            r, _, _ = select.select([fd], [], [], 0.05)
            if r:
                seq = _read_esc_seq(fd)
                if seq == b"[200~":
                    paste_content = _read_until_paste_end(fd)
                    return paste_content
            elif chars:
                sys.stdout.write("\b \b" * len(chars))
                sys.stdout.flush()
                chars = []
            continue
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x04":
            raise EOFError
        if ch == "\t" or ch.isprintable() or ch.isspace():
            chars.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()


def read_user_input(prompt):
    """Read user input in raw mode. Detects paste instantly, ESC to clear."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(prompt, end="", flush=True)
    try:
        tty.setcbreak(fd)
        a = termios.tcgetattr(fd)
        a[3] &= ~(termios.ECHO | termios.ICANON | termios.ISIG)
        termios.tcsetattr(fd, termios.TCSANOW, a)

        first = _read_line_raw(fd)

        extra_data = b""
        r, _, _ = select.select([fd], [], [], 0.1)
        while r:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            extra_data += chunk
            r, _, _ = select.select([fd], [], [], 0.2)

        if extra_data:
            r, _, _ = select.select([fd], [], [], 0.4)
            while r:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                extra_data += chunk
                r, _, _ = select.select([fd], [], [], 0.2)

        if extra_data:
            extra = extra_data.decode("utf-8", errors="replace").split("\n")
            if extra[-1] == "":
                extra = extra[:-1]
            lines = [first] + extra if first else extra
            for l in extra:
                print(f"  {l}")
            print("\033[2m（空行で確定）\033[0m")
            while True:
                r, _, _ = select.select([fd], [], [], 0)
                while r:
                    chunk = os.read(fd, 4096)
                    if not chunk:
                        break
                    extra_data += chunk
                    r, _, _ = select.select([fd], [], [], 0)
                if extra_data:
                    more_parts = (
                        extra_data.decode("utf-8", errors="replace")
                        .rstrip("\n")
                        .split("\n")
                    )
                    extra_data = b""
                    for mp in more_parts:
                        lines.append(mp)
                        print(f"  {mp}")
                more = _read_line_raw(fd)
                if not more:
                    break
                lines.append(more)
            return "\n".join(lines).strip()

        if not first:
            return None
        return first.strip()

    except (KeyboardInterrupt, EOFError):
        raise
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except:
            pass


def _strip_think(text_iter, no_think, in_think=False):
    """Strip <think>/</think> blocks from stream if no_think is set.

    以下3形式に対応:
    1. <think>...</think> 形式（Qwen標準、templateが開きタグを出力）
    2. 開きタグ無しで推論内容→</think> 形式（一部fine-tuneモデル）
    3. in_think=True: Thinking専用モデル（templateがプロンプト側に <think> を
       置くため出力にタグが無い）。最初から think 内として </think> まで捨てる。
    """
    if not no_think:
        yield from text_iter
        return

    if in_think:
        buf = ""
        for piece in text_iter:
            buf += piece
            if "</think>" in buf:
                after = buf.split("</think>", 1)[1]
                if after.strip():
                    yield after.lstrip("\n")
                yield from text_iter
                return
            # </think> がタグ途中で分割されて届く場合に備え末尾だけ保持
            buf = buf[-16:]
        return

    buf = ""
    for piece in text_iter:
        buf += piece
        # 開きタグがあれば以降を discard
        if "<think" in buf:
            buf = buf[buf.find("<think") + len("<think>") :]
            while "</think>" not in buf:
                buf = next(text_iter, "")
                if not buf:
                    return
            after = buf.split("</think>", 1)[1]
            if after:
                yield after
            yield from text_iter
            return
        if "</think>" in buf:
            after = buf.split("</think>", 1)[1]
            if after:
                yield after
            yield from text_iter
            return
        yield buf
        buf = ""
    if buf:
        yield buf


def main():
    argv = sys.argv[1:]

    if "--list" in argv:
        models = list_models()
        print(f"利用可能なモデル（ELFMOON_MODELS_ROOT={MODELS_ROOT}）:")
        for name, has_store, is_native in models:
            if is_native:
                print(f"  {name}  ✅ オンメモリ動作")
            elif has_store:
                print(f"  {name}")
            else:
                print(f"  {name}  ⚠️ store/ 未生成（integrate.py split_all が必要）")
        if not models:
            print("  (見つかりません)")
        return

    no_think = "--no-think" in argv
    perf = "--perf" in argv
    fast = "--fast" in argv
    if fast and not os.environ.get("ELFMOON_TOP_K"):
        # \u5b9f\u6e2c ~1.4-1.6x\uff08\u54c1\u8cea\u30c8\u30ec\u30fc\u30c9\u30aa\u30d5\u3042\u308a\u30fbopt-in\uff09\u3002\u660e\u793a env \u304c\u3042\u308c\u3070\u305d\u3061\u3089\u3092\u512a\u5148\u3002
        os.environ["ELFMOON_TOP_K"] = "6"
    model_name = None
    if "--model" in argv:
        idx = argv.index("--model")
        model_name = argv[idx + 1].strip().replace("\u3000", "")
        argv = argv[:idx] + argv[idx + 2 :]
    cap_strs = [a for a in argv if a not in ("--no-think", "--perf", "--fast")]
    cap = int(cap_strs[0]) if cap_strs else 6144

    model_path, store_dir = resolve_model(model_name)
    if KVC:
        # KV キャッシュをモデル別に分離（モデル間の同一プロンプト衝突＝形状不一致を防ぐ）
        try:
            from kv_manager import kv_manager

            kv_manager.set_namespace(os.path.basename(model_path))
        except Exception:
            pass
    import json

    with open(os.path.join(model_path, "config.json")) as f:
        _cfg = json.load(f)
    _model_type = _cfg.get("model_type", "")

    _sampler_kwargs = {}
    _model_name = os.path.basename(model_path).lower()
    if _model_type == "gemma4":
        TEMP = 1.0
        _sampler_kwargs = dict(temp=TEMP, top_p=0.95, top_k=64)
    elif "ornith" in _model_name:
        # Ornith 推奨: agentic coding temp=1.0, top_p=1.0
        TEMP = 1.0
        _sampler_kwargs = dict(temp=TEMP, top_p=1.0, top_k=64)
    elif "glm" in _model_name:
        # GLM 推奨: temp=1.0, top_p=0.95, min_p=0.01（repeat_penalty=1.0は未対応）
        TEMP = 1.0
        _sampler_kwargs = dict(temp=TEMP, top_p=0.95, min_p=0.01)
    elif "agents-a1" in _model_name:
        # Agents-A1 推奨: temp=0.85, top_p=0.95, top_k=20（presence_penalty は未対応）
        TEMP = 0.85
        _sampler_kwargs = dict(temp=TEMP, top_p=0.95, top_k=20)
    else:
        TEMP = 0.4

    mode = "性能" if perf else "省メモリ"
    print(f"モデル: {model_path}（type={_model_type}）")
    print(f"モデルをロード中...（{mode}モード, capacity={cap}）")
    t0 = time.perf_counter()

    if _model_type == "deepseek_v4":
        from model_v4 import DeepseekV4Model
        from stream_model import _wire_deepseek_v4

        model = DeepseekV4Model(model_path, fused_quant=True)
        cache, _ = _wire_deepseek_v4(
            model, cap, top_k=6, store_dir=store_dir, model_path=model_path
        )
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path)
    else:
        _mp = Path(model_path)
        model, _ = load_model(_mp, lazy=True)
        # トークナイザのロード（カスタムtokenizer_class対応）
        try:
            _, tok = load(model_path, lazy=True)
        except Exception:
            from transformers import PreTrainedTokenizerFast
            from tokenizers import Tokenizer

            tk = Tokenizer.from_file(str(_mp / "tokenizer.json"))
            tok = PreTrainedTokenizerFast(tokenizer_object=tk)
            ct_path = _mp / "chat_template.jinja"
            if ct_path.exists():
                tok.chat_template = ct_path.read_text()
            # config.json から EOS トークンを動的設定
            import json

            with open(_mp / "config.json") as __cfgf:
                __cfg = json.load(__cfgf)
            __eos_raw = __cfg.get("eos_token_id", 1)
            __eos_ids = __eos_raw if isinstance(__eos_raw, list) else [__eos_raw]
            tok.eos_token_id = __eos_ids[0]
            from mlx_lm.tokenizer_utils import TokenizerWrapper

            tok = TokenizerWrapper(tok, eos_token_ids=__eos_ids)
        if _model_type == "gemma4" or not os.path.isdir(
            os.path.join(model_path, "store")
        ):
            cache = None
        else:
            cache, _ = wire_streaming(
                model, cap, perf=perf, store_dir=store_dir, model_path=model_path
            )

    import mlx.core as mx

    # mx.compile: 全denseモデルを高速化（streaming MoE は store/ があるので除外）
    if _model_type != "deepseek_v4" and not os.path.isdir(
        os.path.join(model_path, "store")
    ):
        try:
            mx.compile(model.__call__)
        except Exception:
            pass

    print(
        f"準備完了（{time.perf_counter() - t0:.0f}秒）。会話をどうぞ。'exit' か Ctrl-D で終了。\n"
    )

    messages = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            user = read_user_input("\n\033[1;36mあなた>\033[0m ")
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break
        if user is None:
            continue
        if user.lower() in ("exit", "quit"):
            print("終了します。")
            break
        if not user:
            continue

        messages.append({"role": "user", "content": user})
        if len(messages) > 1 + MAX_HISTORY * 2:
            messages = [messages[0]] + messages[-MAX_HISTORY * 2 :]

        prompt = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=not no_think,
        )

        print("\033[1;32mElfMoon>\033[0m ", end="", flush=True)
        resp, t, answer_t = "", time.perf_counter(), 0.0
        n = 0
        think_s = 0.0

        try:
            if _model_type == "deepseek_v4":
                # DeepseekV4Model は独自 generate() を使用（非ストリーミング）
                import mlx.core as mx

                model.reset_state()
                ids = tok.encode(prompt)
                out_ids = model.generate(mx.array(ids), max_new=MAX_TOKENS)
                text = tok.decode(out_ids, skip_special_tokens=True)
                # prompt 部分を除去して応答のみ抽出
                resp = text[len(tok.decode(ids, skip_special_tokens=True)) :]
                n = len(out_ids) - len(ids)
                answer_t = time.perf_counter()
                print(resp, end="", flush=True)
            else:
                _sampler = make_sampler(**_sampler_kwargs)
                _gen_kwargs = dict(
                    model=model,
                    tokenizer=tok,
                    prompt=prompt,
                    max_tokens=MAX_TOKENS,
                    sampler=_sampler,
                    prefill_step_size=PREFILL_STEP,
                )
                _kvc_save_ids, _kvc_snap = None, None
                _prefill_n = 0
                _prefill_t0 = None
                _prefill_t1_ref = [None]
                if KVC:
                    # 会話履歴部分の KV を再利用し、毎ターンの全履歴再プレフィルを回避する。
                    # 失敗時は従来経路にフォールバック（会話は止めない）。
                    try:
                        from kv_manager import kv_manager
                        import mlx.core as mx

                        prompt_ids = tok.encode(prompt)
                        nogen = tok.apply_chat_template(
                            messages,
                            add_generation_prompt=False,
                            tokenize=False,
                            enable_thinking=not no_think,
                        )
                        nogen_ids = tok.encode(nogen)
                        boundary = 0
                        for bi in range(min(len(nogen_ids), len(prompt_ids))):
                            if prompt_ids[bi] != nogen_ids[bi]:
                                break
                            boundary = bi + 1
                        cached_cache, cached_len = kv_manager.lookup(prompt_ids, model)
                        if cached_cache is not None and cached_len < len(prompt_ids):
                            prompt_cache = cached_cache
                        else:
                            prompt_cache = make_prompt_cache(model)
                            cached_len = 0
                        _prefill_n = len(prompt_ids) - cached_len
                        _prefill_t0 = time.perf_counter()
                        if cached_len < boundary:
                            remaining = prompt_ids[cached_len:boundary]
                            for ci in range(0, len(remaining), PREFILL_STEP):
                                model(
                                    mx.array([remaining[ci : ci + PREFILL_STEP]]),
                                    cache=prompt_cache,
                                )
                            _kvc_snap = kv_manager.snapshot(prompt_cache)
                            _kvc_save_ids = prompt_ids[:boundary]
                        if cached_len:
                            print(
                                f"\033[2m（KVC: {cached_len}tok 再利用）\033[0m ",
                                end="",
                                flush=True,
                            )
                        _gen_kwargs.update(
                            prompt=prompt_ids[boundary:], prompt_cache=prompt_cache
                        )
                    except Exception:
                        pass  # 従来経路（全プレフィル）で続行
                if _prefill_n == 0:
                    prompt_ids = (
                        tok.encode(prompt) if isinstance(prompt, str) else prompt
                    )
                    _prefill_n = len(prompt_ids) if isinstance(prompt_ids, list) else 0
                    _prefill_t0 = time.perf_counter()
                generator = stream_generate(**_gen_kwargs)

                # Thinking専用モデル: template がプロンプト側に <think> を置き
                # 出力に開きタグが現れないため、最初から think 内として扱う
                _gp = _gen_kwargs["prompt"]
                if isinstance(_gp, str):
                    _in_think = no_think and _gp.rstrip().endswith("<think>")
                else:
                    _tail = tok.decode(list(_gp[-8:])) if len(_gp) else ""
                    _in_think = no_think and _tail.rstrip().endswith("<think>")

                _first_raw_t = [0.0]

                def _texts():
                    for out in generator:
                        if _first_raw_t[0] == 0.0:
                            _first_raw_t[0] = time.perf_counter()
                            if _prefill_t1_ref[0] is None:
                                _prefill_t1_ref[0] = _first_raw_t[0]
                        yield out.text

                if _in_think:
                    print("\033[2m（思考中…）\033[0m", end="", flush=True)
                for piece in _strip_think(_texts(), no_think, in_think=_in_think):
                    if answer_t == 0.0:
                        answer_t = time.perf_counter()
                        if _in_think:
                            # 思考中表示を消して回答を書き始める
                            print(
                                "\r\033[K\033[1;32mElfMoon>\033[0m ", end="", flush=True
                            )
                        piece = piece.lstrip("\n")  # 回答冒頭の空行を除去
                        if not piece:
                            answer_t = 0.0  # 空白のみなら次piece を冒頭扱い
                            continue
                    print(piece, end="", flush=True)
                    resp += piece
                    n += 1
                # 隠した思考の時間（応答 t/s の分母には乗せない）
                think_s = (
                    (answer_t - _first_raw_t[0])
                    if (_in_think and answer_t and _first_raw_t[0])
                    else 0.0
                )
                if _kvc_save_ids is not None:
                    try:
                        from kv_manager import kv_manager

                        kv_manager.save(_kvc_save_ids, _kvc_snap)
                    except Exception:
                        pass
        except Exception as e:
            print(f"\n\033[1;31m[エラー] 生成が中断されました: {e}\033[0m")

        elapsed = (
            (time.perf_counter() - answer_t) if answer_t else (time.perf_counter() - t)
        )
        _pf_elapsed = (
            _prefill_t1_ref[0] - _prefill_t0
            if (
                _prefill_t0 is not None
                and _prefill_t1_ref[0] is not None
                and _prefill_t1_ref[0] > _prefill_t0
            )
            else None
        )
        pf_info = ""
        if _pf_elapsed is not None and _pf_elapsed > 0:
            pf_speed = _prefill_n / _pf_elapsed
            pf_info = f"プリフィル {_prefill_n}tok {pf_speed:.0f}tok/s ／ "
        hit = f", 命中率{cache.hit_rate * 100:.0f}%" if cache else ""
        think_info = f"思考 {think_s:.1f}s ／ " if think_s > 0 else ""
        print(
            f"\n\033[2m（{think_info}{pf_info}出力 {n} tokens, {n / elapsed:.1f} tok/s{hit}）\033[0m"
        )
        messages.append({"role": "assistant", "content": resp})


if __name__ == "__main__":
    main()
