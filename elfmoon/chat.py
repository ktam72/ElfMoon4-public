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
import os
import sys
import time

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
from stream_model import MODELS_ROOT, list_models, resolve_model, wire_streaming
from pathlib import Path

SYSTEM = "You are an expert coding assistant. Write clean, correct, concise code."
MAX_TOKENS = 16384
MAX_HISTORY = 8
TEMP = 0.4
# プレフィルのチャンク幅。stream_generate 既定(512)は融合gather経路の閾値未満で
# 高速化されないため大きくする。api_server.py と同じ環境変数で連動。
PREFILL_STEP = int(os.environ.get("ELFMOON_PREFILL_STEP", "4096"))


def _strip_think(text_iter, no_think):
    """Strip <think>/</think> blocks from stream if no_think is set.

    以下2形式に対応:
    1. <think>...</think> 形式（Qwen標準、templateが開きタグを出力）
    2. 開きタグ無しで推論内容→</think> 形式（一部fine-tuneモデル）
    """
    if not no_think:
        yield from text_iter
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
    else:
        TEMP = 0.4

    mode = "性能" if perf else "省メモリ"
    print(f"モデル: {model_path}（type={_model_type}）")
    print(f"モデルをロード中...（{mode}モード, capacity={cap}）")
    t0 = time.perf_counter()

    if _model_type == "deepseek_v4":
        from model_v4 import DeepseekV4Model

        model = DeepseekV4Model(model_path, fused_quant=True)
        cache = None
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
            user = input("\n\033[1;36mあなた>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了します。")
            break
        if user.lower() in ("exit", "quit"):
            print("終了します。")
            break
        if not user:
            continue

        messages.append({"role": "user", "content": user})
        if len(messages) > 1 + MAX_HISTORY * 2:
            messages = [messages[0]] + messages[-MAX_HISTORY * 2 :]

        if _model_type == "deepseek_v4":
            from encoding_dsv4 import encode_messages

            thinking_mode = "chat" if no_think else "thinking"
            prompt = encode_messages(messages, thinking_mode=thinking_mode)
        else:
            prompt = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=not no_think,
            )

        print("\033[1;32mElfMoon>\033[0m ", end="", flush=True)
        resp, t, answer_t = "", time.perf_counter(), 0.0
        n = 0

        try:
            if _model_type == "deepseek_v4":
                import mlx.core as mx

                ids = tok.encode(prompt)
                ids_arr = mx.array(ids, dtype=mx.int64)
                new_ids = []
                answer_t = 0.0
                MAX_V4 = min(512, MAX_TOKENS)  # 0.3 tok/s では少量ずつ
                for token_id in model.generate_stream(
                    ids_arr,
                    max_new=MAX_V4,
                    temperature=TEMP,
                    top_p=0.9,
                ):
                    new_ids.append(token_id)
                    if answer_t == 0.0:
                        answer_t = time.perf_counter()
                    piece = tok.decode([token_id], skip_special_tokens=True)
                    print(piece, end="", flush=True)
                n = len(new_ids)
                resp = tok.decode(new_ids, skip_special_tokens=True)
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
                generator = stream_generate(**_gen_kwargs)

                def _texts():
                    for out in generator:
                        yield out.text

                for piece in _strip_think(_texts(), no_think):
                    if answer_t == 0.0:
                        answer_t = time.perf_counter()
                    print(piece, end="", flush=True)
                    resp += piece
                    n += 1
        except Exception as e:
            print(f"\n\033[1;31m[エラー] 生成が中断されました: {e}\033[0m")

        elapsed = (
            (time.perf_counter() - answer_t) if answer_t else (time.perf_counter() - t)
        )
        hit = f", 命中率{cache.hit_rate * 100:.0f}%" if cache else ""
        print(f"\n\033[2m（{n} tokens, {n / elapsed:.1f} tok/s{hit}）\033[0m")
        messages.append({"role": "assistant", "content": resp})


if __name__ == "__main__":
    main()
