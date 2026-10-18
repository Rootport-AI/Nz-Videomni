"""CPU-only: tokenize the selftest prompt the way engine_h3 does, print ids."""
import json, sys
from transformers import AutoTokenizer
import transformers, tokenizers
from engine_h3.worker import _SELFTEST_PROMPT
from engine_h3.text_encoder import ensure_extra_special_tokens
tok = AutoTokenizer.from_pretrained("models/MiniMaxH3/TextEncoder/qwen3vl_tokenizer")
sp = ensure_extra_special_tokens(tok)
ids = list(tok(_SELFTEST_PROMPT, add_special_tokens=False)["input_ids"])
print(json.dumps({"exe": sys.executable, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                  "tok_class": type(tok).__name__, "n": len(ids), "ids": ids, "special": sp,
                  "pieces": tok.convert_ids_to_tokens(ids)}, ensure_ascii=False))
