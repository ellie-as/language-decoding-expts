from __future__ import annotations

from transformers import AutoTokenizer


def load_tokenizer(name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def decode_token_ids(tokenizer, token_ids, skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens).strip()
