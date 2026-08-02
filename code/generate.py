# coding: utf-8
"""
Standalone inference script. Run this to try the fine-tuned model on
a few sentences without re-running training.

Usage:
    python generate.py "Tôi ghét nhất con dê" "Tôi thích nhất thịt heo"
"""

import sys

import config
from model_utils import build_tokenizer, build_model, get_device, generate_batch


def load_for_inference(model_dir: str = config.OUTPUT_DIR, lang: str = config.LANG):
    tokenizer, tgt_lang_id = build_tokenizer(model_dir, lang)
    model = build_model(model_dir, tgt_lang_id)
    model.eval()
    device = get_device()
    model.to(device)
    return model, tokenizer, tgt_lang_id, device


def generate(text: str, model, tokenizer, tgt_lang_id, device) -> str:
    return generate_batch(model, tokenizer, tgt_lang_id, [text], device)[0]


if __name__ == "__main__":
    model, tokenizer, tgt_lang_id, device = load_for_inference()

    sentences = sys.argv[1:] or [
        "Tôi ghét nhất con dê",
        "Tôi thích nhất thịt heo",
    ]

    for s in sentences:
        print(generate(s, model, tokenizer, tgt_lang_id, device))
