# coding: utf-8
"""
Helpers to build/load the mBART model and tokenizer, and to run
generation with the shared decoding settings from config.py.
"""

import random
import os

import numpy as np
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

import config


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_tokenizer(model_dir: str = config.MODEL_NAME, lang: str = config.LANG):
    """Load a tokenizer from either the base model name or a fine-tuned
    checkpoint directory. Returns (tokenizer, tgt_lang_id)."""
    tokenizer = MBart50TokenizerFast.from_pretrained(
        model_dir, src_lang=lang, tgt_lang=lang
    )
    tgt_lang_id = tokenizer.lang_code_to_id[lang]
    return tokenizer, tgt_lang_id


def build_model(model_dir: str = config.MODEL_NAME, tgt_lang_id: int = None):
    """Load a model from either the base model name or a fine-tuned
    checkpoint directory. If tgt_lang_id is given, sets it as the
    forced BOS token for generation."""
    model = MBartForConditionalGeneration.from_pretrained(model_dir)
    if tgt_lang_id is not None:
        model.generation_config.forced_bos_token_id = tgt_lang_id
    return model


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def generate_batch(model, tokenizer, tgt_lang_id, texts, device, max_len=config.MAX_LEN):
    """Run generation on a batch of raw text strings using the shared
    decoding hyperparameters, and return decoded strings."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=max_len,
            num_beams=config.NUM_BEAMS,
            no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=config.REPETITION_PENALTY,
            length_penalty=config.LENGTH_PENALTY,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
