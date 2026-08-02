# coding: utf-8
"""
Data loading and tokenization for the gloss2text task.
"""

import pandas as pd
from datasets import Dataset

from config import TSV_COLUMNS


def load_split(path: str) -> pd.DataFrame:
    """Load a train/dev/test TSV split into a DataFrame with columns
    text, gloss, category."""
    return pd.read_csv(path, sep="\t", header=0, names=TSV_COLUMNS)


def load_hf_dataset(path: str) -> Dataset:
    """Load a TSV split directly into a HuggingFace Dataset."""
    return Dataset.from_pandas(load_split(path))


def make_preprocess_fn(tokenizer, max_len: int):
    """
    Build a 'preprocess(example)' function bound to a specific tokenizer.

    Padding tokens in the labels are replaced with -100 (ignored by the loss).
    """

    def preprocess(example):
        model_inputs = tokenizer(
            example["text"],
            max_length=max_len,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=example["gloss"],
            max_length=max_len,
            truncation=True,
            padding=False,
        )
        labels["input_ids"] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in seq]
            for seq in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess
