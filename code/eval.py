# coding: utf-8
"""
Evaluate a fine-tuned checkpoint on test.tsv and dump predictions.
"""

import pandas as pd

import config
from data import load_split
from model_utils import build_tokenizer, build_model, get_device, generate_batch
from metrics import report_all


def main():
    tokenizer, tgt_lang_id = build_tokenizer(config.OUTPUT_DIR, config.LANG)
    model = build_model(config.OUTPUT_DIR, tgt_lang_id)
    model.eval()
    device = get_device()
    model.to(device)

    df = load_split(config.TEST_PATH)
    texts = df["text"].tolist()
    refs = df["gloss"].tolist()

    preds = []
    for i in range(0, len(texts), config.EVAL_BATCH_SIZE):
        batch = texts[i : i + config.EVAL_BATCH_SIZE]
        preds.extend(generate_batch(model, tokenizer, tgt_lang_id, batch, device))

    metrics = report_all(refs, preds)
    print(metrics)

    pd.DataFrame({"text": texts, "ref": refs, "pred": preds}).to_csv(
        config.PREDICTIONS_PATH, sep="\t", index=False
    )


if __name__ == "__main__":
    main()
