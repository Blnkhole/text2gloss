# coding: utf-8
"""
compute_metrics callback passed to Seq2SeqTrainer.
"""

import numpy as np

from metrics import bleu, chrf, rouge, wer_list, token_accuracy, sequence_accuracy


def make_compute_metrics(tokenizer):
    """Bind compute_metrics to a tokenizer so Trainer can call it with
    just (preds, labels)."""

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.where(
            np.array(preds).astype(int) >= 0, np.array(preds).astype(int), tokenizer.pad_token_id
        )
        labels = np.where(
            np.array(labels).astype(int) >= 0, np.array(labels).astype(int), tokenizer.pad_token_id
        )

        decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)]

        bleu_res = bleu(decoded_labels, decoded_preds)
        chrf_res = chrf(decoded_labels, decoded_preds)
        rouge_res = rouge(decoded_labels, decoded_preds)
        wer_res = wer_list(decoded_labels, decoded_preds)
        tok_acc = token_accuracy(decoded_labels, decoded_preds)
        seq_acc = sequence_accuracy(decoded_labels, decoded_preds)

        return {
            "bleu": bleu_res["bleu"],
            "bleu_1": bleu_res["bleu_1"],
            "bleu_2": bleu_res["bleu_2"],
            "bleu_3": bleu_res["bleu_3"],
            "bleu_4": bleu_res["bleu_4"],
            "chrf": chrf_res,
            "rouge1": rouge_res["rouge1"],
            "rouge2": rouge_res["rouge2"],
            "rougeL": rouge_res["rougeL"],
            "wer": wer_res["wer"],
            "token_acc": tok_acc,
            "seq_acc": seq_acc,
        }

    return compute_metrics
