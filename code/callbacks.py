# coding: utf-8

import torch
from transformers import TrainerCallback

import config
from model_utils import generate_batch


class PrintBatchCallback(TrainerCallback):
    #Every SAMPLE_LOG_EVERY_N_STEPS logging steps

    def __init__(self, model, tokenizer, tgt_lang_id, val_dataset, raw_val_df):
        self.model = model
        self.tokenizer = tokenizer
        self.tgt_lang_id = tgt_lang_id
        self.raw_val_df = raw_val_df  # untokenized text/gloss for printing

    def on_log(self, args, state, control, **kwargs):
        if state.global_step % config.SAMPLE_LOG_EVERY_N_STEPS != 0 or state.global_step == 0:
            return

        self.model.eval()

        n = config.NUM_SAMPLES_TO_LOG
        texts = self.raw_val_df["text"].tolist()[:n]
        refs = self.raw_val_df["gloss"].tolist()[:n]

        with torch.no_grad():
            preds = generate_batch(
                self.model, self.tokenizer, self.tgt_lang_id, texts, self.model.device
            )

        print("\n===== SAMPLES =====")
        for t, r, p in zip(texts, refs, preds):
            print(f"T: {t}")
            print(f"R: {r}")
            print(f"P: {p}")
            print("-" * 30)

        self.model.train()
