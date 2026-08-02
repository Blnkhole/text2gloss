# coding: utf-8
"""
Full fine-tunes mBART-50 on the gloss2text task.

"""

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

import config
from data import load_split, load_hf_dataset, make_preprocess_fn
from model_utils import set_seed, build_tokenizer, build_model
from compute_metrics import make_compute_metrics
from callbacks import PrintBatchCallback
from generate import generate as generate_text


def main():
    set_seed(config.SEED)

    tokenizer, tgt_lang_id = build_tokenizer(config.MODEL_NAME, config.LANG)

    train_dataset = load_hf_dataset(config.TRAIN_PATH)
    val_dataset = load_hf_dataset(config.DEV_PATH)
    val_df_raw = load_split(config.DEV_PATH)  # kept untokenized for callback printing

    preprocess = make_preprocess_fn(tokenizer, config.MAX_LEN)
    train_dataset = train_dataset.map(preprocess, batched=True)
    val_dataset = val_dataset.map(preprocess, batched=True)

    model = build_model(config.MODEL_NAME, tgt_lang_id)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.CHECKPOINT_DIR,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_ratio=config.WARMUP_RATIO,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=config.MAX_LEN,
        generation_num_beams=config.NUM_BEAMS,
        seed=config.SEED,
        data_seed=config.SEED,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        fp16=config.FP16,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=make_compute_metrics(tokenizer),
    )

    trainer.add_callback(
        PrintBatchCallback(model, tokenizer, tgt_lang_id, val_dataset, val_df_raw)
    )

    trainer.train()

    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

    # quick sanity check with the freshly trained model
    device = model.device
    print(generate_text("Tôi ghét nhất con dê", model, tokenizer, tgt_lang_id, device))
    print(generate_text("Tôi thích nhất thịt heo", model, tokenizer, tgt_lang_id, device))


if __name__ == "__main__":
    main()
