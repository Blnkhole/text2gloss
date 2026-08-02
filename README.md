# gloss2text (mBART-50 fine-tune)

Fine-tune `facebook/mbart-large-50` to translate Vietnamese text into gloss, and evaluate with BLEU / chrF(++) / ROUGE / WER.

## Project Architecture

```
gloss2text/
├── config.py             # All constants: paths, hyperparameters, model name...
├── data.py               # Reads TSV, tokenization + label masking (-100)
├── model_utils.py        # Build model/tokenizer, set_seed, generate_batch (shared decoding)
├── compute_metrics.py    # compute_metrics function for Seq2SeqTrainer (uses metrics.py)
├── callbacks.py          # PrintBatchCallback — logs prediction samples every N steps
├── metrics.py            # BLEU / chrF / ROUGE / WER — unchanged logic
├── train.py              # Orchestration: load data -> build model -> Trainer -> train -> save
├── eval.py               # Load trained checkpoint, run on test.tsv (with references), print metrics + save predictions
├── generate.py           # Quick inference on 1-2 sentences without re-running training
└── infer.py              # Translate any plain .tsv file
```

## How to Run

```bash
pip install transformers datasets sacrebleu evaluate rouge_score torch pandas

# 1. Modify paths in config.py if your data is not located at:
#    /content/drive/MyDrive/gloss2text/{train,dev,test}.tsv

# 2. Train (saves checkpoint to config.OUTPUT_DIR upon completion)
python train.py

# 3. Evaluate on test.tsv, which must have a gloss (reference) column
#    (prints metrics + saves predictions.tsv)
python eval.py

# 4. Quick inference on a few sentences using the trained checkpoint
python generate.py "Tôi ghét nhất con dê" "Tôi thích nhất thịt heo"

# 5. Inference on an arbitrary text file — no reference/gloss column required
python translate_file.py input.txt -o output.tsv
```

## Inference on a plain text file (no references)

Two ways to run inference on data that has no ground-truth gloss:

- **`generate.py`** — pass sentences directly as command-line arguments,
  prints the translation for each. Good for a quick spot check.
  ```bash
  python generate.py "Tôi ghét nhất con dê" "Tôi thích nhất thịt heo"
  ```

- **`infer.py`** — pass a `.tsv` file where each line is one
  input sentence. Prints `text -> gloss` for every line, and optionally
  saves a `.tsv` with columns `text`, `gloss`.
  ```bash
  python infer.py input.tsv -o output.tsv
  python infer.py input.tsv -o output.tsv --model-dir /path/to/checkpoint --lang vi_VN --batch-size 8
  ```

This is different from `eval.py`: `eval.py` expects a `.tsv` that already
has a `gloss` reference column, and computes BLEU/chrF/ROUGE/WER against
it. `translate_file.py`/`generate.py` don't require or compute any
reference-based metric — they're for translating new, unlabeled text.

If you're working outside this module layout (e.g. a single Colab cell),
`infer_text2gloss.py` is a flat, standalone equivalent of `translate_file.py`
with no imports from `config.py`/`model_utils.py` — just edit the
`MODEL_DIR`/`INPUT_PATH`/`OUTPUT_PATH` constants at the top and run it.

## Data Format

Each `.tsv` file (train/dev/test) contains 3 tab-separated columns without
requiring specific headers (the script automatically assigns column
names): `text`, `gloss`, and `category`.

For inference-only input (`translate_file.py`), use a plain `.txt` file
with one sentence per line — no `gloss`/`category` columns needed.

## Notes

- `set_seed(42)` is called at the beginning of `train.py` to ensure
  reproducibility (random, numpy, torch, cuDNN deterministic).
- `metric_for_best_model="bleu_4"` — The Trainer automatically selects
  the best checkpoint based on BLEU-4 on the dev set (configurable via
  `config.METRIC_FOR_BEST_MODEL`).
- Decoding hyperparameters (`num_beams`, `no_repeat_ngram_size`,
  `repetition_penalty`, `length_penalty`) live in `config.py` and are
  shared by the training sample-logging callback, `eval.py`,
  `generate.py`, and `translate_file.py`, so they can't drift out of
  sync between training-time logging and actual inference.
- To switch to a different model (other than mBART-50) or train on a
  different language pair, simply update `MODEL_NAME` / `LANG` in
  `config.py` — no changes to `train.py` or `eval.py` are required.
