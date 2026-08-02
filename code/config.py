# coding: utf-8

"""
Central configuration for the gloss2text project.
"""

# model 
MODEL_NAME = "facebook/mbart-large-50"
LANG = "vi_VN"
MAX_LEN = 64

# data
DATA_DIR = "/content/drive/MyDrive/gloss2text"
TRAIN_PATH = f"{DATA_DIR}/train.tsv"
DEV_PATH = f"{DATA_DIR}/dev.tsv"
TEST_PATH = f"{DATA_DIR}/test.tsv"
PREDICTIONS_PATH = f"{DATA_DIR}/predictions_text2gloss.tsv"
TSV_COLUMNS = ["text", "gloss", "category"]

# checkpoints
OUTPUT_DIR = "/content/drive/MyDrive/Mbart(cosine)_model"
CHECKPOINT_DIR = "/content/checkpoints"

# generation (shared by training callback, eval.py and generate.py)
NUM_BEAMS = 2
NO_REPEAT_NGRAM_SIZE = 2
REPETITION_PENALTY = 1.2
LENGTH_PENALTY = 1.0

# training 
SEED = 42
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8  # used by eval.py, can differ from training eval batch size
LEARNING_RATE = 1e-5
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.06
NUM_TRAIN_EPOCHS = 6
LOGGING_STEPS = 100
SAVE_TOTAL_LIMIT = 2
FP16 = True
METRIC_FOR_BEST_MODEL = "bleu_4"

# misc 
SAMPLE_LOG_EVERY_N_STEPS = 1000
NUM_SAMPLES_TO_LOG = 5
