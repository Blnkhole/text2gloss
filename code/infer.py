import torch
import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from metrics import report_all

MODEL_DIR  = "/content/drive/MyDrive/Mbart(cosine)_model"
TEST_PATH  = "/content/drive/MyDrive/gloss2text/test.tsv"
LANG       = "vi_VN"
MAX_LEN    = 64
BATCH_SIZE = 8
NUM_BEAMS  = 2

tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_DIR, src_lang=LANG, tgt_lang=LANG)
TGT_LANG_ID = tokenizer.lang_code_to_id[LANG]

model = MBartForConditionalGeneration.from_pretrained(MODEL_DIR)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

df = pd.read_csv(TEST_PATH, sep="\t", header=0, names=["text", "gloss", "category"])
texts = df["text"].tolist()
refs  = df["gloss"].tolist()

def generate_batch(batch_texts):
    inputs = tokenizer(
        batch_texts, return_tensors="pt",
        padding=True, truncation=True, max_length=MAX_LEN
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=TGT_LANG_ID,
            max_length=MAX_LEN,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            length_penalty=1.0,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

preds = []
for i in range(0, len(texts), BATCH_SIZE):
    preds.extend(generate_batch(texts[i:i+BATCH_SIZE]))

metrics = report_all(refs, preds)
print(metrics)

pd.DataFrame({"text": texts, "ref": refs, "pred": preds}).to_csv(
    "/content/drive/MyDrive/gloss2text/predictions_text2gloss.tsv", sep="\t", index=False
)