import argparse

import torch
import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


def read_texts(input_path):
   
    if input_path.lower().endswith(".tsv"):
        df = pd.read_csv(input_path, sep="\t", header=0, names=["text", "gloss", "category"])
        return df["text"].tolist()
    with open(input_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def generate_batch(model, tokenizer, tgt_lang_id, batch_texts, device, max_len, num_beams):
    inputs = tokenizer(
        batch_texts, return_tensors="pt",
        padding=True, truncation=True, max_length=max_len
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=max_len,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2,
            length_penalty=1.0,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Text -> gloss inference (mBART-50)")
    parser.add_argument("input", help="File .tsv or .txt")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--model-dir", default="/content/drive/MyDrive/Mbart(cosine)_model")
    parser.add_argument("--lang", default="vi_VN")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=2)
    args = parser.parse_args()

    tokenizer = MBart50TokenizerFast.from_pretrained(args.model_dir, src_lang=args.lang, tgt_lang=args.lang)
    tgt_lang_id = tokenizer.lang_code_to_id[args.lang]

    model = MBartForConditionalGeneration.from_pretrained(args.model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    texts = read_texts(args.input)

    preds = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]
        preds.extend(generate_batch(model, tokenizer, tgt_lang_id, batch, device, args.max_len, args.num_beams))

    for t, p in zip(texts, preds):
        print(f"{t}\t->\t{p}")

    if args.output:
        pd.DataFrame({"text": texts, "gloss": preds}).to_csv(args.output, sep="\t", index=False)
        print(f"\nĐã lưu {len(texts)} dòng vào: {args.output}")


if __name__ == "__main__":
    main()
