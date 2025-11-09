# --- filename: eval/run_eval.py
import argparse
import json
import torch

from utils.scoring import load_model_and_tokenizer, compare_pair


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    p.add_argument("--spm", required=True, help="SentencePiece model path")
    p.add_argument("--pairs", required=True, help="JSONL with {seq_a, seq_b}")
    p.add_argument("--out", required=True, help="Output JSONL with {choice, nll_a, nll_b}")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model, sp = load_model_and_tokenizer(args.ckpt, args.spm, device)

    with open(args.pairs, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            ex = json.loads(line)
            choice, nlla, nllb = compare_pair(model, sp, ex["seq_a"], ex["seq_b"], max_len=args.max_len, device=device)
            ex_out = {"choice": choice, "nll_a": nlla, "nll_b": nllb}
            fout.write(json.dumps(ex_out) + "\n")
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
