

# --- filename: eval/run_blimp_eval.py
import argparse
import json
import os
import sys
from glob import glob
from collections import defaultdict
import logging

# Fix import path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    # Graceful fallback if tqdm not installed
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        return iterable if iterable is not None else []

from utils.scoring import load_model_and_tokenizer, avg_nll


SUPPORTED_GOOD_KEYS = [
    "sentence_good",
    "good_sentence",
    "good",
]
SUPPORTED_BAD_KEYS = [
    "sentence_bad",
    "bad_sentence",
    "bad",
]


def find_key(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def collect_blimp_files(blimp_dir, pattern):
    # Return mapping subset_name -> file_path
    paths = glob(os.path.join(blimp_dir, pattern))
    mapping = {}
    for p in sorted(paths):
        subset = os.path.splitext(os.path.basename(p))[0]
        mapping[subset] = p
    return mapping


def count_lines(path):
    """Count lines in a file for progress bar total."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


@torch.no_grad()
def evaluate_subset(model, sp, file_path, max_len, device, logger, progress=True):
    total = 0
    correct = 0
    margins = []  # nll_bad - nll_good (positive if good is better)

    n = count_lines(file_path)
    iterator = iter_jsonl(file_path)
    if progress:
        iterator = tqdm(iterator, total=n, desc=os.path.basename(file_path), leave=False)

    for i, ex in enumerate(iterator):
        if i >= 200:
            break  
        # gk = find_key(ex, SUPPORTED_GOOD_KEYS)
        # bk = find_key(ex, SUPPORTED_BAD_KEYS)
        good = ex["sentence_good"] 
        bad = ex["sentence_bad"]
        # if gk is None or bk is None:
        #     # Skip malformed line
        #     continue
        # good = ex[gk]
        # bad = ex[bk]
        nll_good = avg_nll(model, sp, good, max_len=max_len, device=device)
        nll_bad = avg_nll(model, sp, bad, max_len=max_len, device=device)
        total += 1
        if nll_good < nll_bad:
            correct += 1
        margins.append(nll_bad - nll_good)

    acc = (correct / total) if total > 0 else 0.0
    avg_margin = (sum(margins) / len(margins)) if margins else 0.0
    return {"total": total, "correct": correct, "accuracy": acc, "avg_margin": avg_margin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    ap.add_argument("--spm", required=True, help="SentencePiece model path")
    ap.add_argument("--blimp_dir", required=True, help="Directory containing BLiMP JSONL files")
    ap.add_argument("--pattern", default="*.jsonl", help="Glob for BLiMP files (default: *.jsonl)")
    ap.add_argument("--out", required=True, help="Output JSON report path")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    args = ap.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("blimp_eval")

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    logger.info(f"Device: {device}")
    logger.info(f"Loading checkpoint: {args.ckpt}")
    logger.info(f"Loading SentencePiece: {args.spm}")
    
    model, sp = load_model_and_tokenizer(args.ckpt, args.spm, device)
    logger.info("Model and tokenizer loaded successfully")

    files = collect_blimp_files(args.blimp_dir, args.pattern)
    logger.info(f"Found {len(files)} BLiMP subsets in {args.blimp_dir}")
    
    results = {}
    overall_total = 0
    overall_correct = 0

    # Progress bar over subsets
    for subset, path in tqdm(files.items(), desc="Evaluating subsets", total=len(files)):
        logger.debug(f"Processing {subset}")
        r = evaluate_subset(model, sp, path, max_len=args.max_len, device=device, logger=logger, progress=True)
        results[subset] = r
        overall_total += r["total"]
        overall_correct += r["correct"]
        logger.info(f"{subset}: {r['accuracy']:.4f} ({r['correct']}/{r['total']})")

    results["overall"] = {
        "total": overall_total,
        "correct": overall_correct,
        "accuracy": (overall_correct / overall_total) if overall_total > 0 else 0.0,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results written to: {args.out}")
    logger.info(f"Overall accuracy: {results['overall']['accuracy']:.4f} ({overall_correct}/{overall_total} pairs)")


if __name__ == "__main__":
    main()
