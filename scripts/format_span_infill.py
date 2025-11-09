# --- filename: scripts/format_span_infill.py
import argparse
import random
import re

SENTINELS = [f"<extra_id_{i}>" for i in range(100)]


def mask_line(line: str, mask_ratio: float = 0.15, min_span: int = 1, max_span: int = 5) -> str:
    tokens = line.strip().split()
    if not tokens:
        return line.strip()
    n_to_mask = max(1, int(len(tokens) * mask_ratio))
    spans = []
    i = 0
    while i < n_to_mask:
        span_len = random.randint(min_span, max_span)
        start = random.randint(0, max(0, len(tokens) - span_len))
        spans.append((start, min(len(tokens), start + span_len)))
        i += span_len
    # merge overlapping spans
    spans = sorted(spans)
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    out = []
    cursor = 0
    sid = 0
    for s, e in merged:
        if s > cursor:
            out.extend(tokens[cursor:s])
        out.append(SENTINELS[sid])
        sid += 1
        cursor = e
    out.extend(tokens[cursor:])
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mask_ratio", type=float, default=0.15)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(mask_line(line, args.mask_ratio) + "\n")
    print(f"[span] wrote {args.output}")


if __name__ == "__main__":
    main()

