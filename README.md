# README – BabyLM Strict MVP

This README explains how to train and evaluate the **Strict-track MVP** using the included scripts. The code is designed for clarity, reproducibility, and full compliance with the 2025 BabyLM challenge rules.

---

## 1. Environment Setup

```bash
conda create -y -n babylm python=3.10
conda activate babylm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install sentencepiece transformers datasets accelerate bitsandbytes peft einops pillow
```

---

## 2. Prepare Data

The official BabyLM corpus is available from the competition website. For the **Strict track**, use the 100M-word text corpus (child-directed + adult simplified text).

```bash
mkdir -p data_root/strict
# Place your training file as: data_root/strict/train.txt
# and optionally a validation file: data_root/strict/valid.txt
```

---

## 3. Train the Tokenizer

Train a 32k unigram SentencePiece model **only on the BabyLM text**.

```bash
python scripts/train_spm.py \
  --input ./data_root/strict/bnc_sentences.txt  \
  --vocab_size 32000 \
  --model_prefix spm32k_bpe \
  --model_type bpe \
  --character_coverage 0.9995 
```

This creates `spm32k_unigram.model` and `spm32k_unigram.vocab`.

---

## 4. Format Data for Span-Infill Objective

Span infilling provides partial denoising supervision using sentinel tokens.

```bash
python scripts/format_span_infill.py \
  --input ./data_root/strict/bnc_sentences.txt \
  --output ./data_root/strict/bnc_sentences_span.txt \
  --mask_ratio 0.15
```

This writes a masked version of each line with `<extra_id_n>` sentinels.

---

## 5. Train the Model (Strict Track)

```bash
python train_strict.py \
  --train ./data_root/strict/train_span.txt \
  --spm ./spm32k_unigram.model \
  --out ./ckpts/strict \
  --buckets 256 512 1024 \
  --batch_size 8 \
  --lr 3e-4 \
  --max_words 2e8
```

**Key points:**
- Tracks and logs *whitespace-separated word exposures*.
- Automatically checkpoints at milestones: 1M→10M→100M words.
- Stops once `max_words` is reached (configurable; ≤1B for leaderboard).
- Uses AMP and gradient scaling for efficient GPU utilization.

---

## 6. Evaluate Model

### 6.1 Generate Evaluation Pairs (example)

Create a JSONL file with pairs to compare:
```json
{"seq_a": "The cat sits on the mat.", "seq_b": "The cat sit on mat."}
```
Save as `pairs.jsonl`.

### 6.2 Run Evaluation

```bash
python eval/run_eval.py \
  --ckpt ./ckpts/strict/ckpt_words_10000000.pt \
  --spm ./spm32k_unigram.model \
  --pairs ./pairs.jsonl \
  --out ./preds/strict_eval.jsonl
```

Each line in `strict_eval.jsonl` contains:
```json
{"choice": 0, "nll_a": 1.23, "nll_b": 2.11}
```
(`choice=0` means `seq_a` is preferred, i.e., lower NLL.)

---

## 7. Checkpoint Accounting and Compliance

- **Checkpoints** are saved under `./ckpts/strict/ckpt_words_<N>.pt`.
- `utils/accounting.py` ensures exposure limits and milestone saves.
- The model’s config and optimizer state are stored for reproducibility.

---

## 8. Directory Summary

```
babylm-mvp/
├── models/u_transformer.py       # Unified transformer model
├── utils/accounting.py           # Exposure tracking + milestones
├── utils/scoring.py              # LL / pseudo-LL functions
├── scripts/train_spm.py          # Train tokenizer
├── scripts/format_span_infill.py # Mask spans for infill objective
├── train_strict.py               # Main training loop
└── eval/run_eval.py              # Zero-shot pairwise evaluation
```

---

## 9. Extending Beyond MVP

Later phases can reuse this setup for the **Multimodal** and **Interaction** tracks:
- Add a small Perceiver cross-attention adapter for images.
- Add a text-only teacher feedback loop for interaction.

The Strict MVP provides the solid base architecture and training logic required for all tracks.

---

**End of README**
