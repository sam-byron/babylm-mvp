# --- filename: train_strict.py
import argparse
import os

import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import sentencepiece as spm
from typing import Any, Optional

from models.u_transformer import UTransformerConfig, UTransformerLM
from utils.accounting import ExposureAccounting


class LineDataset(IterableDataset):
    """Yield one sequence per line with BOS/EOS. Also returns whitespace-word count.

    Optional prefetch: if prefetch_lines > 0, we read up to that many non-empty
    lines (or all if larger than file) into memory once to avoid repeated disk IO.
    """
    def __init__(self, text_path: str, sp_model: str, add_bos: bool = True, add_eos: bool = True, max_len: int = 1024, prefetch_lines: int = 0):
        super().__init__()
        self.text_path = text_path
        self.sp = spm.SentencePieceProcessor(model_file=sp_model)
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.bos = self.sp.bos_id() if add_bos and self.sp.bos_id() >= 0 else None
        self.eos = self.sp.eos_id() if add_eos and self.sp.eos_id() >= 0 else None
        self.max_len = max_len
        self.prefetch_lines = prefetch_lines
        self._buffer = None
        if prefetch_lines != 0:
            self._buffer = []
            count = 0
            with open(self.text_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    self._buffer.append(line)
                    count += 1
                    if 0 < self.prefetch_lines <= count:
                        break

    def __iter__(self):
        if self._buffer is not None:
            source_iter = iter(self._buffer)
        else:
            source_iter = (r.strip() for r in open(self.text_path, "r", encoding="utf-8"))
        for line in source_iter:
            if not line:
                continue
            ids = self.sp.encode(line, out_type=int)
            if self.bos is not None:
                ids = [self.bos] + ids
            if self.eos is not None:
                ids = ids + [self.eos]
            ids = ids[: self.max_len]
            words = len(line.split())
            yield torch.tensor(ids, dtype=torch.long), words


class TokenBudgetBatcher(IterableDataset):
    """
    Wraps a (sequence, words) IterableDataset and emits pre-padded batches that
    respect a fixed token budget using length buckets. Each emitted batch tensor
    has shape [N, B] where B is the bucket size used for that batch, and N is
    chosen as floor(token_budget / B) (at least 1). This keeps the rectangular
    token count per batch <= token_budget with minimal padding.
    """

    def __init__(self, base: IterableDataset, pad_id: int, buckets, token_budget: int):
        super().__init__()
        self.base = base
        self.pad_id = pad_id
        self.buckets = sorted(list(buckets))
        self.token_budget = int(token_budget)

    def _capacity(self, bucket_size: int) -> int:
        cap = self.token_budget // bucket_size
        return max(1, cap)

    def _collate_bucket(self, samples, bucket_size: int):
        # samples: list of (tensor, words)
        seqs = [s[0] for s in samples]
        words = sum(s[1] for s in samples)
        x = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=self.pad_id)
        # right-pad to the fixed bucket width so all batches in this bucket are identical width
        if x.size(1) < bucket_size:
            pad_cols = bucket_size - x.size(1)
            x = torch.nn.functional.pad(x, (0, pad_cols), value=self.pad_id)
        return x, words

    def __iter__(self):
        # Maintain small queues per bucket and emit when capacity reached.
        queues = {b: [] for b in self.buckets}
        caps = {b: self._capacity(b) for b in self.buckets}

        for seq, words in self.base:
            L = int(seq.size(0))
            # choose the smallest bucket that fits L
            bucket = None
            for b in self.buckets:
                if L <= b:
                    bucket = b
                    break
            if bucket is None:
                bucket = self.buckets[-1]
                # truncate hard just in case
                seq = seq[:bucket]
                L = bucket

            queues[bucket].append((seq, words))
            # emit as many full batches as possible for this bucket
            cap = caps[bucket]
            while len(queues[bucket]) >= cap:
                batch_samples = queues[bucket][:cap]
                queues[bucket] = queues[bucket][cap:]
                yield self._collate_bucket(batch_samples, bucket)

        # flush any remaining (may be underfilled)
        for b in self.buckets:
            if queues[b]:
                yield self._collate_bucket(queues[b], b)


def _sample_lengths(text_path: str, sp: spm.SentencePieceProcessor, add_bos: bool, add_eos: bool, max_len_cap: int, sample: int) -> list:
    lens = []
    bos = sp.bos_id() if add_bos and sp.bos_id() >= 0 else None
    eos = sp.eos_id() if add_eos and sp.eos_id() >= 0 else None
    n = 0
    with open(text_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            ids = sp.encode(line, out_type=int)
            if bos is not None:
                ids = [bos] + ids
            if eos is not None:
                ids = ids + [eos]
            L = min(len(ids), max_len_cap)
            lens.append(L)
            n += 1
            if sample > 0 and n >= sample:
                break
    return lens


def _suggest_buckets(lens: list, token_budget: int, min_bucket: int, max_bucket: int, quantiles=(50, 75, 90, 95), multiple: int = 64, max_k: int = 5):
    import numpy as np
    if len(lens) == 0:
        # fallback
        return [min(max_bucket, max(min_bucket, 512))]
    arr = np.array(lens)
    qs = np.percentile(arr, list(quantiles)).astype(int).tolist()
    # candidate sizes snapped to nearest multiple and bounded
    def snap(b):
        b = max(min_bucket, min(max_bucket, int(b)))
        # snap to nearest multiple of 'multiple'
        low = (b // multiple) * multiple
        high = low + multiple
        cand = min([low if low >= min_bucket else multiple, high if high <= max_bucket else high], key=lambda x: (abs(x - b), token_budget % max(1, x)))
        return max(min_bucket, min(max_bucket, cand))

    snapped = sorted({snap(b) for b in qs})
    # ensure divisibility preference: sort by (token_budget % B, B)
    snapped = sorted(snapped, key=lambda B: (token_budget % B, B))
    # limit number of buckets
    if len(snapped) > max_k:
        snapped = snapped[:max_k]
    # always include max_bucket to cap long sequences if necessary
    if max_bucket not in snapped:
        snapped.append(max_bucket)
    # unique and sort ascending
    buckets = sorted(set(snapped))
    # drop buckets that would yield capacity < 1 (shouldn't happen) and tiny ones under min_bucket
    buckets = [b for b in buckets if b >= min_bucket and (token_budget // b) >= 1]
    # final guard
    if not buckets:
        buckets = [min(max_bucket, max(min_bucket, multiple))]
    return buckets


def collate_lines(batch, pad_id: int):
    # legacy single-batch collate (unused with TokenBudgetBatcher but kept for reference)
    seqs = [b[0] for b in batch]
    words = sum(b[1] for b in batch)
    x = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    return x, words


def _debug_attn(model: Any, batch: torch.Tensor, pad_id: int, step: int):
    """Runtime-only attention debug for layer 0 (pad token attention mass).
    Uses loose typing (Any) to avoid static analysis noise; failures are swallowed.
    """
    try:
        layers = getattr(model, "layers", None)
        if not layers or len(layers) == 0:
            return
        sample_layer = layers[0]
        tok_emb = getattr(model, "tok_emb", None)
        norm1 = getattr(sample_layer, "norm1", None)
        attn_module = getattr(sample_layer, "attn", None)
        if not (callable(tok_emb) and callable(norm1) and callable(attn_module)):
            return
        Bdbg, Tdbg = batch.size()
        causal_mask = getattr(model, "causal_mask", None)
        attn_mask_dbg: Optional[torch.Tensor] = None
        if isinstance(causal_mask, torch.Tensor):
            attn_mask_dbg = causal_mask[:, :, :Tdbg, :Tdbg]
        key_pad_dbg = (batch == pad_id).unsqueeze(1).unsqueeze(2)
        if attn_mask_dbg is not None:
            attn_mask_dbg = attn_mask_dbg + key_pad_dbg * (-1e4)
        with torch.no_grad():
            emb_dbg = tok_emb(batch)
            norm_x_dbg = norm1(emb_dbg)
            attn_out = attn_module(norm_x_dbg, attn_mask=attn_mask_dbg, return_attn=True)
            if isinstance(attn_out, tuple) and len(attn_out) == 2:
                _, attn_weights_dbg = attn_out
                denom = key_pad_dbg.float().sum().clamp_min(1)
                pad_mass_dbg = (attn_weights_dbg * key_pad_dbg.float()).sum() / denom
                print(f"[dbg-attn] step={step} pad_mass={pad_mass_dbg.item():.6f} T={Tdbg}")
    except Exception as e:
        print(f"[dbg-attn] failed: {e}")


def save_checkpoint(model, cfg, optimizer, step_words, out_dir, scaler=None, step: int | None = None):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_words_{step_words}.pt")
    payload = {
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "words_seen": step_words,
    }
    if scaler is not None:
        try:
            payload["scaler"] = scaler.state_dict()
        except Exception:
            pass
    if step is not None:
        payload["step"] = int(step)
    torch.save(payload, path)
    print(f"[ckpt] saved {path}")


def _find_resume_checkpoint(out_dir: str, resume_spec: str | None) -> str | None:
    """Return path to checkpoint given a spec ('latest' or file path)."""
    if not resume_spec:
        return None
    if os.path.isfile(resume_spec):
        return resume_spec
    if resume_spec == "latest":
        if not os.path.isdir(out_dir):
            return None
        candidates = [f for f in os.listdir(out_dir) if f.startswith("ckpt_words_") and f.endswith(".pt")]
        if not candidates:
            return None
        def _words_from_name(name: str) -> int:
            try:
                base = os.path.splitext(name)[0]
                return int(base.split("ckpt_words_")[-1])
            except Exception:
                return -1
        candidates.sort(key=lambda n: (_words_from_name(n), os.path.getmtime(os.path.join(out_dir, n))), reverse=True)
        return os.path.join(out_dir, candidates[0])
    # treat as path under out_dir
    path = os.path.join(out_dir, resume_spec)
    return path if os.path.isfile(path) else None


def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Backend precision optimizations (CUDA only)
    if device == "cuda":
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        # Scaled Dot-Product Attention backend toggles (PyTorch 2.x)
        sdp = (args.sdp or "auto").lower()
        try:
            torch.backends.cuda.enable_flash_sdp(sdp in ("flash", "auto"))
            torch.backends.cuda.enable_math_sdp(sdp in ("math", "auto"))
            torch.backends.cuda.enable_mem_efficient_sdp(sdp in ("mem", "auto"))
        except Exception:
            pass  # silently ignore if not available

    sp = spm.SentencePieceProcessor(model_file=args.spm)
    vocab_size = sp.vocab_size()
    pad_id = vocab_size  # external pad
    if sp.pad_id() >= 0:
        pad_id = sp.pad_id()
    else:
        pad_id = vocab_size       # put PAD just past the last vocab id

    cfg = UTransformerConfig(
        vocab_size=vocab_size,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        kv_heads=1,
        mlp_mult=4.0,
        dropout=0.1,
        max_seq_len=max(args.buckets),
        bos_id=sp.bos_id() if sp.bos_id() >= 0 else 1,
        eos_id=sp.eos_id() if sp.eos_id() >= 0 else 2,
        pad_id=pad_id,
    )

    model = UTransformerLM(cfg).to(device)

    # Optimizer with fused / foreach support if available
    fused_opt = None
    foreach_opt = None
    if device == "cuda":
        if hasattr(optim.AdamW, "fused") and args.fused_adam:
            fused_opt = True
        if hasattr(optim.AdamW, "foreach") and args.foreach_optim:
            foreach_opt = True
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=fused_opt,
        foreach=foreach_opt,
    )

    # AMP / BF16 selection
    use_amp = (device == "cuda") and (not args.no_amp)
    amp_dtype = torch.bfloat16 if (args.bf16 and device == "cuda") else torch.float16
    # GradScaler only needed for FP16, not BF16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    # Optionally auto-derive buckets from data sample
    if args.auto_buckets:
        sample_lens = _sample_lengths(args.train, sp, add_bos=True, add_eos=True, max_len_cap=max(args.buckets), sample=args.bucket_sample)
        suggested = _suggest_buckets(sample_lens, token_budget=args.token_budget, min_bucket=args.min_bucket, max_bucket=max(args.buckets),
                                     quantiles=tuple(args.bucket_quantiles), multiple=args.bucket_multiple, max_k=args.bucket_max_k)
        buckets = suggested
        print(f"[buckets] auto-derived {buckets} from {len(sample_lens)} samples")
    else:
        buckets = [b for b in args.buckets if b >= args.min_bucket]
        buckets = sorted(buckets, reverse=bool(args.bucket_desc))
    ds_lines = LineDataset(args.train, args.spm, add_bos=True, add_eos=True, max_len=max(buckets), prefetch_lines=args.prefetch_lines)
    ds = TokenBudgetBatcher(ds_lines, pad_id=pad_id, buckets=buckets, token_budget=args.token_budget)
    # DataLoader just yields already batched tensors; set batch_size=None to avoid extra batching
    if args.workers > 0:
        dl = DataLoader(
            ds,
            batch_size=None,
            num_workers=int(args.workers),
            pin_memory=(device == "cuda"),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor),
            shuffle=False,
            drop_last=False,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=None,
            num_workers=0,
            pin_memory=(device == "cuda"),
            shuffle=False,
            drop_last=False,
        )

    milestones = []
    milestones += [int(1e6 * i) for i in range(1, 11)]
    milestones += [int(1e7 * i) for i in range(2, 11)]
    accounting = ExposureAccounting(milestones=milestones, max_words_seen=int(args.max_words), save_dir=args.out)

    model.train()
    step = 0
    accum = 0.0
    # token-weighted running average loss over entire training
    running_token_sum = 0
    running_loss_token_sum = 0.0
    # throughput / padding metrics
    window_t0 = time.perf_counter()
    window_tokens = 0
    bucket_stats = {}

    # Resume support: load checkpoint if requested
    start_words_seen = 0
    start_step = 0
    if args.resume:
        ckpt_path = _find_resume_checkpoint(args.out, args.resume)
        if ckpt_path:
            print(f"[resume] loading {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            # Rebuild model if vocab/config differs? Assume same.
            model.load_state_dict(ckpt["model"], strict=True)
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"[resume] optimizer state load failed: {e}")
            if use_amp and amp_dtype == torch.float16 and "scaler" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler"])
                except Exception as e:
                    print(f"[resume] scaler state load failed: {e}")
            start_words_seen = int(ckpt.get("words_seen", 0))
            start_step = int(ckpt.get("step", 0))
            accounting.words_seen = start_words_seen
            print(f"[resume] words_seen={start_words_seen} step={start_step}")
        else:
            print(f"[resume] specified resume target '{args.resume}' not found; starting fresh")

    # initialize step from resume
    if start_step > 0:
        step = start_step

    for batch, words in dl:
        batch = batch.to(device, non_blocking=(device == "cuda"))
        targets = batch.clone()
        targets[targets == pad_id] = -100  # ignore pad positions
        # invariant: batch.numel() == batch.size(0)*batch.size(1) ~= token_budget (within bucket rounding)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            # _, loss = model(batch, targets=targets)
            # attach scalar step only if safe; ignore static analyzer complaints
            try:
                setattr(model, "_debug_step_int", int(step))
            except Exception:
                pass
            _, loss = model(batch, targets=targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # if using AMP
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()

        step += 1
        accum += float(loss.item())
        # track token-weighted running average
        with torch.no_grad():
            active_tokens = (targets != -100).sum().item()
            running_token_sum += int(active_tokens)
            running_loss_token_sum += float(loss.item()) * float(active_tokens)
            total_tokens_rect = batch.size(0) * batch.size(1)
            pad_tokens = total_tokens_rect - active_tokens
            pad_frac = pad_tokens / max(1, total_tokens_rect)
            window_tokens += active_tokens
            b_len = batch.size(1)
            s = bucket_stats.setdefault(b_len, {"active":0, "rect":0, "batches":0, "pad_frac_sum":0.0})
            s["active"] += active_tokens
            s["rect"] += total_tokens_rect
            s["batches"] += 1
            s["pad_frac_sum"] += pad_frac
        crossed = accounting.add(int(words))

        # Attention debug (layer 0) every debug_attn_every steps
        if getattr(args, "debug_attn", False) and step % max(1, getattr(args, "debug_attn_every", 500)) == 0:
            _debug_attn(model, batch, pad_id, step)

        if step % args.log_every == 0:
            tokens_rect = batch.size(0) * batch.size(1)
            avg_loss_window = accum/args.log_every
            avg_loss_global = running_loss_token_sum / max(1, running_token_sum)
            # throughput timing (CPU wall clock)
            secs = max(1e-6, time.perf_counter() - window_t0)
            toks_per_sec = window_tokens / secs
            print(f"[train] step={step} words_seen={accounting.words_seen} tokens={tokens_rect} loss_window={avg_loss_window:.4f} loss_avg={avg_loss_global:.4f} toks_sec={toks_per_sec:.0f}")
            accum = 0.0
            window_tokens = 0
            window_t0 = time.perf_counter()

        for i in crossed:
            ws = accounting.milestones[i]
            save_checkpoint(model, cfg, optimizer, ws, args.out, scaler=scaler if (use_amp and amp_dtype==torch.float16) else None, step=step)

        if accounting.should_stop():
            print("[train] Reached exposure ceiling. Stopping.")
            break

    accounting.dump_state()
    # final average loss across all seen tokens
    if running_token_sum > 0:
        final_avg = running_loss_token_sum / running_token_sum
        print(f"[train] final_avg_loss={final_avg:.4f} over_tokens={running_token_sum}")
    # bucket summary
    if bucket_stats:
        print("[buckets] summary")
        for blen, stat in sorted(bucket_stats.items()):
            active = stat["active"]
            rect = stat["rect"]
            batches = stat["batches"]
            avg_pad_frac = stat["pad_frac_sum"] / max(1, batches)
            eff = active / max(1, rect)
            print(f"  len={blen} batches={batches} eff_tokens={active} rect_tokens={rect} eff_ratio={eff:.3f} avg_pad_frac={avg_pad_frac:.3f}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--spm", required=True)
    ap.add_argument("--out", default="./ckpts/strict")
    ap.add_argument("--buckets", type=int, nargs="+", default=[256,512,1024])
    ap.add_argument("--min_bucket", type=int, default=256, help="Drop buckets smaller than this size")
    ap.add_argument("--bucket_desc", action="store_true", help="Order buckets descending (longer sequences first)")
    ap.add_argument("--prefetch_lines", type=int, default=0, help="Prefetch this many lines into memory (0=disabled, -1=all)")
    # auto bucket optimization
    ap.add_argument("--auto_buckets", action="store_true", help="Derive bucket sizes from a sample of the data")
    ap.add_argument("--bucket_sample", type=int, default=50000, help="How many lines to sample for bucket derivation (<=0 means all)")
    ap.add_argument("--bucket_quantiles", type=int, nargs="+", default=[50,75,90,95], help="Quantiles used to derive candidate lengths")
    ap.add_argument("--bucket_multiple", type=int, default=64, help="Snap bucket sizes to nearest multiple of this")
    ap.add_argument("--bucket_max_k", type=int, default=5, help="Maximum number of buckets (excluding mandatory max)")
    ap.add_argument("--token_budget", type=int, default=8192, help="Approx tokens per batch (seq_len * batch_size)")
    ap.add_argument("--batch_size", type=int, default=8, help="(unused with token-budget batching; retained for backwards compat)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=4, help="Prefetch batches per worker")
    ap.add_argument("--persistent_workers", action="store_true", help="Keep DataLoader workers alive between epochs")
    ap.add_argument("--fused_adam", action="store_true", help="Enable fused AdamW if available")
    ap.add_argument("--foreach_optim", action="store_true", help="Enable foreach optimizer implementation if available")
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast on CUDA (no GradScaler)")
    ap.add_argument("--tf32", action="store_true", help="Allow TF32 matrix operations for speed on Ampere+")
    ap.add_argument("--sdp", type=str, default="auto", choices=["auto","flash","math","mem"], help="Scaled dot-product attention backend preference")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_words", type=float, default=2e8)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path or 'latest'")
    ap.add_argument("--debug_attn", action="store_true", help="Enable periodic attention debug print (layer 0 pad mass)")
    ap.add_argument("--debug_attn_every", type=int, default=250, help="Steps between attention debug prints")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    train(args)
