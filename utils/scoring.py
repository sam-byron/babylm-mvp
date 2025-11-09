# --- filename: utils/scoring.py
from typing import Tuple
import torch
import sentencepiece as spm

from models.u_transformer import UTransformerLM, UTransformerConfig


def load_model_and_tokenizer(ckpt_path: str, spm_model_path: str, device: str = "cuda"):
    sp = spm.SentencePieceProcessor(model_file=spm_model_path)
    state = torch.load(ckpt_path, map_location=device)
    cfg = UTransformerConfig(**state["config"])  # config stored in checkpoint
    model = UTransformerLM(cfg).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, sp


def sentence_to_ids(sp: spm.SentencePieceProcessor, text: str, max_len: int, bos_id: int = 1, eos_id: int = 2) -> torch.Tensor:
    ids = sp.encode(text, out_type=int)
    if bos_id is not None and bos_id >= 0:
        ids = [bos_id] + ids
    if eos_id is not None and eos_id >= 0:
        ids = ids + [eos_id]
    ids = ids[:max_len]
    return torch.tensor([ids], dtype=torch.long)


@torch.no_grad()
def avg_nll(model: UTransformerLM, sp: spm.SentencePieceProcessor, text: str, max_len: int = 1024, device: str = "cuda") -> float:
    bos_id = getattr(model.cfg, "bos_id", 1)
    eos_id = getattr(model.cfg, "eos_id", 2)
    ids = sentence_to_ids(sp, text, max_len, bos_id=bos_id, eos_id=eos_id).to(device)
    logits, _ = model(ids)
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = ids[:, 1:]
    nll = -logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    length = max(1, tgt.numel())
    return float(nll.sum() / length)


@torch.no_grad()
def compare_pair(model: UTransformerLM, sp: spm.SentencePieceProcessor, a: str, b: str, max_len: int = 1024, device: str = "cuda") -> Tuple[int, float, float]:
    nlla = avg_nll(model, sp, a, max_len, device)
    nllb = avg_nll(model, sp, b, max_len, device)
    choice = 0 if nlla < nllb else 1
    return choice, nlla, nllb