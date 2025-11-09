# --- filename: scripts/train_spm.py
import argparse
import sentencepiece as spm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to plain text training data")
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--model_prefix", default="spm32k_unigram")
    p.add_argument("--character_coverage", type=float, default=0.9995)
    p.add_argument("--model_type", default="unigram", choices=["unigram", "bpe", "char", "word"])
    args = p.parse_args()

    spm.SentencePieceTrainer.Train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        input_sentence_size=10000000,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
    )
    print(f"[spm] Wrote {args.model_prefix}.model and .vocab")


if __name__ == "__main__":
    main()
