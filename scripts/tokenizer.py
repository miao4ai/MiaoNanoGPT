import sentencepiece as spm
import yaml

def train_tokenizer(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    spm.SentencePieceTrainer.train(
        input=cfg["input"],
        model_prefix=cfg["model_prefix"],
        vocab_size=cfg["vocab_size"],
        model_type=cfg.get("model_type", "bpe"),
        character_coverage=cfg.get("character_coverage", 1.0),
        pad_id=cfg.get("pad_id", 0),
        unk_id=cfg.get("unk_id", 1),
        bos_id=cfg.get("bos_id", 2),
        eos_id=cfg.get("eos_id", 3)
    )
    print("✅ Tokenizer training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tokenizer.yaml")
    args = parser.parse_args()
    train_tokenizer(args.config)
