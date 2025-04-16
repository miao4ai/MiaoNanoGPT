# Miao's Nano GPT : Lightweight LLM with LLMOps Magic ✨

> A beautiful journey from nanoGPT to LLM mastery.

---

## 🧠 What is Miao's Nano?

**MiaoNanoGPT** is a lightweight, extensible GPT implementation based on [nanoGPT](https://github.com/karpathy/nanoGPT), powered by modern **LLMOps** principles:

- 💡 Modular design for research
- ⚙️ Reproducible training with MLflow
- 🧪 Config-driven experimentation
- 🚀 Ready for production with FastAPI serving

Perfect for prototyping LLMs (Language Models) in resource-constrained environments, and fine-tuning them on custom datasets.

---

## 🔧 Key Features

- ✅ Rewritten nanoGPT for clarity & modularity
- 🈶 SentencePiece tokenizer (for Chinese & multilingual support)
- ⚙️ YAML-based config system
- 📊 MLflow integration for full lifecycle tracking
- 🧠 QLoRA-ready for efficient fine-tuning
- 📦 ONNX export support for optimized deployment
- 🌐 FastAPI inference API
- 🔬 Jupyter notebooks for LLM exploration

---

## 🗂 Project Structure

```
MiaosNano/
├── configs/                # YAML config files
├── data/                   # Raw and processed training data
├── miaosnano/              # Core GPT model & training logic
├── scripts/                # CLI interfaces (train/eval/sample)
├── llmops/                 # LLMOps utilities (MLflow, config mgmt)
├── notebooks/              # Research notebooks
├── api/                    # Inference API server (FastAPI)
├── tests/                  # Unit tests
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data & tokenizer
```bash
python scripts/tokenize_data.py --config configs/tokenizer.yaml
```

### 3. Train your first model
```bash
python scripts/train.py --config configs/train.yaml
```

### 4. Launch inference API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

