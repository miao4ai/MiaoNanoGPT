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

## ✍️ 手撕 Transformer 之旅

> 2026-04-06 起航 🚢

今天开始手撕 Transformer！从零一行行敲出 Attention、MultiHead、FFN、LayerNorm、位置编码……不调包，不抄作业，用代码把论文里的每一个公式都跑通一遍。哈哈，冲鸭 🥳

### 进度

| 模块 | 文件 | 状态 |
|------|------|------|
| Embedding | `miaosnano/embedding.py` | ✅ 完成 |
| Multi-Head Attention | `miaosnano/multiheadattention.py` | ✅ 完成 |
| Feed-Forward Network | `miaosnano/transformer_block.py` | ✅ 完成 |
| LayerNorm | `miaosnano/transformer_block.py` (用 `nn.LayerNorm`) | ✅ 完成 |
| Positional Encoding | `miaosnano/positional_encoding.py` | ✅ 完成 |
| Transformer Block | `miaosnano/transformer_block.py` | ✅ 完成 |
| Causal Mask | `miaosnano/multiheadattention.py` (`mask` 参数) | ✅ 完成 |
| **完整 GPT** | `miaosnano/gpt.py` | ✅ 完成 🎉 |

---

