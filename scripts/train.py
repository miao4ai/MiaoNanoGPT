import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
import yaml
from miaosnano.model import GPT
from miaosnano.data_utils import load_dataset
from llmops.config_manager import load_config

def train(config_path: str):
    # 1. 读取配置
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 初始化模型
    model = GPT(cfg['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'])

    # 3. 数据加载
    train_dataset = load_dataset(cfg['data']['train_path'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True)

    # 4. 学习率调度器
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg['train']['warmup_steps'],
        num_training_steps=len(train_loader) * cfg['train']['epochs']
    )

    loss_fn = nn.CrossEntropyLoss()
    model.train()

    # 5. 开始训练
    for epoch in range(cfg['train']['epochs']):
        total_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch['input_ids'].to(device), batch['labels'].to(device)
            logits = model(inputs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to training config file")
    args = parser.parse_args()

    train(args.config)