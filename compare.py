#!/usr/bin/env python
"""
compare.py
-------------------
用于用 wikitext2 或 wikitext103 数据集对 GPT 模型进行从零开始预训练，
并对比 sink softmax（--use_sink）与标准 softmax在预训练阶段 loss 下降速度（用 epoch 表示）的差别。
如果使用 --compare 参数，则同时跑两种模式，并绘制在同一张图上。
"""

import argparse
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import GPT, GPTConfig


def load_dataset(file_path):
    """加载文本数据并构建词表，采用简单空格切分"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    words = text.split()
    vocab = sorted(set(words))
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}
    tokens = [stoi[w] for w in words]
    data = torch.tensor(tokens, dtype=torch.long)
    return data, len(vocab), stoi, itos


def get_batch(data, block_size, batch_size):
    """
    从 data 中随机采样一个 batch
    注意：为了简单观察训练过程，每个 epoch 随机采样一批
    """
    n = len(data) - block_size
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y


def pretrain(dataset_file, use_sink, epochs, batch_size, block_size, lr):
    # 加载数据集
    data, vocab_size, _, _ = load_dataset(dataset_file)
    # 划分训练/验证（这里主要用于训练 loss 的记录），简单划分 90% 训练
    n = int(0.9 * len(data))
    train_data = data[:n]
    
    print(f"从 {dataset_file} 加载数据，token 总数：{len(data)}，词表大小：{vocab_size}。")
    
    # 构造 GPT 模型配置（小规模模型便于快速实验）
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=2,   # 层数较少
        n_head=2,    # 注意力头数
        n_embd=64,   # embedding 维度
        dropout=0.0,
        bias=True
    )
    # 配置是否采用 sink softmax
    config.use_sink_softmax = use_sink

    model = GPT(config)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    losses = []
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        x, y = get_batch(train_data, block_size, batch_size)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 1 == 0:
            elapsed = time.time() - start_time
            mode = "Sink" if use_sink else "Baseline"
            print(f"[{mode}] Epoch {epoch:3d}: Loss = {loss.item():.4f}, Elapsed = {elapsed:.2f}s")
    
    return losses, model


def plot_loss(loss_curves, labels, title, filename):
    plt.figure()
    for losses, label in zip(loss_curves, labels):
        plt.plot(losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/wikitext2.txt",
                        help="数据集文件路径，可选 data/wikitext2.txt 或 data/wikitext103.txt")
    parser.add_argument("--use_sink", action="store_true",
                        help="使用 sink softmax（默认不使用，即标准 softmax）")
    parser.add_argument("--compare", action="store_true",
                        help="同时运行 sink 与 baseline 模型进行对比")
    parser.add_argument("--epochs", type=int, default=10,
                        help="预训练的 epoch 数（这里每个 epoch 采样一个 batch）")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="训练批次大小")
    parser.add_argument("--block_size", type=int, default=128,
                        help="训练序列的长度")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="学习率")
    args = parser.parse_args()
    
    if args.compare:
        print("Running comparison experiment: both Sink softmax and Baseline softmax")
        losses_sink, _ = pretrain(args.dataset, use_sink=True, epochs=args.epochs,
                                    batch_size=args.batch_size, block_size=args.block_size, lr=args.lr)
        losses_baseline, _ = pretrain(args.dataset, use_sink=False, epochs=args.epochs,
                                      batch_size=args.batch_size, block_size=args.block_size, lr=args.lr)
        plot_loss([losses_sink, losses_baseline],
                  ["Sink Softmax", "Baseline Softmax"],
                  f"Pretrain Loss Comparison on {args.dataset}",
                  "plots/pretrain_loss_comparison.png")
    else:
        mode = "Sink Softmax" if args.use_sink else "Baseline Softmax"
        print(f"Running experiment with {mode}")
        losses, _ = pretrain(args.dataset, use_sink=args.use_sink, epochs=args.epochs,
                             batch_size=args.batch_size, block_size=args.block_size, lr=args.lr)
        plot_loss([losses],
                  [mode],
                  f"Pretrain Loss Curve on {args.dataset}",
                  "plots/pretrain_loss_curve.png")


if __name__ == "__main__":
    main()
