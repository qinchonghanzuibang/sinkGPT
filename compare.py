#!/usr/bin/env python
"""
compare.py
-------------------
用于用 wikitext2 或 wikitext103 数据集对 GPT 模型进行从零开始预训练，
并对比 sink softmax（--use_sink）与标准 softmax 在预训练阶段 loss 下降速度（以 epoch 表示）的差别。
如果使用 --compare 参数，则同时跑两种模式，并绘制在同一张图上。
"""

import argparse
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import GPT, GPTConfig

import tiktoken

def load_dataset_GPT2(file_path):
    """
    加载文本数据并使用 GPT-2 的 tiktoken tokenizer 进行编码，
    确保生成的 token id 落在 [0, 50257) 范围内。
    
    TODO: 发现这个tiktoken要联网
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)
    return data, 50257, None, None


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

def load_dataset_partial(file_path, max_lines=100000):
    """只加载文件的前 max_lines 行，并按空格切分生成 tokens"""
    tokens = []
    vocab_set = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if line:
                words = line.split()
                tokens.extend(words)
                vocab_set.update(words)
    vocab = sorted(vocab_set)
    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}
    token_ids = [stoi[w] for w in tokens]
    data = torch.tensor(token_ids, dtype=torch.long)
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


def pretrain(dataset_file, use_sink, epochs, batch_size, lr):
    data, computed_vocab_size, _, _ = load_dataset(dataset_file)
    # data, computed_vocab_size, _, _ = load_dataset_partial(dataset_file)
    # data, vocab_size, _, _ = load_dataset_GPT2(dataset_file)

    vocab_size = computed_vocab_size
    # 固定使用 GPT-2 的配置
    # block_size = 1024       # GPT-2 的上下文长度
    block_size = 512        # 为了加快训练速度，使用较小的上下文长度
    
    print(f"从 {dataset_file} 加载数据，token 总数：{len(data)}，GPT-2 词表大小设置为：{vocab_size}，block_size={block_size}。")
    
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
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
        x, y = get_batch(data[:int(0.9*len(data))], block_size, batch_size)
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
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
                        help="预训练的 epoch 数")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="训练批次大小")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="学习率")
    args = parser.parse_args()
    
    if args.compare:
        print("Running comparison experiment: both Sink softmax and Baseline softmax")
        losses_sink, _ = pretrain(args.dataset, use_sink=True, epochs=args.epochs,
                                    batch_size=args.batch_size, lr=args.lr)
        losses_baseline, _ = pretrain(args.dataset, use_sink=False, epochs=args.epochs,
                                      batch_size=args.batch_size, lr=args.lr)
        plot_loss([losses_sink, losses_baseline],
                  ["Sink Softmax", "Baseline Softmax"],
                  f"Pretrain Loss Comparison on {args.dataset}",
                  "plots/pretrain_loss_comparison.png")
    else:
        mode = "Sink Softmax" if args.use_sink else "Baseline Softmax"
        print(f"Running experiment with {mode}")
        losses, _ = pretrain(args.dataset, use_sink=args.use_sink, epochs=args.epochs,
                             batch_size=args.batch_size, lr=args.lr)
        plot_loss([losses],
                  [mode],
                  f"Pretrain Loss Curve on {args.dataset}",
                  "plots/pretrain_loss_curve.png")


if __name__ == "__main__":
    main()
