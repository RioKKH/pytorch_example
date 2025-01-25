#!/usr/bin/env python

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR10の訓練データをロード(データ拡張無し)
# transform = transforms.ToTensor()
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=50000, shuffle=False, num_workers=2)

# データ全体を1つのパッチとしてロードする
data = next(iter(train_loader))
images, _ = data  # imagesの形状: [50000, 3, 32, 32]

# 各チャンネルの平均と標準偏差を計算
# [バッチサイズ, チャネル数, 高さ, 幅]
mean = images.mean([0, 2, 3])  # 各チャネル次元毎の平均値。(R, G, B)の平均値
std = images.std([0, 2, 3])

print(f"Mean: {mean}")
print(f"Std: {std}")
