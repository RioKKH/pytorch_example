#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SimpleNet(nn.Module):
    def __init__(self):
        # モデルはnn.Moduleを継承して定義します。
        super(SimpleNet, self).__init__()
        # nn.Linearは全結合層を定義します
        self.fc1 = nn.Linear(784, 128)  # 入力層から隠れ層
        self.fc2 = nn.Linear(128, 10)  # 隠れ層から出力層

    def forward(self, x):
        """
        データの流れを定義します
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = SimpleNet()
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# ダミーデータ
inputs = torch.randn(64, 784)  # バッチサイズ64, 入力サイズ784
labels = torch.randint(0, 10, (64,))  # ラベル

# 訓練ステップ
optimizer.zero_grad()  # 勾配の初期化
outputs = net(inputs)  # 順伝播
loss = criterion(outputs, labels)  # 損失計算
loss.backward()  # 逆伝播
optimizer.step()  # パラメータ更新

print(loss.item())
