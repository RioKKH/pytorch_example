#!/usr/bin/env python

import torch

# テンソルの作成
a = torch.tensor([1, 2, 3])
b = torch.randn(3, 3)

# テンソルをGPUに移動させる
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = a.to(device)
b = b.to(device)

print(a)
print(b)
