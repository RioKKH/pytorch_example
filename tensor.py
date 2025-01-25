#!/usr/bin/env python

import torch

# テンソルの作成
a = torch.tensor([1, 2, 3])
b = torch.randn(3, 3)  # ランダムな値のテンソル

# 基本的な操作
c = a + 2
d = a + a
e = torch.matmul(b, b)

print(c)
print(d)
print(e)
