#!/usr/bin/env python

import torch

# 勾配追跡を有効にする
x = torch.ones(2, 2, requires_grad=True)
print(f"x:\n {x}")
y = x + 2
print(f"y = x + 2:\n {y}")
z = y * y * 3
print(f"z = y * y * 3:\n {z}")
out = z.mean()
print(f"out = z.mean():\n {out}")

# 勾配の計算
out.backward()
# print(f"out.backward: {out.backward()}")
print(f"x.grad:\n {x.grad}")
