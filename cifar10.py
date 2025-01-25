#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR-10のデータ変換(データ拡張を含む)
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# 訓練データとテストデータのダウンロード
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# データローダの作成
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pool
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 + ReLU + Pool
        x = self.dropout(x)
        x = torch.flatten(x, 1)  # フラット化
        x = F.relu(self.fc1(x))  # FC1 + ReLU
        x = self.dropout(x)
        x = self.fc2(x)  # FC2
        output = F.log_softmax(x, dim=1)  # ログソフトマックス
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()  # モデルを訓練モードに設定
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # データをデバイスに転送
        optimizer.zero_grad()  # 勾配の初期化
        output = model(data)  # 順伝播
        loss = F.nll_loss(output, target)  # 損失計算
        loss.backward()  # 逆伝播
        optimizer.step()  # パラメータ更新
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch {epoch} [{batch_idx * len(data)} / {len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


def test(model, device, test_loader):
    model.eval()  # モデルを評価モードに設定 モデルのインスタンス化
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 勾配計算を無効化
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # 順伝播
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # 損失計算
            pred = output.argmax(dim=1, keepdim=True)  # 予測ラベル
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({accuracy:.2f}%)\n"
    )


# モデルのインスタンス化
model = CIFAR10CNN()
print(model)

# 訓練の実行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 21):  # 20エポック訓練
    # トレーニング実施
    train(model, device, train_loader, optimizer, epoch)
    # 各エポック毎にテスト
    test(model, device, test_loader)

torch.save(model.state_dict(), "cifar10_cnn.pth")
print("モデルを保存しました。")

# モデルのロード
model = CIFAR10CNN()
model.load_state_dict(torch.load("cifar10_cnn.pth"))
model.to(device)
model.eval()
print("モデルをロードしました。")
