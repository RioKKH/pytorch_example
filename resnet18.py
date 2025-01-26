#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# データ変換(データ拡張と正規化)
transform_train = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# データセットのダウンロードとデータローダーの作成
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=2)

# ResNet18を事前訓練済みモデルとしてロード
model = models.resnet18(pretrained=True)

# CIFAR-10は32x32ピクセルなので、ResNet18の最初の畳み込みそうを変更する
# 元のResNet18はconv1はkernel_size=7, stride=2, padding=3 なので、これを以下に変更する
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

# 最初の最大プーリング層を削除(オリジナルでは7x7カーネル層の後にMaxPoolがあるため)
model.max_pool = nn.Identity()

# 最終全結合層をCIFAR-10を合わせて変更する
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

model = model.to(device)
print(model)

# オプティマイザの競ってい(Weight decayを含む)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 学習率スケジューラの設定(エポック毎に学習率を減少)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# トレーニング関数の定義
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)  # 順伝播
        loss = F.cross_entropy(outputs, target)  # 損失計算
        loss.backward()  # 逆伝播
        optimizer.step()  # パラメータ更新

        running_loss += loss.item() * data.size(0)
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(
                f"Epoch [{epoch}], Step [{batch_idx + 1} / {len(train_loader)}],"
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    print(f"Train Epoch: {epoch}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


# テスト関数の定義
def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = F.cross_entropy(outputs, target, reduction="sum")  # パッチ全体の損失
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= total
    test_acc = 100.0 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    return test_acc


# モデルの保存関数
def save_model(model, path="resnet18_cifar10.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# モデルのロード関数
def load_model(model, path="resnet18_cifar10.pth"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()
        print(f"Model loaded from {path}")
    else:
        print(f"No model found at {path}")


# メイン関数の定義
def main():
    num_epochs = 20
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader)
        scheduler.step()  # 学習率の更新

        # ベストモデルの保存
        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, "best_resnet18_cifar10.pth")

    print(f"Best Test Accuracy: {best_acc:.2f}%")

    # ベストモデルのロードと最終評価
    load_model(model, "best_resnet18_cifar10.pth")
    test(model, device, test_loader)


if __name__ == "__main__":
    main()
