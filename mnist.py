#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# データ変換(テンソル化と正規化)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# 訓練データとテストデータのダウンロード
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# データローダ作成
# train=True, train=False で異なるデータセットが提供される
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 畳み込み層1: 入力チャネル1 (グレースケール)、出力チャネル32、カーネルサイズ3、ストライド1
        # 入力画像1チャネルx28x28 --> 32チャネルx26x26
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 畳み込み層2: 入力チャネル32, 出力チャネル64, カーネルサイズ3、ストライド1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # ドロップアウト層 (過学習防止)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # 全結合層1
        self.fc1 = nn.Linear(9216, 128)  # 64チャネル x 12 x 12 = 9216
        # 全結合層2
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)  # 畳み込み層1
        x = F.relu(x)  # ReLU活性化関数
        x = self.conv2(x)  # 畳み込み層2
        x = F.relu(x)  # ReLU活性化関数
        x = F.max_pool2d(x, 2)  # 最大プーリング(カーネルサイズ2)
        x = self.dropout1(x)  # ドロップアウト
        x = torch.flatten(x, 1)  # フラット化
        x = self.fc1(x)  # ReLU活性化関数
        x = F.relu(x)  # ReLU活性化関数
        x = self.dropout2(x)  # ドロップアウト
        x = self.fc2(x)  # 全結合増2(出力)
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


model = CNN()
print(model)

# 訓練の実行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 11):  # 10エポック訓練
    # トレーニング
    train(model, device, train_loader, optimizer, epoch)
    # 各エポック事にテスト
    test(model, device, test_loader)

# 新しいオプティマイザ(学習率0.0001)で再設定
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(11, 21):  # エポック11から20まで
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# モデルの保存
torch.save(model.state_dict(), "mnist_cnn.pth")

# # モデルのロード
# model = CNN()
# model.load_state_dict(torch.load("mnist_cnn.pth"))
# model.to(device)
# model.eval()
# print("モデルをロードしました。")
