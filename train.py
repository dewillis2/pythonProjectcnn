import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层，输出特征图大小为 (28-3+1, 28-3+1) = (26, 26)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        # 第一个池化层，输出特征图大小为 (26/2, 26/2) = (13, 13)
        self.pool = nn.MaxPool2d(2, 2)
        # 第二个卷积层，输出特征图大小为 (13-3+1, 13-3+1) = (11, 11)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        # 第二个池化层，输出特征图大小为 (11/2, 11/2) = (5, 5)（向下取整）
        # 因此，我们需要调整第二个池化层，使其输出特征图大小为 (6, 6)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # 全连接层的输入特征数量为 64 * 6 * 6
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (32, 26, 26) -> (32, 13, 13)
        x = self.pool2(F.relu(self.conv2(x)))  # (64, 11, 11) -> (64, 6, 6)
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化模型、损失函数和优化器
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 5
print("Training the model...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        outputs = model(images)

        labels = labels.squeeze().long()

        # 计算损失
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# 测试模型
print("Testing the model...")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # 确保标签形状正确
        labels = labels.squeeze()

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total:.4f}")

# 保存模型
torch.save(model.state_dict(), 'cnn_model.pth')
print("Model saved as cnn_model.pth")
