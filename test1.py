import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# 定义与训练时相同的模型结构
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

# 加载图片并预测
def predict_image(image_path, model_path):

    # 创建模型实例并加载权重
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))  # 加载权重
    model.eval()  # 切换到评估模式

    # 预处理图片
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img).unsqueeze(0)

    # 推理
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    # 显示图片和预测结果
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {predicted.item()}")
    plt.axis('off')
    plt.show()

    return predicted.item()

# 示例使用
if __name__ == "__main__":
    model_path = 'cnn_model.pth'  # 模型权重文件
    image_path = 'my_digit.jpg'  # 测试图片路径

    predicted_number = predict_image(image_path, model_path)
    print(f"The model predicts the number is: {predicted_number}")
