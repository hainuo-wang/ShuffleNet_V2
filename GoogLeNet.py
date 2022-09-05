import sys
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

dataset_dir = 'RAF-DB'
# ImageFolder
classes = os.listdir(dataset_dir)

info_array = []  # N,3的array
col = ['filename', 'filepath', 'label']
for kind_name in os.listdir(dataset_dir):
    classpath = dataset_dir + '/' + kind_name
    for filename in os.listdir(classpath):
        filepath = classpath + '/' + filename
        label = classes.index(kind_name)  # str->index int
        info_array.append([filename, filepath, label])
info_array = np.array(info_array)
df = pd.DataFrame(info_array, columns=col)
df.to_csv('RAF-DB.csv', encoding='utf-8')


class KMUFEDDataset(Dataset):
    def __init__(self, dataset_dir, csv_path, resize_shape):
        # init方法一般要编写数据的transformer、数据的基本参数。
        self.dataset_dir = dataset_dir
        self.csv_path = csv_path
        self.shape = resize_shape
        self.df = pd.read_csv(self.csv_path, encoding='utf-8')
        self.transformer = transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor(),
        ])

    def __len__(self):  # 返回数据规模
        return len(self.df)

    def __getitem__(self, idx):  # getitem,idx=index
        x_train = self.transformer(Image.open(self.df['filepath'][idx]))
        y_train = self.df['label'][idx]

        return x_train, y_train


data_ds = KMUFEDDataset('RAF-DB', 'RAF-DB.csv', (224, 224))

num_sample = len(data_ds)
train_percent = 0.8
train_num = int(train_percent * num_sample)
test_num = num_sample - train_num
train_ds, test_ds = random_split(data_ds, [train_num, test_num])

train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_ds, batch_size=16, shuffle=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 第一个通道，输入通道为in_channels,输出通道为16，卷积盒的大小为1*1的卷积层
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        # 第二个通道
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        # 第三个通道
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        # 第四个通道
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # 拼接
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=64)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(247192, 7)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# 交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)  # momentum动量


def train(epoch, train_dataloader):
    running_loss = 0.0
    times = 0
    # 返回了数据下标和数据
    train_dataloader = tqdm(train_dataloader, desc="train", file=sys.stdout)
    for batch_idx, data in enumerate(train_dataloader, 0):
        # 送入两个张量，一个张量是64个图像的特征，一个张量图片对应的数字
        inputs, target = data
        # 把输入输出迁入GPU
        inputs, target = inputs.to(device), target.to(device)
        # 梯度归零
        optimizer.zero_grad()

        # forward+backward+update
        outputs = model(inputs)
        # 计算损失，用的交叉熵损失函数
        loss = criterion(outputs, target)
        # 反馈
        loss.backward()
        # 随机梯度下降更新
        optimizer.step()
        running_loss += loss.item()
        times += 1
    print('epoch:%2d   loss:%.3f' % (epoch + 1, running_loss / times))


def test(test_dataloader):
    correct = 0
    total = 0
    # 不会计算梯度
    with torch.no_grad():
        test_dataloader = tqdm(test_dataloader, desc="test ", file=sys.stdout, colour="Green")
        for data in test_dataloader:  # 拿数据
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 预测
            # outputs.data是一个矩阵，每一行10个量，最大值的下标就是预测值
            _, predicted = torch.max(outputs.data, dim=1)  # 沿着第一维度，找最大值的下标，返回最大值和下标
            total += labels.size(0)  # labels.size(0)=64 每个都是64个元素，就可以计算总的元素
            # (predicted == labels).sum()这个是张量，而加了item()变为一个数字，即相等的数量
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))  # 正确的数量除以总数
    return 100 * correct / total


if __name__ == '__main__':
    total_accuracy = []
    for epoch in range(50):
        train(epoch, train_dataloader)
        single_accuracy = test(test_dataloader)
        total_accuracy.append(single_accuracy)
    figure = plt.figure(figsize=(8, 6))
    plt.title("GoogLeNet")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(visible=True)
    plt.plot(range(50), total_accuracy)
    plt.show()
