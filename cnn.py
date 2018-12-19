import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

batch_size = 128
learning_rate = 0.01
num_epoch = 10

# 实例化MNIST数据集对象
train_data = datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)

# train_loader：以batch_size大小的样本组为单位的可迭代对象
train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data)

class CNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 6, 3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm2 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        x = self.batch_norm1(self.conv1(x))
        x = F.relu(x)
        x = self.pool(x)
        x = self.batch_norm2(self.conv2(x))
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def print_model_name(self):
        print("Model Name: CNN")


class Cnn(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2))

        self.fc = nn.Sequential(
            nn.Linear(400, 120), nn.Linear(120, 84), nn.Linear(84, n_class))

    def forward(self, x):
        # print(x.size()) torch.Size([1024, 1, 28, 28])
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        # print(out.size()) = torch.Size([1024, 400])
        out = self.fc(out)
        # print(out.size()) torch.Size([1024, 10])
        return out

    def print_model_name(self):
        print("Model Name: Cnn")


isGPU = torch.cuda.is_available()
print(isGPU)
model = CNN(1, 10)
if isGPU:
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(num_epoch):
    running_acc = 0.0
    running_loss = 0.0
    for i, data in enumerate(train_loader, 1): # train_loader：以batch_size大小的样本组为单位的可迭代对象
        img, label = data
        img = Variable(img)
        label = Variable(label)
        if isGPU:
            img = img.cuda()
            label = label.cuda()
        # forward
        out = model(img)
        loss = criterion(out, label)
        # print(label)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
	
        _, pred = torch.max(out, dim=1)  # 按维度dim 返回最大值
        running_loss += loss.item()*label.size(0)
        current_num = (pred == label).sum() # variable
        acc = (pred == label).float().mean()        # variable
        running_acc += current_num.item()

        if i % 100 == 0:
            print("epoch: {}/{}, loss: {:.6f}, running_acc: {:.6f}"
                  .format(epoch+1, num_epoch, loss.item(), acc.item()))
    print("epoch: {}, loss: {:.6f}, accuracy: {:.6f}".format(epoch+1, running_loss, running_acc/len(train_data)))

model.eval()
current_num = 0
for i , data in enumerate(test_loader, 1):
    img, label = data
    if isGPU:
        img = img.cuda()
        label = label.cuda()
    with torch.no_grad():
        img = Variable(img)
        label = Variable(label)
    out = model(img)
    _, pred = torch.max(out, 1)
    current_num += (pred == label).sum().item()

print("Test result: accuracy: {:.6f}".format(float(current_num/len(test_data))))

torch.save(model.state_dict(), './cnn.pth') # 保存模型


