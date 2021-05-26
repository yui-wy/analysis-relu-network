import numpy as np
import torch
import torch.nn as nn
import torchvision
import os
import os.path as osp
from analysis_lib import analysisNet, analysisResnet
from torch.utils.data import dataloader

from dataset import cifar


GPU_ID = 1
MAX_EPOCH = 20000
ROOT_DIR = osp.abspath(osp.dirname(__file__))
DATA_ROOT_DIR = osp.join(ROOT_DIR, 'data', 'cifar')
SAVE_DIR = osp.join(ROOT_DIR, 'CIFAR10_SAVE')
if not osp.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')


class Cifar10Net(analysisNet.AnalysisNet):
    def __init__(self, device=device, input_size=(3, 32, 32)):
        super(Cifar10Net, self).__init__(device, input_size)
        self.transition = nn.Parameter(torch.rand(3, 32, 32), requires_grad=True)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = x + self.transition
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x


class RandomData(torch.utils.data.Dataset):
    def __init__(self, transform, length=50000):
        super(RandomData, self).__init__()
        self.transform = transform
        self.length = length

    def __getitem__(self, index):
        x = np.random.randint(low=0, high=256, size=(32, 32, 3), dtype=np.uint8)
        x = self.transform(x)
        return x

    def __len__(self):
        return self.length


def transform_test(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    img = transform(img)
    return img


def accuracy(x, classes):
    arg_max = torch.argmax(x, dim=1).long()
    eq = torch.eq(classes, arg_max)
    return torch.sum(eq).float()


def val_net(net, val_dataloader):
    net.eval()
    with torch.no_grad():
        val_accuracy_sum = 0
        for _, (x, y, _) in enumerate(val_dataloader, 1):
            x, y = x.to(device), y.long().to(device)
            x = net(x)
            val_acc = accuracy(x, y)
            val_accuracy_sum += val_acc
        val_accuracy_sum /= len(val_dataloader.dataset)
    return val_accuracy_sum


if __name__ == "__main__":
    SAVE_DIR_NET = osp.join(SAVE_DIR, 'cifar10Net-resnet18')
    if not osp.exists(SAVE_DIR_NET):
        os.mkdir(SAVE_DIR_NET)

    cifar_10_train = cifar.CIFAR10(DATA_ROOT_DIR, train=True, transform=transform_test, download=False)
    cifar_10_test = cifar.CIFAR10(DATA_ROOT_DIR, train=False, transform=transform_test, download=False)
    randomData_train = RandomData(transform=transform_test)

    train_dataloader = dataloader.DataLoader(cifar_10_train, batch_size=128, shuffle=True)
    random_dataloader = dataloader.DataLoader(randomData_train, batch_size=2048)
    test_dataloader = dataloader.DataLoader(cifar_10_test, batch_size=128)

    net = Cifar10Net()
    net.load_state_dict(torch.load(osp.join(SAVE_DIR_NET, 'net_1900.pth')))

    # ==========================
    # target net --> resnet50
    teacher_path = osp.join(SAVE_DIR, 'resnet18-cifar10', 'net_40.pth')
    net_teacher = analysisResnet.resnet18(num_classes=10, device=device, input_size=(3, 32, 32))
    net_teacher.load_state_dict(torch.load(teacher_path))
    net_teacher.to(device)
    net_teacher.eval()
    val_acc = val_net(net_teacher, test_dataloader)
    print(f"TeachNet: Val_Acc: {val_acc:.4f}")
    # ==========================

    net.to(device)
    ce = torch.nn.CrossEntropyLoss()
    l2Loss = torch.nn.MSELoss()

    lr = 0.002
    optim = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.90, weight_decay=0.0005)
    val_log_net = open(os.path.join(SAVE_DIR_NET, "net_acc_log.txt"), 'w')
    for i in range(1900, MAX_EPOCH):
        if (i != 0) and (i % 2000 == 0):
            lr /= 2
            for param_group in optim.param_groups:
                param_group['lr'] = lr
        net.train()
        for j, x in enumerate(random_dataloader, start=1):
            x = x.float().to(device)
            with torch.no_grad():
                y = net_teacher(x).detach()
            x = net(x)
            loss = l2Loss(x, y) * 5
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(f"Epoch: {i+1} / {MAX_EPOCH}, Loss: {loss:.4f}")

        val_acc = val_net(net, test_dataloader)
        print(f"Epoch: {i+1} / {MAX_EPOCH}, Val_Acc: {val_acc:.4f}")
        val_log_net.write(f"Epoch: {i+1} / {MAX_EPOCH}, Val_Acc: {val_acc:.4f}\n")
        val_log_net.flush()

        if 0 == ((i+1) % 100):
            torch.save(net.state_dict(), osp.join(SAVE_DIR_NET, f'net_{i+1}.pth'))
