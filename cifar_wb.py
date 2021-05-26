import numpy as np
import torch
import sys
import torchvision
import os
import os.path as osp
from analysis_lib import analysisNet, analysisResnet, timer
from torch.utils.data import dataloader

from dataset import cifar

# sys.stdout = open("cifar_wb.txt", "w")


class WapperNet(torch.nn.Module):
    """  
    运用并行的装饰器
    """

    def __init__(self, net):
        super(WapperNet, self).__init__()
        self.net = net

    def forward(self, x):
        return self.net.forward_graph(x)


"""
    cifar-10 每一个数据的包括(train, test)的weight_graph与bias_graph
"""

GPU_ID = 0
TEST_EPOCH = 40
BATCH_SIZE = 4
ROOT_DIR = osp.abspath(osp.dirname(__file__))
SAVE_DIR = osp.join(ROOT_DIR, 'CIFAR10_SAVE')
DATA_ROOT_DIR = osp.join(ROOT_DIR, 'data', 'cifar')
WB_SAVE_DIR = osp.join(ROOT_DIR, 'save_wb', 'resnet18-cifar10')
if not osp.exists(WB_SAVE_DIR):
    os.makedirs(WB_SAVE_DIR)

device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')


def transform_test(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    img = transform(img)
    return img


def get_wb(net, dataloader, name):
    save_dir = os.path.join(WB_SAVE_DIR, name)
    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    net.eval()
    data_len = len(dataloader.dataset)
    with torch.no_grad():
        for i, (x, _, index) in enumerate(dataloader, 1):
            con = True
            for j in range(index.size(0)):
                save_path = os.path.join(save_dir, f'index_wb_{index[j]}.pth')
                if not os.path.exists(save_path):
                    con = False
            if con:
                sys.stdout.write(f"\r | Step : {i*BATCH_SIZE-1} \ {data_len}\t")
                sys.stdout.flush()
                continue

            x = x.to(device)
            timer.timer.tic()
            if isinstance(net, torch.nn.DataParallel):
                output, weight_graph, bias_graph = net(x)
            else:
                output, weight_graph, bias_graph = net.forward_graph(x)
            timer.timer.toc()

            for j in range(index.size(0)):
                save_dict = {
                    'weight_graph': weight_graph[j],
                    'bias_graph': bias_graph[j]
                }
                save_path = os.path.join(save_dir, f'index_wb_{index[j]}.pth')
                torch.save(save_dict, save_path)
                sys.stdout.write(f"\r | Step : {(i-1)*BATCH_SIZE+j} \ {data_len}, time: {timer.timer.average_time():.4f}s\t")
                sys.stdout.flush()


if __name__ == "__main__":
    SAVE_DIR_NET = osp.join(SAVE_DIR, 'resnet18-cifar10')
    net_load_path = osp.join(SAVE_DIR_NET, f"net_{TEST_EPOCH}.pth")

    cifar_10_train = cifar.CIFAR10(DATA_ROOT_DIR, train=True, transform=transform_test, download=False)
    cifar_10_test = cifar.CIFAR10(DATA_ROOT_DIR, train=False, transform=transform_test, download=False)

    train_dataloader = dataloader.DataLoader(cifar_10_train, batch_size=BATCH_SIZE)
    test_dataloader = dataloader.DataLoader(cifar_10_test, batch_size=BATCH_SIZE)

    net = analysisResnet.resnet18(num_classes=10, input_size=torch.Size((3, 32, 32)))
    net.load_state_dict(torch.load(net_load_path))
    # ========================================
    # 多GPU
    net = WapperNet(net)
    net = torch.nn.DataParallel(net)
    # ========================================
    net.to(device)
    net.eval()

    with torch.no_grad():
        print("Get Test wb_graph:")
        get_wb(net, test_dataloader, 'test')
        print("Get Train wb_graph:")
        get_wb(net, train_dataloader, 'train')
