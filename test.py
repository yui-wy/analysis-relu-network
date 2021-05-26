import numpy as np
import torch
import sys
import torchvision
import os
import os.path as osp
from analysis_lib import analysisNet, analysisResnet
from torch.utils.data import dataloader

from dataset import cifar


GPU_ID = 0
TEST_EPOCH = 40
NET_TAG = 'resnet18-cifar10'
ROOT_DIR = osp.abspath(osp.dirname(__file__))
DATA_ROOT_DIR = osp.join(ROOT_DIR, 'data', 'cifar')
SAVE_DIR = osp.join(ROOT_DIR, 'CIFAR10_SAVE')
DIS_SAVE_DIR = osp.join(ROOT_DIR, 'save_dt', 'cifar_10')
WB_SAVE_DIR = osp.join(ROOT_DIR, 'save_wb', NET_TAG)

if not osp.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
# if not osp.exists(TEST_SAVE_DIR):
#     os.mkdir(TEST_SAVE_DIR)

device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')


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


def experiment_1(net, test_dataloader, train_dataloader):
    """  
    统计:
        1. L2距离的smallest Top 100:
            a. top100的距离;
            b. top100的标签;
        2. 相同label的smallest Top 100:
            a. top100的距离;
            b. top100的index;
        3. 统计相同label的L2 smallest是不是全局最近点;
            a. 是  : 统计个数
            b. 不是: 统计
    """
    def read_graph(graph_path):
        graph_dict = torch.load(graph_path)
        return graph_dict['weight_graph'], graph_dict['bias_graph']

    net.eval()
    train_labels = train_dataloader.dataset.targets
    train_labels = torch.Tensor(train_labels).to(device)
    index_total = torch.arange(0, len(train_dataloader.dataset))
    test_graph_dir = osp.join(WB_SAVE_DIR, 'test')
    train_graph_dir = osp.join(WB_SAVE_DIR, 'train')
    with torch.no_grad():
        right_num = 0
        wrong_num = 0
        # ===================
        right_top1_e_num = 0
        right_top5_e_num = 0
        right_top10_e_num = 0
        right_top50_e_num = 0
        right_top100_e_num = 0
        # ===================
        wrong_top1_e_num = 0
        wrong_top5_e_num = 0
        wrong_top10_e_num = 0
        wrong_top50_e_num = 0
        wrong_top100_e_num = 0
        # ===================
        for i, (x, y, index) in enumerate(test_dataloader, 1):
            distance_path = osp.join(DIS_SAVE_DIR, f'test_{index[0]}.pkl')
            distance = torch.load(distance_path)
            x, y = x.to(device), y.long().to(device)
            x = net(x)
            pre_label = torch.argmax(x, dim=1)
            isRight = (pre_label == y).cpu()
            # -----------------------------
            # 数据获取部分
            bool_labels = (train_labels == pre_label)
            same_label_index = index_total[bool_labels]
            same_dis = distance[bool_labels]
            # 距离最近的100个点 与 同标签距离最近的100个点
            # L2距离的 Top 100
            all_near_100, all_near_100_index = torch.topk(distance, k=100, largest=False)
            all_near_labels = train_labels[all_near_100_index]
            # 相同Label的 L2距离 Top 100
            same_near_100, same_near_100_index = torch.topk(same_dis, k=100, largest=False)
            same_near_100_real_index = same_label_index[same_near_100_index]
            # ===========================================
            # 统计 weight_graph 和 bias_graph
            # test - graph
            test_graph_path = osp.join(test_graph_dir, f'index_wb_{int(index[0])}.pth')
            test_weight_graph, test_bias_graph = read_graph(test_graph_path)

            # all near - 100
            
            for j in range(100):
                train_graph_path = osp.join(train_graph_dir, f'index_wb_{int(all_near_100_index[j])}.pth')
                train_weight_graph, train_bias_graph = read_graph(train_graph_path)
                print('\n---------------------------------------------')
                print(test_weight_graph - train_weight_graph)
                print(((test_bias_graph- train_bias_graph)**2).sum())
                print(f'Num : {j} ,Dis : {distance[all_near_100_index[j]]}')
                print('----------------------------------------------')
            # ===========================================
            # 统计
            # 验证观点: 拟合数据是拥有近似数据的
            if isRight:
                right_num += 1
                # Top1 对的总数
                if all_near_100[0] == same_near_100[0]:
                    right_top1_e_num += 1
                # Top5 对的总数
                if same_near_100[0] in all_near_100[:5]:
                    right_top5_e_num += 1
                # Top10 对的总数
                if same_near_100[0] in all_near_100[:10]:
                    right_top10_e_num += 1
                if same_near_100[0] in all_near_100[:50]:
                    right_top50_e_num += 1
                if same_near_100[0] in all_near_100[:100]:
                    right_top100_e_num += 1
            else:
                wrong_num += 1
                if all_near_100[0] == same_near_100[0]:
                    wrong_top1_e_num += 1
                if same_near_100[0] in all_near_100[:5]:
                    wrong_top5_e_num += 1
                if same_near_100[0] in all_near_100[:10]:
                    wrong_top10_e_num += 1
                if same_near_100[0] in all_near_100[:50]:
                    wrong_top50_e_num += 1
                if same_near_100[0] in all_near_100[:100]:
                    wrong_top100_e_num += 1
            # ===========================================
            sys.stdout.write(f"\rDataset_Num : {i}\t")
            sys.stdout.flush()
        print(f"\nRight_Num : {right_num}, Wrong_Num : {wrong_num}")
        print(f"Right_top1_equal_num : {right_top1_e_num}, Wrong_top1_equal_num : {wrong_top1_e_num}")
        print(f"Right_top5_equal_num : {right_top5_e_num}, Wrong_top5_equal_num : {wrong_top5_e_num}")
        print(f"Right_top10_equal_num : {right_top10_e_num}, Wrong_top10_equal_num : {wrong_top10_e_num}")
        print(f"Right_top50_equal_num : {right_top50_e_num}, Wrong_top50_equal_num : {wrong_top50_e_num}")
        print(f"Right_top100_equal_num : {right_top100_e_num}, Wrong_top100_equal_num : {wrong_top100_e_num}")
        # 300个例外完全可以忽略


if __name__ == "__main__":
    SAVE_DIR_NET = osp.join(SAVE_DIR, NET_TAG)
    net_load_path = osp.join(SAVE_DIR_NET, f"net_{TEST_EPOCH}.pth")

    cifar_10_train = cifar.CIFAR10(DATA_ROOT_DIR, train=True, transform=transform_test, download=False)
    cifar_10_test = cifar.CIFAR10(DATA_ROOT_DIR, train=False, transform=transform_test, download=False)

    train_dataloader = dataloader.DataLoader(cifar_10_train, batch_size=128)
    test_dataloader = dataloader.DataLoader(cifar_10_test, batch_size=1)

    net = analysisResnet.resnet18(num_classes=10, input_size=(3, 32, 32))
    net.load_state_dict(torch.load(net_load_path))
    net.to(device)
    net.eval()

    with torch.no_grad():
        # Experiment
        # =========================================
        # Test Acc
        # val_acc = val_net(net, test_dataloader)
        # print(f"Epoch: {TEST_EPOCH}, Val_Acc: {val_acc:.4f}")
        # =========================================
        # Experiment 1
        experiment_1(net, test_dataloader, train_dataloader)
        # =========================================
        # Experiment 2
        # 寻找相似的权重与bias
