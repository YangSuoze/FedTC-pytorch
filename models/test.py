'''
用户测试模型效果
'''
import copy
import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    '''
    params: 总数据集，对应的子索引
    return: client对应的训练或测试数据集
    '''
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    

dataloader_kwargs = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': 4,
    }

def test_img_cur_model(net_g, dataset, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False,**dataloader_kwargs)
    for idx,(data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).sum()#.long().cpu().sum()
    if len(data_loader.dataset)!=0:
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
    else:
        test_loss = -1
        accuracy = -1
    return accuracy, test_loss  
