import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import optimizer as optimizer_
from torch_geometric.utils import to_dense_batch, to_dense_adj

class Trainer(object):
    r"""模型训练器
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    # 训练过程
    def train(self, data_loader: DataLoader, optimizer: optimizer_, criterion, device: torch.device):
        self.model.train()
        self.model.to(device)
        criterion.to(device)
        losses = []
        for step, data in enumerate(data_loader):
            data = data.to(device)
            logits, l_lp, l_e = self.model(data)  # 前向传播
            loss = criterion(logits, data.y)  # 计算损失函数值
            loss = loss + l_lp + l_e
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % 5 == 0:
                print('batch:{} loss:{:.6f}'.format(step, loss.item()))
            losses.append(loss.item())
        return np.mean(losses)

    # 验证过程
    def evaluate(self, data_loader: DataLoader, criterion, device: torch.device):
        self.model.eval()
        self.model.to(device)
        criterion.to(device)
        correct = 0
        losses = []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(device)
                logits, l_lp, l_e = self.model(data)
                loss = criterion(logits, data.y)
                loss = loss + l_lp + l_e
                pred = logits.argmax(-1)
                correct += pred.eq(data.y).sum().item()
                losses.append(loss.item())
        return np.mean(losses), correct / len(data_loader.dataset)

    def get_parameters(self):
        return self.model.parameters()