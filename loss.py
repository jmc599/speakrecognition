import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AAMSoftmax(nn.Module):
    def __init__(self, in_features, n_class, s=20.0, m=0.20):
        super().__init__()
        self.in_features = in_features
        self.n_class = n_class
        self.s = s
        self.m = m
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.FloatTensor(n_class, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # x: 输入的 Embedding
        # label: 标签

        # 1. 归一化权重和特征
        cosine = F.linear(
            F.normalize(x, dim=1, eps=self.eps),
            F.normalize(self.weight, dim=1, eps=self.eps),
        )
        cosine = cosine.clamp(min=-1.0 + self.eps, max=1.0 - self.eps)

        # 2. 加上 Margin
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(self.eps, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 3. 生成 One-hot
        one_hot = torch.zeros_like(cosine, device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        loss = F.cross_entropy(output, label)
        return loss
