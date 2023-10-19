import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
    def forward(self, x, target):

        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)

class cutmix_cross_entropy(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(cutmix_cross_entropy, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return cutmix_ce(x_m, target)


def cutmix_ce(x, target):
    logit=F.softmax(x,dim=1)
    all_label_logit=torch.log(logit) * target
    return -all_label_logit.sum() / (target.shape[0])
