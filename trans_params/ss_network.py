import torch
import joblib

import numpy as np
import pandas as pd

import torch.nn as nn

from torch.nn.functional import relu, sigmoid
from utils import Params


class LocalConv1D(nn.Module):
    def __init__(self, params: Params):
        super(LocalConv1D, self).__init__()
        self.params = params
        # self.p_ind = joblib.load('graph/point_index.jl')
        self.conv = nn.Conv1d(in_channels=self.params.num_nwp, out_channels=self.params.local_cnn_dims,
                              kernel_size=self.params.local_points)
        # self.bn = nn.BatchNorm1d(num_features=self.params.local_cnn_dims, affine=False)
        self.act = nn.ELU()
        self.output_size = self.params.local_cnn_dims
        self.wsize = self.conv.weight.numel()

    def forward(self, x: torch.Tensor):
        # x is (batch_size, num_nodes, num_features), here num_nodes do not include any dump nodes.
        y = self.conv(x)
        y = self.act(y)
        return y.reshape((x.size()[0], -1))

    def reinit(self):
        lim = 1./np.sqrt(self.conv.kernel_size[0]*self.conv.in_channels)
        nn.init.uniform_(self.conv.weight, -lim, lim)
        nn.init.uniform_(self.conv.bias, -lim, lim)


class Mlp(nn.Module):
    def __init__(self, params: Params, conv_input_size: int):
        super(Mlp, self).__init__()
        self.params = params
        q = params.q
        if isinstance(q, list):
            self.q = q
            self.q_num = len(q)
        else:
            self.q = [i / (q + 1.) for i in range(1, q + 1)]
            self.q_num = q
        self.l1 = nn.Linear(in_features=self.params.num_temp + conv_input_size, out_features=self.params.mlp_hid_dim1)
        # self.bn1 = nn.BatchNorm1d(num_features=self.params.mlp_hid_dim[0], affine=False)
        self.act1 = nn.ELU()
        self.l2 = nn.Linear(in_features=self.params.mlp_hid_dim1, out_features=self.params.mlp_hid_dim2)
        # self.bn2 = nn.BatchNorm1d(num_features=self.params.mlp_hid_dim[1], affine=False)
        self.act2 = nn.ELU()
        self.l3 = nn.Linear(in_features=self.params.mlp_hid_dim2, out_features=self.q_num)
        # self.bn3 = nn.BatchNorm1d(num_features=self.q_num, affine=False)
        # self.act3 = nn.Softmax(dim=1)
        self.act3 = nn.ReLU()
        self.wsize = self.l1.weight.numel() + self.l2.weight.numel() + self.l3.weight.numel()

    def forward(self, x, t):
        y = self.l1(torch.cat((x, t), dim=1))
        y = self.act1(y)
        y = self.l2(y)
        y = self.act2(y)
        y = self.l3(y)
        # y = self.act3(torch.cat((y, torch.zeros((x.shape[0], 1), device=x.device)), dim=1))[:, :-1]
        # y = self.act3(y)
        # return y.cumsum(dim=1)
        return y

    def reinit(self):
        lim1 = 1./np.sqrt(self.l1.in_features)
        lim2 = 1./np.sqrt(self.l2.in_features)
        lim3 = 1./np.sqrt(self.l3.in_features)
        nn.init.uniform_(self.l1.weight, -lim1, lim1)
        nn.init.uniform_(self.l1.bias, -lim1, lim1)
        nn.init.uniform_(self.l2.weight, -lim2, lim2)
        nn.init.uniform_(self.l2.bias, -lim2, lim2)
        nn.init.uniform_(self.l3.weight, -lim3, lim3)
        nn.init.uniform_(self.l3.bias, -lim3, lim3)


# hereafter assembling *******************************************
class SCM(nn.Module):
    def __init__(self, params: Params):
        super(SCM, self).__init__()
        self.params = params
        self.conv = LocalConv1D(params)
        self.mlp = Mlp(params, self.conv.output_size)
        q = params.q
        if isinstance(q, list):
            self.q = q
            self.q_num = len(q)
        else:
            self.q = [i / (q + 1.) for i in range(1, q + 1)]
            self.q_num = q

    def forward(self, x, t):
        # x is the convolutional outputs, t is the temporal features, q is the quantile level
        tmp_x = self.conv(x)
        y = self.mlp(tmp_x, t)
        return y

    def reinit(self):
        self.conv.reinit()
        self.mlp.reinit()

    def qs_loss(self, pred: torch.Tensor, label: torch.Tensor):
        # pred is (batch_size, num_qs), label is (batch_size)
        ind = pred >= label.unsqueeze(dim=1)
        q_score = (pred - label.unsqueeze(dim=1))[ind].sum() - \
                  ((pred - label.unsqueeze(dim=1)) * torch.tensor(self.q, device=pred.device)).sum()
        return q_score / pred.numel()

    def qs_huber_loss(self, pred: torch.Tensor, label: torch.Tensor, alpha=0.1):
        # pred is (batch_size, num_qs), label is (batch_size)
        ind = pred >= label.unsqueeze(dim=1)

        ind2 = (pred - label.unsqueeze(dim=1)).abs() < alpha
        coeff = torch.tensor(self.q, device=pred.device).repeat((pred.shape[0], 1))
        coeff[ind] = 1. - coeff[ind]
        loss = (pred - label.unsqueeze(dim=1)).abs() - alpha / 2
        loss[ind2] = ((pred - label.unsqueeze(dim=1))[ind2]**2) / (2*alpha)
        loss = coeff * ind

        return loss.sum() / pred.numel()

    def qs_diffrelu_loss(self, pred: torch.Tensor, label: torch.Tensor):
        # pred is (batch_size, num_qs), label is (batch_size)
        ind = pred >= label.unsqueeze(dim=1)

        q_score = (pred - label.unsqueeze(dim=1))[ind].sum() - \
                  ((pred - label.unsqueeze(dim=1)) * torch.tensor(self.q, device=pred.device)).sum()
        add = relu(-pred.diff()).sum()
        return q_score / pred.numel() + self.params.diffrelu * add / label.numel()

    def crps_loss(self, pred: torch.Tensor, label: torch.Tensor, alpha=1, beta=500.):
        # pred is (batch_size, num_qs), label is (batch_size)
        # this func use crps as loss, note that this crps is calculated from several quantiles.
        crps = torch.abs((pred - label.unsqueeze(dim=1))).sum() - \
               alpha * (torch.diff(pred) * torch.tensor(range(1, len(self.q), 1), device=pred.device) *
                torch.tensor(range(len(self.q)-1, 0, -1), device=pred.device)).sum()/pred.shape[1]
        # cover = (label.unsqueeze(dim=1) > pred).sum(dim=0)/label.numel()
        # relia = cover ** 2 - 2 * torch.tensor(self.q, device=label.device) * cover
        return crps / pred.numel()

    def crps_reg(self, pred: torch.Tensor, label: torch.Tensor):
        crps = torch.abs((pred - label.unsqueeze(dim=1))).sum() - \
               (torch.diff(pred) * torch.tensor(range(1, len(self.q), 1), device=pred.device) *
                torch.tensor(range(len(self.q)-1, 0, -1), device=pred.device)).sum()/pred.shape[1]
        reg_term = self.params.reg_lam[0]*self.conv.conv.weight.abs().sum()/self.conv.wsize + \
                   self.params.reg_lam[1]*(self.mlp.l1.weight.abs().sum()+self.mlp.l2.weight.abs().sum()+
                                           self.mlp.l3.weight.abs().sum())/self.mlp.wsize
        return crps / pred.numel() + reg_term

    def qs_reg(self, pred: torch.Tensor, label: torch.Tensor):
        # pred is (batch_size, num_qs), label is (batch_size)
        ind = pred >= label.unsqueeze(dim=1)
        q_score = (pred - label.unsqueeze(dim=1))[ind].sum() - \
                  ((pred - label.unsqueeze(dim=1)) * torch.tensor(self.q, device=pred.device)).sum()
        reg_term = self.params.reg_lam[0]*self.conv.conv.weight.abs().sum()/self.conv.wsize + \
                   self.params.reg_lam[1]*(self.mlp.l1.weight.abs().sum()+self.mlp.l2.weight.abs().sum()+
                                           self.mlp.l3.weight.abs().sum())/self.mlp.wsize
        return q_score / pred.numel() + reg_term

    def mqs_loss(self, pred: torch.Tensor, label: torch.Tensor, alpha=0.5):
        ind = pred >= label.unsqueeze(dim=1)
        qs = (label.unsqueeze(dim=1) - pred) * torch.tensor(self.q, device=pred.device) + \
             (pred - label.unsqueeze(dim=1)) * ind
        loss = qs.mean() + alpha*qs.max(dim=1).values.sum()/label.numel()
        return loss

    def crps_qc(self, pred: torch.Tensor, label: torch.Tensor, alpha=1.1):
        # pred is (batch_size, num_qs), label is (batch_size)
        # this func use crps as loss, note that this crps is calculated from several quantiles.
        crps = torch.abs((pred - label.unsqueeze(dim=1))).sum() - \
               (torch.diff(pred) * torch.tensor(range(1, len(self.q), 1), device=pred.device) *
                torch.tensor(range(len(self.q)-1, 0, -1), device=pred.device)).sum()/pred.shape[1]
        qc_score = (nn.functional.relu(-torch.diff(pred))).sum()
        return (crps + alpha*qc_score) / pred.numel()