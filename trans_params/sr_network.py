import os

import torch
import joblib

import numpy as np
import pandas as pd

import torch.nn as nn

from torch.nn.parameter import Parameter
from utils import Params


# tackle the grid points around a station
class LocalConv1D(nn.Module):
    def __init__(self, params: Params):
        super(LocalConv1D, self).__init__()
        self.params = params
        # self.p_ind = joblib.load('graph/point_index.jl')
        self.conv = nn.Conv1d(in_channels=self.params.num_nwp * self.params.num_stats,
                              out_channels=self.params.local_cnn_dims * self.params.num_stats,
                              kernel_size=self.params.local_points, groups=self.params.num_stats)
        self.act = nn.ReLU()
        self.output_size = self.params.local_cnn_dims

    def transfer_params(self, stats: list):
        stats_dir = [str(stat).rjust(4, '0') for stat in stats]
        stats_dir.sort()
        for stat_i, stat_dir in enumerate(stats_dir):
            tmp_dir = os.listdir(os.path.join('ss_models', 'hyperparams_search', stat_dir))[0]
            tmp_model_state = torch.load(
                os.path.join('ss_models', 'hyperparams_search', stat_dir, tmp_dir, 'best.pth.tar'),
                map_location=self.params.device)
            if stat_i == 0:
                cov_weight = tmp_model_state['structure_dict']['conv.conv.weight']
                cov_bias = tmp_model_state['structure_dict']['conv.conv.bias']
            else:
                cov_weight = torch.cat((cov_weight, tmp_model_state['structure_dict']['conv.conv.weight']))
                cov_bias = torch.cat((cov_bias, tmp_model_state['structure_dict']['conv.conv.bias']))
        self.conv.weight = Parameter(cov_weight)
        self.conv.bias = Parameter(cov_bias)

    def detect_trans_params(self, stats: list):
        stats_dir = [str(stat).rjust(4, '0') for stat in stats]
        stats_dir.sort()
        rmse = np.zeros((len(stats), 2))
        start = 0
        for stat_i, stat_dir in enumerate(stats_dir):
            end = start + self.params.local_cnn_dims
            tmp_dir = os.listdir(os.path.join('ss_models', 'hyperparams_search', stat_dir))[0]
            tmp_model_state = torch.load(
                os.path.join('ss_models', 'hyperparams_search', stat_dir, tmp_dir, 'best.pth.tar'),
                map_location=self.params.device)
            cov_weight = tmp_model_state['structure_dict']['conv.conv.weight']
            cov_bias = tmp_model_state['structure_dict']['conv.conv.bias']
            rmse[stat_i, 0] = torch.sqrt((torch.sum((cov_weight - self.conv.weight[start:end]) ** 2) /
                                          torch.sum(torch.abs(cov_weight)))).cpu().detach().numpy()
            rmse[stat_i, 1] = torch.sqrt((torch.sum((cov_bias - self.conv.bias[start:end]) ** 2) /
                                          torch.sum(torch.abs(cov_bias)))).cpu().detach().numpy()
            start = end
        return rmse

    def freeze_params(self):
        self.conv.weight.requires_grad = False
        self.conv.bias.requires_grad = False

    def forward(self, x: torch.Tensor):
        # x is (batch_size, num_nodes, num_features*num_stats)
        return self.act(self.conv(x)).reshape((x.size()[0], self.params.num_stats, -1))

    def reinit(self):
        lim = np.sqrt(self.params.num_stats / (self.conv.kernel_size[0] * self.conv.in_channels))
        nn.init.uniform_(self.conv.weight, -lim, lim)
        nn.init.uniform_(self.conv.bias, -lim, lim)


class WDSGCN2(nn.Module):
    def __init__(self, params: Params):
        super(WDSGCN2, self).__init__()
        self.params = params
        # incidence matrix by node partition
        inc_list = joblib.load(params.partition_inc_path)
        self.linc = torch.tensor(inc_list[params.partition_id][0], device=params.device).float()
        self.rinc = torch.tensor(inc_list[params.partition_id][1], device=params.device).float()
        # edge features by node partition
        ef_list = joblib.load(params.partition_ef_path)
        self.ef = torch.tensor(ef_list[params.partition_id], device=params.device).float().T
        self.cap_feature = torch.tensor(np.load(os.path.join(params.data_dir, 'SR_'+str(params.partition_id), 'cap_feature.npy')), device=params.device).float()
        self.partition_nodes = np.load(
            os.path.join(params.data_dir, 'SR_' + str(params.partition_id), 'partition_nodes.npy'))

        self.k1 = nn.Linear(in_features=params.edge_feature_dims, out_features=params.decf_mlp_hidden)
        self.k2 = nn.Linear(in_features=params.num_temp, out_features=params.decf_mlp_hidden, bias=False)
        self.k3 = nn.Linear(in_features=params.local_cnn_dims, out_features=params.decf_mlp_hidden, bias=False)
        self.k4 = nn.Linear(in_features=params.local_cnn_dims, out_features=params.decf_mlp_hidden, bias=False)
        self.k5 = nn.Linear(in_features=params.decf_mlp_hidden, out_features=params.weight_function_dim)
        self.gc = nn.Linear(in_features=params.local_cnn_dims*params.weight_function_dim,
                            out_features=params.gnn_dim)

        # self.e2k_L1 = nn.Linear(in_features=params.edge_feature_dims, out_features=params.decf_mlp_hidden[0])
        # self.t2k_L2 = nn.Linear(in_features=params.num_temp, out_features=params.decf_mlp_hidden[0], bias=False)
        # self.f2k_L3 = nn.Linear(in_features=params.local_cnn_dims, out_features=params.decf_mlp_hidden[0], bias=False)
        # self.f2k_L4 = nn.Linear(in_features=params.local_cnn_dims, out_features=params.decf_mlp_hidden[0], bias=False)
        # self.k_L3 = nn.Linear(in_features=params.decf_mlp_hidden[0], out_features=params.weight_function_dim)
        self.k_act = nn.LeakyReLU(negative_slope=0.02)
        self.k_act1 = nn.ReLU()
        # self.gc = nn.Linear(in_features=params.local_cnn_dims*params.weight_function_dim,
        #                     out_features=params.gnn_dims[0])
        self.gc_act = nn.ReLU()

        # input_dim = self.partition_end[i] - self.partition_start[i] + self.params.num_temp + \
        #             self.params.gnn_dims[0] * len(self.partition_nodes)
        self.readout0 = nn.Linear(in_features=self.params.gnn_dim * self.partition_nodes.sum(),
                                  out_features=self.params.readout_hid_dim, bias=False)  # dynamic part
        self.readout1 = nn.Linear(in_features=self.params.num_temp, out_features=self.params.readout_hid_dim)  # temporal part
        self.readout2 = nn.Linear(in_features=self.partition_nodes.sum(), out_features=self.params.readout_hid_dim, bias=False)  # static part
        self.readout3 = nn.Linear(in_features=self.params.readout_hid_dim, out_features=self.partition_nodes.sum())
        self.readout_act1 = nn.LeakyReLU()
        self.readout_act2 = nn.ReLU()
        # self.readout_act2 = nn.Softmax(dim=-1)
        self.l2_act = nn.LeakyReLU(negative_slope=0.02)
        self.output_size = self.params.gnn_dim

    def forward(self, x, t):
        # x is (batch_size, num_nodes, num_features)
        # res = torch.randn(x.shape[0], 0, self.params.gnn_dims[0], device=x.device)
        kernel = self.k1(self.ef[:, :-1]) + self.k2(t).unsqueeze(dim=1).repeat(1, self.ef.shape[0], 1)
        kernel = kernel + self.k3(torch.matmul(self.linc.permute(1, 0).unsqueeze(dim=0), x[:,  self.partition_nodes, :].unsqueeze(dim=0)).squeeze(dim=0)) + self.k4(torch.matmul(self.rinc.unsqueeze(dim=0), x.unsqueeze(dim=0)).squeeze(dim=0))
        kernel = self.k_act1(self.k5(self.k_act(kernel)))
        y = torch.mul(kernel.permute(2, 0, 1).unsqueeze(dim=1), torch.matmul(self.rinc, x).permute(2, 0, 1))
        y = torch.matmul(self.linc, y.permute(0, 2, 3, 1))
        y = self.gc_act(self.gc(y.permute(1, 2, 0, 3).reshape(x.shape[0], self.partition_nodes.sum(), -1)))
        weight = self.readout_act2(self.readout3(self.readout_act1(
            self.readout0(y.reshape(x.shape[0], -1)) + self.readout1(t) + self.readout2(self.cap_feature))))
        res = torch.matmul(weight.unsqueeze(dim=1), y)
        return res.reshape(x.shape[0], -1)

    def reinit(self):
        lim0 = 1./np.sqrt(self.readout0.in_features)
        lim1 = 1./np.sqrt(self.readout1.in_features)
        lim2 = 1./np.sqrt(self.readout2.in_features)
        lim3 = 1./np.sqrt(self.readout3.in_features)
        nn.init.uniform_(self.readout0.weight, -lim0, lim0)
        nn.init.uniform_(self.readout1.weight, -lim1, lim1)
        nn.init.uniform_(self.readout1.bias, -lim1, lim1)
        nn.init.uniform_(self.readout2.weight, -lim2, lim2)
        nn.init.uniform_(self.readout3.weight, -lim3, lim3)
        nn.init.uniform_(self.readout3.bias, -lim3, lim3)

        lim_e0 = 1./np.sqrt(self.k1.in_features)
        lim_e1 = 1./np.sqrt(self.k2.in_features)
        lim_e2 = 1./np.sqrt(self.k3.in_features)
        lim_e3 = 1./np.sqrt(self.k4.in_features)
        lim_e4 = 1./np.sqrt(self.k5.in_features)
        lim_gc = 1./np.sqrt(self.gc.in_features)
        nn.init.uniform_(self.k1.weight, -lim_e0, lim_e0)
        nn.init.uniform_(self.k1.bias, -lim_e0, lim_e0)
        nn.init.uniform_(self.k2.weight, -lim_e1, lim_e1)
        nn.init.uniform_(self.k3.weight, -lim_e2, lim_e2)
        nn.init.uniform_(self.k4.weight, -lim_e3, lim_e3)
        nn.init.uniform_(self.k5.weight, -lim_e4, lim_e4)
        nn.init.uniform_(self.k5.bias, -lim_e4, lim_e4)
        nn.init.uniform_(self.gc.weight, -lim_gc, lim_gc)
        nn.init.uniform_(self.gc.bias, -lim_gc, lim_gc)


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

    def forward(self, x):
        # y = self.l1(torch.cat((x, t), dim=1))
        y = self.l1(x)
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


class SMlp(nn.Module):
    def __init__(self, params: Params, conv_input_size: int):
        super(SMlp, self).__init__()
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
        self.l3 = nn.Linear(in_features=self.params.mlp_hid_dim1, out_features=self.q_num)
        # self.bn3 = nn.BatchNorm1d(num_features=self.q_num, affine=False)
        # self.act3 = nn.Softmax(dim=1)
        self.act3 = nn.ReLU()
        self.wsize = self.l1.weight.numel() + self.l2.weight.numel() + self.l3.weight.numel()

    def forward(self, x):
        # y = self.l1(torch.cat((x, t), dim=1))
        y = self.l1(x)
        y = self.act1(y)
        # y = self.l2(y)
        # y = self.act2(y)
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


class GGC2LW(nn.Module):
    def __init__(self, params: Params):
        super(GGC2LW, self).__init__()
        self.params = params
        self.LC = LocalConv1D(self.params)
        self.params.gnn_input_dims = self.LC.output_size
        # self.GGC = GConv1C(self.params)
        self.GGC = WDSGCN2(self.params)
        self.OL = SMlp(params, conv_input_size=self.GGC.output_size)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1 is (batch_size, num_nodes, num_nwp_features), x2 is (batch_size, num_temporal_features)
        return self.OL(torch.cat((self.GGC(self.LC(x1), x2), x2), dim=1))

    def qs_loss(self, pred: torch.Tensor, label: torch.Tensor):
        # pred is (batch_size, num_qs), label is (batch_size)
        ind = pred >= label.unsqueeze(dim=1)
        q_score = ((pred - label.unsqueeze(dim=1)) * torch.tensor(self.OL.weights, device=pred.device))[ind].sum() - \
                  ((pred - label.unsqueeze(dim=1)) * torch.tensor(self.OL.q, device=pred.device) *
                   torch.tensor(self.OL.weights, device=pred.device)).sum()
        return q_score / pred.numel()

    def crps_loss(self, pred: torch.Tensor, label: torch.Tensor, alpha=1.1):
        # pred is (batch_size, num_qs), label is (batch_size)
        # this func use crps as loss, note that this crps is calculated from several quantiles.
        crps = torch.abs((pred - label.unsqueeze(dim=1))).sum() - \
               (torch.diff(pred) * torch.tensor(range(1, len(self.OL.q), 1), device=pred.device) *
                torch.tensor(range(len(self.OL.q)-1, 0, -1), device=pred.device)).sum()/pred.shape[1]
        return crps / pred.numel()

    def mqs_loss(self, pred: torch.Tensor, label: torch.Tensor, alpha=5):
        ind = pred >= label.unsqueeze(dim=1)
        qs = (label.unsqueeze(dim=1) - pred) * torch.tensor(self.OL.q, device=pred.device) + \
             (pred - label.unsqueeze(dim=1)) * ind
        loss = (qs**2).sum()
        # loss = qs.sum() + alpha * qs.max(dim=1).values.sum()
        return loss / pred.numel()

    def qs_qc(self, pred: torch.Tensor, label: torch.Tensor):
        # pred is (batch_size, num_qs), label is (batch_size)
        ind = pred >= label.unsqueeze(dim=1)
        q_score = ((pred - label.unsqueeze(dim=1)) * torch.tensor(self.OL.weights, device=pred.device))[ind].sum() - \
                  ((pred - label.unsqueeze(dim=1)) * torch.tensor(self.OL.q, device=pred.device) *
                   torch.tensor(self.OL.weights, device=pred.device)).sum()
        # qc_score = (nn.functional.relu(-torch.diff(pred))**2).sum()
        qc_score = (nn.functional.relu(-torch.diff(pred))).sum()
        return (q_score + 0.3*qc_score) / pred.numel()

    def crps_qc(self, pred: torch.Tensor, label: torch.Tensor):
        # pred is (batch_size, num_qs), label is (batch_size)
        # this func use crps as loss, note that this crps is calculated from several quantiles.
        crps = torch.abs((pred - label.unsqueeze(dim=1))).sum() - \
               (torch.diff(pred) * torch.tensor(range(1, len(self.OL.q), 1), device=pred.device) *
                torch.tensor(range(len(self.OL.q)-1, 0, -1), device=pred.device)).sum()/pred.shape[1]
        qc_score = (nn.functional.relu(-torch.diff(pred))).sum()
        return (crps + self.params.alpha*qc_score) / pred.numel()

    # def crps_trans_loss(self, pred: torch.Tensor, label: torch.Tensor, alpha=0.1):
    #     crps = torch.abs((pred - label.unsqueeze(dim=1))).sum() - \
    #            (torch.diff(pred) * torch.tensor(range(1, len(self.OL.q), 1), device=pred.device) *
    #             torch.tensor(range(len(self.OL.q)-1, 0, -1), device=pred.device)).sum()/pred.shape[1]
    #     self.LC.weight**2
    #     return crps / pred.numel()

    def reinit(self):
        self.LC.reinit()
        self.GGC.reinit()
        self.OL.reinit()

    def transfer_params(self, stats):
        self.LC.transfer_params(stats)

    def detect_trans_params(self, stats):
        return self.LC.detect_trans_params(stats)

    def freeze_params(self):
        self.LC.freeze_params()