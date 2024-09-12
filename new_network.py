import os

import torch
import joblib

import numpy as np
import pandas as pd

import torch.nn as nn

from torch.nn.parameter import Parameter
from utils import Params


# # the following is for single station
# building blocks

# tackle the grid points around a station
class LocalConv1D(nn.Module):
    def __init__(self, params: Params):
        super(LocalConv1D, self).__init__()
        self.params = params
        # self.p_ind = joblib.load('graph/point_index.jl')
        self.l1 = nn.Linear(in_features=self.params.local_points,
                            out_features=self.params.num_nwp*self.params.num_stats)
        self.bn = nn.BatchNorm1d(num_features=self.params.num_nwp)
        # self.conv = nn.Conv1d(in_channels=self.params.num_nwp * self.params.num_stats,
        #                       out_channels=self.params.local_cnn_dims * self.params.num_stats,
        #                       kernel_size=self.params.local_points, groups=self.params.num_stats)
        self.f = nn.Softmax(dim=0)

    def transfer_params(self):
        pass
        # files = os.listdir(os.path.join('trans_params', 'ss_models', 'hyperparams_search'))
        # files.sort()
        # for file_i, file in enumerate(files):
        #     tmp_dir = os.path.join('trans_params', 'ss_models', 'hyperparams_search', file)
        #     tmp_model_state = torch.load(os.path.join(tmp_dir, os.listdir(tmp_dir)[0], 'best.pth.tar'),
        #                                  map_location=self.params.device)
        #     if file_i == 0:
        #         cov_weight = tmp_model_state['structure_dict']['conv.conv.weight']
        #         cov_bias = tmp_model_state['structure_dict']['conv.conv.bias']
        #     else:
        #         cov_weight = torch.cat((cov_weight, tmp_model_state['structure_dict']['conv.conv.weight']))
        #         cov_bias = torch.cat((cov_bias, tmp_model_state['structure_dict']['conv.conv.bias']))
        # self.conv.weight = Parameter(cov_weight)
        # self.conv.bias = Parameter(cov_bias)

    def detect_trans_params(self):
        pass
        # files = os.listdir(os.path.join('trans_params', 'ss_models', 'hyperparams_search'))
        # files.sort()
        # rmse = np.zeros((self.params.num_stats, 2))
        # start = 0
        # for file_i, file in enumerate(files):
        #     tmp_dir = os.path.join('trans_params', 'ss_models', 'hyperparams_search', file)
        #     end = start + self.params.local_cnn_dims
        #     tmp_model_state = torch.load(os.path.join(tmp_dir, os.listdir(tmp_dir)[0], 'best.pth.tar'),
        #                                  map_location=self.params.device)
        #     cov_weight = tmp_model_state['structure_dict']['conv.conv.weight']
        #     cov_bias = tmp_model_state['structure_dict']['conv.conv.bias']
        #     rmse[file_i, 0] = torch.sqrt((torch.sum((cov_weight - self.conv.weight[start:end]) ** 2) /
        #                                   torch.sum(torch.abs(cov_weight)))).cpu().detach().numpy()
        #     rmse[file_i, 1] = torch.sqrt((torch.sum((cov_bias - self.conv.bias[start:end]) ** 2) /
        #                                   torch.sum(torch.abs(cov_bias)))).cpu().detach().numpy()
        #     start = end
        # return rmse

    def freeze_params(self):
        pass
        # self.conv.weight.requires_grad = False
        # self.conv.bias.requires_grad = False

    def forward(self, x: torch.Tensor):
        # x is (batch_size, num_nodes, num_features*num_stats)
        weight = self.l1(torch.eye(self.params.local_points, self.params.local_points, device=self.params.device))
        weight = self.f(weight)
        y = torch.mul(x, weight).sum(dim=1).reshape(x.shape[0], self.params.num_stats, -1)
        return self.bn(y.permute(0, 2, 1)).permute(0, 2, 1)

    def reinit(self):
        lim1 = 1. / np.sqrt(self.l1.in_features)
        nn.init.uniform_(self.l1.weight, -lim1, lim1)


class GlobalConv1D(nn.Module):
    def __init__(self, params: Params):
        super(GlobalConv1D, self).__init__()
        self.params = params
        self.conv = nn.Conv1d(in_channels=self.params.gcn_input_dims, out_channels=self.params.global_cnn_dims,
                              kernel_size=self.params.num_stats)
        self.act = nn.ReLU()
        self.output_size = self.params.global_cnn_dims

    def forward(self, x: torch.Tensor):
        # x is (batch_size, num_nodes, num_features)
        # output = x.clone()
        # for layer in range(self.num_layers):
        #     output = \
        #         self.acts[layer](self.trans0[layer](output) + self.trans1[layer](torch.matmul(self.norm_lap, output)))
        return self.act(self.conv(x[:, :, :].permute(0, 2, 1)).reshape((x.size()[0], -1)))

    def reinit(self):
        lim = np.sqrt(1. / (self.conv.kernel_size[0] * self.conv.in_channels))
        nn.init.uniform_(self.conv.weight, -lim, lim)
        nn.init.uniform_(self.conv.bias, -lim, lim)


class DECF(nn.Module):
    def __init__(self, params: Params):
        super(DECF, self).__init__()
        self.params = params
        # now incidence is (nodes, edges)
        self.incidence = torch.tensor(np.load(params.inc_path), device=params.device).float()
        # now edge feature matrix is (edges, features)
        self.ef = torch.tensor(np.load(params.ef_path), device=params.device).float().T
        self.mlp = nn.Sequential(
            nn.Linear(in_features=params.edge_feature_dims, out_features=params.decf_mlp_hidden[0]),
            nn.ReLU(),
            nn.Linear(in_features=params.decf_mlp_hidden[0], out_features=params.local_cnn_dims*params.gnn_dims[0])
        )
        self.gc = nn.Linear(in_features=params.gnn_dims[0], out_features=params.gnn_dims[0])
        self.gc.weight.data = torch.eye(params.gnn_dims[0])
        self.gc.weight.requires_grad = False
        self.gc_act = nn.ReLU()
        self.output_size = self.params.num_stats*self.params.gnn_dims[0]

    def forward(self, x):
        # x is (batch_size, num_nodes, num_features)
        kernel = self.mlp(self.ef).reshape(self.ef.shape[0], self.params.gnn_dims[0], -1)
        y = torch.matmul(kernel, torch.matmul(x.permute(0, 2, 1),
                                              self.incidence).permute(0, 2, 1).unsqueeze(dim=-1)).squeeze(dim=-1)
        y = torch.matmul(self.incidence, y)  # now y is (batch_size, nodes, nodal_features)
        y = self.gc_act(self.gc(y))  # now y is (batch_size, nodes, nodal_features)
        return y.reshape((x.size()[0], -1))


class DECF(nn.Module):
    def __init__(self, params: Params):
        super(DECF, self).__init__()
        self.params = params
        # now incidence is (nodes, edges)
        self.incidence = torch.tensor(np.load(params.inc_path), device=params.device).float()
        # now edge feature matrix is (edges, features)
        self.ef = torch.tensor(np.load(params.ef_path), device=params.device).float().T
        self.mlp = nn.Sequential(
            nn.Linear(in_features=params.edge_feature_dims, out_features=params.decf_mlp_hidden[0]),
            nn.ReLU(),
            nn.Linear(in_features=params.decf_mlp_hidden[0], out_features=params.local_cnn_dims*params.gnn_dims[0])
        )
        self.gc = nn.Linear(in_features=params.gnn_dims[0], out_features=params.gnn_dims[0])
        self.gc.weight.data = torch.eye(params.gnn_dims[0])
        self.gc.weight.requires_grad = False
        self.gc_act = nn.ReLU()
        self.output_size = self.params.num_stats*self.params.gnn_dims[0]

    def forward(self, x):
        # x is (batch_size, num_nodes, num_features)
        kernel = self.mlp(self.ef).reshape(self.ef.shape[0], self.params.gnn_dims[0], -1)
        y = torch.matmul(kernel, torch.matmul(x.permute(0, 2, 1),
                                              self.incidence).permute(0, 2, 1).unsqueeze(dim=-1)).squeeze(dim=-1)
        y = torch.matmul(self.incidence, y)  # now y is (batch_size, nodes, nodal_features)
        y = self.gc_act(self.gc(y))  # now y is (batch_size, nodes, nodal_features)
        return y.reshape((x.size()[0], -1))


class SGCN(nn.Module):
    def __init__(self, params: Params):
        super(SGCN, self).__init__()
        self.params = params
        # now incidence is (nodes, edges)
        linc, rinc = joblib.load(params.inc_path)
        self.linc = torch.tensor(linc, device=params.device).float()
        self.rinc = torch.tensor(rinc, device=params.device).float()
        # now edge feature matrix is (edges, features)
        self.ef = torch.tensor(np.load(params.ef_path), device=params.device).float().T
        self.ef2kernel = nn.Sequential(
            nn.Linear(in_features=params.edge_feature_dims, out_features=params.decf_mlp_hidden[0]),
            nn.ReLU(),
            nn.Linear(in_features=params.decf_mlp_hidden[0], out_features=params.weight_function_dim)
        )
        self.gc = nn.Linear(in_features=params.local_cnn_dims*params.weight_function_dim,
                             out_features=params.gnn_dims[0])
        self.gc_act = nn.ReLU()
        self.output_size = self.params.num_stats*self.params.gnn_dims[0]

    def forward(self, x):
        # x is (batch_size, num_nodes, num_features)
        kernel = torch.diag_embed(self.ef2kernel(self.ef).T)
        kernel = torch.matmul(self.linc, kernel).matmul(self.rinc)
        y = torch.matmul(kernel, x.unsqueeze(dim=1))
        y = y.permute(0, 2, -1, 1).reshape(x.shape[0], self.params.num_stats, -1)
        y = self.gc_act(self.gc(y))
        return y.reshape((x.size()[0], -1))

    def reinit(self):
        lim_e0 = 1./np.sqrt(self.ef2kernel[0].in_features)
        lim_e1 = 1./np.sqrt(self.ef2kernel[2].in_features)
        lim_gc = 1. / np.sqrt(self.gc.in_features)
        nn.init.uniform_(self.ef2kernel[0].weight, -lim_e0, lim_e0)
        nn.init.uniform_(self.ef2kernel[2].weight, -lim_e1, lim_e1)
        nn.init.uniform_(self.gc.weight, -lim_gc, lim_gc)


class DSGCN(nn.Module):
    def __init__(self, params: Params):
        super(DSGCN, self).__init__()
        self.params = params
        # now incidence is (nodes, edges)
        linc, rinc = joblib.load(params.inc_path)
        self.linc = torch.tensor(linc, device=params.device).float()
        self.rinc = torch.tensor(rinc, device=params.device).float()
        # now edge feature matrix is (edges, features)
        self.ef = torch.tensor(np.load(params.ef_path), device=params.device).float().T
        self.e2k_L1 = nn.Linear(in_features=params.edge_feature_dims, out_features=params.decf_mlp_hidden[0])
        self.t2k_L2 = nn.Linear(in_features=params.num_temp, out_features=params.decf_mlp_hidden[0], bias=False)
        self.f2k_L3 = nn.Linear(in_features=params.num_nwp, out_features=params.decf_mlp_hidden[0], bias=False)
        self.f2k_L4 = nn.Linear(in_features=params.num_nwp, out_features=params.decf_mlp_hidden[0], bias=False)
        self.k_L3 = nn.Linear(in_features=params.decf_mlp_hidden[0], out_features=params.weight_function_dim)
        self.k_act = nn.LeakyReLU(negative_slope=0.02)
        self.k_act1 = nn.ReLU()
        self.gc = nn.Linear(in_features=params.num_nwp*params.weight_function_dim,
                            out_features=params.gnn_dims[0])
        self.gc_act = nn.ReLU()
        self.output_size = self.params.num_stats*self.params.gnn_dims[0]

    def forward(self, x, t):
        # x is (batch_size, num_nodes, num_features)
        k1 = self.e2k_L1(self.ef[:, :-1])
        k2 = self.t2k_L2(t)
        kernel = k1 + k2.unsqueeze(dim=1).repeat(1, self.ef.shape[0], 1)
        kernel = kernel + self.f2k_L3(torch.matmul(self.linc.permute(1, 0).unsqueeze(dim=0),
                                                   x.unsqueeze(dim=0)).squeeze(dim=0)) + \
                 self.f2k_L4(torch.matmul(self.rinc.unsqueeze(dim=0), x.unsqueeze(dim=0)).squeeze(dim=0))
        kernel = self.k_act1(self.k_L3(self.k_act(kernel)))
        y = torch.mul(kernel.permute(2, 0, 1).unsqueeze(dim=1), torch.matmul(self.rinc, x).permute(2, 0, 1))
        y = torch.matmul(self.linc, y.permute(0, 2, 3, 1))
        y = self.gc_act(self.gc(y.permute(1, 2, 0, 3).reshape(x.shape[0], self.params.num_stats, -1)))
        return y.reshape((x.size()[0], -1))

    def reinit(self):
        lim_e0 = 1./np.sqrt(self.e2k_L1.in_features)
        lim_e1 = 1./np.sqrt(self.t2k_L2.in_features)
        lim_e2 = 1./np.sqrt(self.k_L3.in_features)
        lim_e3 = 1./np.sqrt(self.f2k_L3.in_features)
        lim_e4 = 1./np.sqrt(self.f2k_L4.in_features)
        lim_gc = 1./np.sqrt(self.gc.in_features)
        nn.init.uniform_(self.e2k_L1.weight, -lim_e0, lim_e0)
        nn.init.uniform_(self.e2k_L1.bias, -lim_e0, lim_e0)
        nn.init.uniform_(self.t2k_L2.weight, -lim_e1, lim_e1)
        nn.init.uniform_(self.k_L3.weight, -lim_e2, lim_e2)
        nn.init.uniform_(self.k_L3.bias, -lim_e2, lim_e2)
        nn.init.uniform_(self.f2k_L3.weight, -lim_e3, lim_e3)
        nn.init.uniform_(self.f2k_L4.weight, -lim_e4, lim_e4)
        nn.init.uniform_(self.gc.weight, -lim_gc, lim_gc)
        nn.init.uniform_(self.gc.bias, -lim_gc, lim_gc)


# this is used before readout operation
class DSGCN2(nn.Module):
    def __init__(self, params: Params):
        super(DSGCN2, self).__init__()
        self.params = params
        # now incidence is (nodes, edges)
        linc, rinc = joblib.load(params.inc_path)
        self.linc = torch.tensor(linc, device=params.device).float()
        self.rinc = torch.tensor(rinc, device=params.device).float()
        # now edge feature matrix is (edges, features)
        self.ef = torch.tensor(np.load(params.ef_path), device=params.device).float().T
        self.e2k_L1 = nn.Linear(in_features=params.edge_feature_dims, out_features=params.decf_mlp_hidden[0])
        self.t2k_L2 = nn.Linear(in_features=params.num_temp, out_features=params.decf_mlp_hidden[0], bias=False)
        self.f2k_L3 = nn.Linear(in_features=params.num_nwp, out_features=params.decf_mlp_hidden[0], bias=False)
        self.f2k_L4 = nn.Linear(in_features=params.num_nwp, out_features=params.decf_mlp_hidden[0], bias=False)
        self.k_L3 = nn.Linear(in_features=params.decf_mlp_hidden[0], out_features=params.weight_function_dim)
        self.k_act = nn.LeakyReLU(negative_slope=0.02)
        self.k_act1 = nn.ReLU()
        self.gc = nn.Linear(in_features=params.num_nwp*params.weight_function_dim,
                            out_features=params.gnn_dims[0])
        self.gc_act = nn.ReLU()

    def forward(self, x, t):
        # x is (batch_size, num_nodes, num_features)
        k1 = self.e2k_L1(self.ef[:, :-1])
        k2 = self.t2k_L2(t)
        kernel = k1 + k2.unsqueeze(dim=1).repeat(1, self.ef.shape[0], 1)
        kernel = kernel + self.f2k_L3(torch.matmul(self.linc.permute(1, 0).unsqueeze(dim=0),
                                                   x.unsqueeze(dim=0)).squeeze(dim=0)) + \
                 self.f2k_L4(torch.matmul(self.rinc.unsqueeze(dim=0), x.unsqueeze(dim=0)).squeeze(dim=0))
        kernel = self.k_act1(self.k_L3(self.k_act(kernel)))
        y = torch.mul(kernel.permute(2, 0, 1).unsqueeze(dim=1), torch.matmul(self.rinc, x).permute(2, 0, 1))
        y = torch.matmul(self.linc, y.permute(0, 2, 3, 1))
        y = self.gc_act(self.gc(y.permute(1, 2, 0, 3).reshape(x.shape[0], self.params.num_stats, -1)))
        return y

    def reinit(self):
        lim_e0 = 1./np.sqrt(self.e2k_L1.in_features)
        lim_e1 = 1./np.sqrt(self.t2k_L2.in_features)
        lim_e2 = 1./np.sqrt(self.k_L3.in_features)
        lim_e3 = 1./np.sqrt(self.f2k_L3.in_features)
        lim_e4 = 1./np.sqrt(self.f2k_L4.in_features)
        lim_gc = 1./np.sqrt(self.gc.in_features)
        nn.init.uniform_(self.e2k_L1.weight, -lim_e0, lim_e0)
        nn.init.uniform_(self.e2k_L1.bias, -lim_e0, lim_e0)
        nn.init.uniform_(self.t2k_L2.weight, -lim_e1, lim_e1)
        nn.init.uniform_(self.k_L3.weight, -lim_e2, lim_e2)
        nn.init.uniform_(self.k_L3.bias, -lim_e2, lim_e2)
        nn.init.uniform_(self.f2k_L3.weight, -lim_e3, lim_e3)
        nn.init.uniform_(self.f2k_L4.weight, -lim_e4, lim_e4)
        nn.init.uniform_(self.gc.weight, -lim_gc, lim_gc)
        nn.init.uniform_(self.gc.bias, -lim_gc, lim_gc)


class WDSGCN2(nn.Module):
    def __init__(self, params: Params):
        super(WDSGCN2, self).__init__()
        self.params = params
        # incidence matrix by node partition
        inc_list = joblib.load(params.partition_inc_path)
        self.linc_list = []
        self.rinc_list = []
        self.node_index_list = []
        for i in range(len(inc_list)):
            self.linc_list.append(torch.tensor(inc_list[i][0], device=params.device).float())
            self.rinc_list.append(torch.tensor(inc_list[i][1], device=params.device).float())
            self.node_index_list.append(inc_list[i][2])
        # edge features by node partition
        ef_list = joblib.load(params.partition_ef_path)
        self.ef_list = [torch.tensor(ef_i, device=params.device).float().T for ef_i in ef_list]
        # nodal partition info
        partition_index = np.load(params.partition_index_path)
        self.partition_index = torch.tensor(partition_index, device=params.device).float()
        self.partition_end = list(np.load(params.partition_border_path))
        self.partition_start = [0] + self.partition_end[:-1]
        self.cap_feature = torch.tensor(np.load(params.partition_cap_feature_path), device=params.device).float()
        self.partition_nodes = []
        for i in range(len(self.partition_start)):
            self.partition_nodes.append(np.where(partition_index[:,
                                                 self.partition_start[i]:self.partition_end[i]])[0].tolist())

        k_list1 = []
        k_list2 = []
        k_list3 = []
        k_list4 = []
        k_list5 = []
        gc_list = []
        for i in range(len(self.partition_start)):
            k_list1.append(nn.Linear(in_features=params.edge_feature_dims, out_features=params.decf_mlp_hidden))
            k_list2.append(nn.Linear(in_features=params.num_temp, out_features=params.decf_mlp_hidden, bias=False))
            k_list3.append(nn.Linear(in_features=params.local_cnn_dims, out_features=params.decf_mlp_hidden,
                                     bias=False))
            k_list4.append(nn.Linear(in_features=params.local_cnn_dims, out_features=params.decf_mlp_hidden,
                                     bias=False))
            k_list5.append(nn.Linear(in_features=params.decf_mlp_hidden, out_features=params.weight_function_dim))
            gc_list.append(nn.Linear(in_features=params.local_cnn_dims*params.weight_function_dim,
                                     out_features=params.gnn_dims))
        self.k_list1 = nn.ModuleList(k_list1)
        self.k_list2 = nn.ModuleList(k_list2)
        self.k_list3 = nn.ModuleList(k_list3)
        self.k_list4 = nn.ModuleList(k_list4)
        self.k_list5 = nn.ModuleList(k_list5)
        self.gc_list = nn.ModuleList(gc_list)
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

        module_list0 = []
        module_list1 = []
        module_list2 = []
        module_list3 = []
        for i in range(len(self.partition_end)):
            input_dim = self.partition_end[i] - self.partition_start[i] + self.params.num_temp + \
                        self.params.gnn_dims * len(self.partition_nodes[i])
            module_list0.append(nn.Linear(in_features=self.params.gnn_dims * len(self.partition_nodes[i]),
                                          out_features=2 * input_dim, bias=False))
            module_list1.append(nn.Linear(in_features=self.params.num_temp, out_features=2 * input_dim))
            module_list2.append(nn.Linear(in_features=len(self.partition_nodes[i]), out_features=2 * input_dim,
                                          bias=False))
            module_list3.append(nn.Linear(in_features=2 * input_dim, out_features=len(self.partition_nodes[i])))
        self.readout0 = nn.ModuleList(module_list0)
        self.readout1 = nn.ModuleList(module_list1)
        self.readout2 = nn.ModuleList(module_list2)
        self.readout3 = nn.ModuleList(module_list3)
        self.readout_act1 = nn.LeakyReLU()
        self.readout_act2 = nn.ReLU()
        # self.readout_act2 = nn.Softmax(dim=-1)
        self.conv = nn.Conv1d(in_channels=self.params.gnn_dims, out_channels=self.params.l2_dim,
                              kernel_size=len(self.partition_end))
        self.l2_act = nn.LeakyReLU(negative_slope=0.02)
        self.output_size = self.params.l2_dim

    def forward(self, x, t):
        # x is (batch_size, num_nodes, num_features)
        res = torch.randn(x.shape[0], 0, self.params.gnn_dims, device=x.device)
        for i in range(len(self.partition_end)):
            kernel = self.k_list1[i](self.ef_list[i][:, :-1]) + \
                     self.k_list2[i](t).unsqueeze(dim=1).repeat(1, self.ef_list[i].shape[0], 1)
            kernel = kernel + self.k_list3[i](torch.matmul(self.linc_list[i].permute(1, 0).unsqueeze(dim=0),
                x[:,  self.partition_nodes[i], :].unsqueeze(dim=0)).squeeze(dim=0)) + self.k_list4[i](
                torch.matmul(self.rinc_list[i].unsqueeze(dim=0),
                             x[:, self.node_index_list[i], :].unsqueeze(dim=0)).squeeze(dim=0))
            kernel = self.k_act1(self.k_list5[i](self.k_act(kernel)))
            y = torch.mul(kernel.permute(2, 0, 1).unsqueeze(dim=1), torch.matmul(self.rinc_list[i], x[:, self.node_index_list[i], :]).permute(2, 0, 1))
            y = torch.matmul(self.linc_list[i], y.permute(0, 2, 3, 1))
            y = self.gc_act(self.gc_list[i](y.permute(1, 2, 0, 3).reshape(x.shape[0], len(self.partition_nodes[i]), -1)))
            weight = self.readout_act2(self.readout3[i](self.readout_act1(
                self.readout0[i](y.reshape(x.shape[0], -1)) + self.readout1[i](t) + self.readout2[i](
                    self.cap_feature[self.partition_start[i]:self.partition_end[i]]))))
            res = torch.cat((res, torch.matmul(weight.unsqueeze(dim=1), y)), dim=1)
        res = self.l2_act(self.conv(res.permute(0, 2, 1)))
        return res.reshape(x.shape[0], -1)

    def reinit(self):
        lim_conv = np.sqrt(1. / (self.conv.kernel_size[0] * self.conv.in_channels))
        nn.init.uniform_(self.conv.weight, -lim_conv, lim_conv)
        nn.init.uniform_(self.conv.bias, -lim_conv, lim_conv)

        for i in range(len(self.partition_start)):
            lim0 = 1./np.sqrt(self.readout0[i].in_features)
            lim1 = 1./np.sqrt(self.readout1[i].in_features)
            lim2 = 1./np.sqrt(self.readout2[i].in_features)
            lim3 = 1./np.sqrt(self.readout3[i].in_features)
            nn.init.uniform_(self.readout0[i].weight, -lim0, lim0)
            nn.init.uniform_(self.readout1[i].weight, -lim1, lim1)
            nn.init.uniform_(self.readout1[i].bias, -lim1, lim1)
            nn.init.uniform_(self.readout2[i].weight, -lim2, lim2)
            nn.init.uniform_(self.readout3[i].weight, -lim3, lim3)
            nn.init.uniform_(self.readout3[i].bias, -lim3, lim3)

            lim_e0 = 1./np.sqrt(self.k_list1[i].in_features)
            lim_e1 = 1./np.sqrt(self.k_list2[i].in_features)
            lim_e2 = 1./np.sqrt(self.k_list3[i].in_features)
            lim_e3 = 1./np.sqrt(self.k_list4[i].in_features)
            lim_e4 = 1./np.sqrt(self.k_list5[i].in_features)
            lim_gc = 1./np.sqrt(self.gc_list[i].in_features)
            nn.init.uniform_(self.k_list1[i].weight, -lim_e0, lim_e0)
            nn.init.uniform_(self.k_list1[i].bias, -lim_e0, lim_e0)
            nn.init.uniform_(self.k_list2[i].weight, -lim_e1, lim_e1)
            nn.init.uniform_(self.k_list3[i].weight, -lim_e2, lim_e2)
            nn.init.uniform_(self.k_list4[i].weight, -lim_e3, lim_e3)
            nn.init.uniform_(self.k_list5[i].weight, -lim_e4, lim_e4)
            nn.init.uniform_(self.k_list5[i].bias, -lim_e4, lim_e4)
            nn.init.uniform_(self.gc_list[i].weight, -lim_gc, lim_gc)
            nn.init.uniform_(self.gc_list[i].bias, -lim_gc, lim_gc)


# with hyperparams and params transferring
class TWDSGCN2(nn.Module):
    def __init__(self, params: Params):
        super(TWDSGCN2, self).__init__()
        self.params = params
        # incidence matrix by node partition
        inc_list = joblib.load(params.partition_inc_path)
        self.linc_list = []
        self.rinc_list = []
        self.node_index_list = []
        for i in range(len(inc_list)):
            self.linc_list.append(torch.tensor(inc_list[i][0], device=params.device).float())
            self.rinc_list.append(torch.tensor(inc_list[i][1], device=params.device).float())
            self.node_index_list.append(inc_list[i][2])
        # edge features by node partition
        ef_list = joblib.load(params.partition_ef_path)
        self.ef_list = [torch.tensor(ef_i, device=params.device).float().T for ef_i in ef_list]
        # nodal partition info
        partition_index = np.load(params.partition_index_path)
        self.partition_index = torch.tensor(partition_index, device=params.device).float()
        self.partition_end = list(np.load(params.partition_border_path))
        self.partition_start = [0] + self.partition_end[:-1]
        self.cap_feature = torch.tensor(np.load(params.partition_cap_feature_path), device=params.device).float()
        self.partition_nodes = []
        for i in range(len(self.partition_start)):
            self.partition_nodes.append(np.where(partition_index[:,
                                                 self.partition_start[i]:self.partition_end[i]])[0].tolist())
        # transfer hyperparams
        self.params.decf_mlp_hidden = []
        self.params.readout_hid_dim = []

        ori_dir = os.path.join('trans_params', 'sr_models', 'hyperparams_search')
        for i in range(len(self.partition_nodes)):
            tmp_dir = os.path.join(ori_dir, 'SR_'+str(i))
            tmp_params = Params(os.path.join(tmp_dir, os.listdir(tmp_dir)[0], 'params.json'))
            self.params.decf_mlp_hidden.append(tmp_params.decf_mlp_hidden)
            self.params.readout_hid_dim.append(tmp_params.readout_hid_dim)

        k_list1 = []
        k_list2 = []
        k_list3 = []
        k_list4 = []
        k_list5 = []
        gc_list = []
        for i in range(len(self.partition_start)):
            k_list1.append(nn.Linear(in_features=self.params.edge_feature_dims, out_features=self.params.decf_mlp_hidden[i]))
            k_list2.append(
                nn.Linear(in_features=self.params.num_temp, out_features=self.params.decf_mlp_hidden[i], bias=False))
            k_list3.append(nn.Linear(in_features=self.params.local_cnn_dims, out_features=self.params.decf_mlp_hidden[i],
                                     bias=False))
            k_list4.append(nn.Linear(in_features=self.params.local_cnn_dims, out_features=self.params.decf_mlp_hidden[i],
                                     bias=False))
            k_list5.append(
                nn.Linear(in_features=self.params.decf_mlp_hidden[i], out_features=self.params.weight_function_dim))
            gc_list.append(nn.Linear(in_features=self.params.local_cnn_dims * self.params.weight_function_dim,
                                     out_features=self.params.gnn_dims))
        self.k_list1 = nn.ModuleList(k_list1)
        self.k_list2 = nn.ModuleList(k_list2)
        self.k_list3 = nn.ModuleList(k_list3)
        self.k_list4 = nn.ModuleList(k_list4)
        self.k_list5 = nn.ModuleList(k_list5)
        self.gc_list = nn.ModuleList(gc_list)
        self.k_act = nn.LeakyReLU(negative_slope=0.02)
        self.k_act1 = nn.ReLU()
        self.gc_act = nn.ReLU()

        module_list0 = []
        module_list1 = []
        module_list2 = []
        module_list3 = []
        for i in range(len(self.partition_end)):
            module_list0.append(nn.Linear(in_features=self.params.gnn_dims * len(self.partition_nodes[i]),
                                          out_features=self.params.readout_hid_dim[i], bias=False))
            module_list1.append(nn.Linear(in_features=self.params.num_temp, out_features=self.params.readout_hid_dim[i]))
            module_list2.append(nn.Linear(in_features=len(self.partition_nodes[i]), out_features=self.params.readout_hid_dim[i],
                                          bias=False))
            module_list3.append(nn.Linear(in_features=self.params.readout_hid_dim[i], out_features=len(self.partition_nodes[i])))
        self.readout0 = nn.ModuleList(module_list0)
        self.readout1 = nn.ModuleList(module_list1)
        self.readout2 = nn.ModuleList(module_list2)
        self.readout3 = nn.ModuleList(module_list3)
        self.readout_act1 = nn.LeakyReLU()
        self.readout_act2 = nn.ReLU()
        # self.readout_act2 = nn.Softmax(dim=-1)
        self.conv = nn.Conv1d(in_channels=self.params.gnn_dims, out_channels=self.params.l2_dim,
                              kernel_size=len(self.partition_end))
        self.l2_act = nn.LeakyReLU(negative_slope=0.02)
        self.output_size = self.params.l2_dim

    def forward(self, x, t):
        # x is (batch_size, num_nodes, num_features)
        res = torch.randn(x.shape[0], 0, self.params.gnn_dims, device=x.device)
        for i in range(len(self.partition_end)):
            kernel = self.k_list1[i](self.ef_list[i][:, :-1]) + \
                     self.k_list2[i](t).unsqueeze(dim=1).repeat(1, self.ef_list[i].shape[0], 1)
            kernel = kernel + self.k_list3[i](torch.matmul(self.linc_list[i].permute(1, 0).unsqueeze(dim=0),
                                                           x[:, self.partition_nodes[i], :].unsqueeze(
                                                               dim=0)).squeeze(dim=0)) + self.k_list4[i](
                torch.matmul(self.rinc_list[i].unsqueeze(dim=0),
                             x[:, self.node_index_list[i], :].unsqueeze(dim=0)).squeeze(dim=0))
            kernel = self.k_act1(self.k_list5[i](self.k_act(kernel)))
            y = torch.mul(kernel.permute(2, 0, 1).unsqueeze(dim=1),
                          torch.matmul(self.rinc_list[i], x[:, self.node_index_list[i], :]).permute(2, 0, 1))
            y = torch.matmul(self.linc_list[i], y.permute(0, 2, 3, 1))
            y = self.gc_act(
                self.gc_list[i](y.permute(1, 2, 0, 3).reshape(x.shape[0], len(self.partition_nodes[i]), -1)))
            weight = self.readout_act2(self.readout3[i](self.readout_act1(
                self.readout0[i](y.reshape(x.shape[0], -1)) + self.readout1[i](t) + self.readout2[i](
                    self.cap_feature[self.partition_start[i]:self.partition_end[i]]))))
            res = torch.cat((res, torch.matmul(weight.unsqueeze(dim=1), y)), dim=1)
        res = self.l2_act(self.conv(res.permute(0, 2, 1)))
        return res.reshape(x.shape[0], -1)

    def reinit(self, all=True):
        lim_conv = np.sqrt(1. / (self.conv.kernel_size[0] * self.conv.in_channels))
        nn.init.uniform_(self.conv.weight, -lim_conv, lim_conv)
        nn.init.uniform_(self.conv.bias, -lim_conv, lim_conv)

        if all:
            for i in range(len(self.partition_start)):
                lim0 = 1. / np.sqrt(self.readout0[i].in_features)
                lim1 = 1. / np.sqrt(self.readout1[i].in_features)
                lim2 = 1. / np.sqrt(self.readout2[i].in_features)
                lim3 = 1. / np.sqrt(self.readout3[i].in_features)
                nn.init.uniform_(self.readout0[i].weight, -lim0, lim0)
                nn.init.uniform_(self.readout1[i].weight, -lim1, lim1)
                nn.init.uniform_(self.readout1[i].bias, -lim1, lim1)
                nn.init.uniform_(self.readout2[i].weight, -lim2, lim2)
                nn.init.uniform_(self.readout3[i].weight, -lim3, lim3)
                nn.init.uniform_(self.readout3[i].bias, -lim3, lim3)

                lim_e0 = 1. / np.sqrt(self.k_list1[i].in_features)
                lim_e1 = 1. / np.sqrt(self.k_list2[i].in_features)
                lim_e2 = 1. / np.sqrt(self.k_list3[i].in_features)
                lim_e3 = 1. / np.sqrt(self.k_list4[i].in_features)
                lim_e4 = 1. / np.sqrt(self.k_list5[i].in_features)
                lim_gc = 1. / np.sqrt(self.gc_list[i].in_features)
                nn.init.uniform_(self.k_list1[i].weight, -lim_e0, lim_e0)
                nn.init.uniform_(self.k_list1[i].bias, -lim_e0, lim_e0)
                nn.init.uniform_(self.k_list2[i].weight, -lim_e1, lim_e1)
                nn.init.uniform_(self.k_list3[i].weight, -lim_e2, lim_e2)
                nn.init.uniform_(self.k_list4[i].weight, -lim_e3, lim_e3)
                nn.init.uniform_(self.k_list5[i].weight, -lim_e4, lim_e4)
                nn.init.uniform_(self.k_list5[i].bias, -lim_e4, lim_e4)
                nn.init.uniform_(self.gc_list[i].weight, -lim_gc, lim_gc)
                nn.init.uniform_(self.gc_list[i].bias, -lim_gc, lim_gc)

    def transfer_params(self):
        ori_dir = os.path.join('trans_params', 'sr_models', 'hyperparams_search')
        for i in range(len(self.partition_nodes)):
            tmp_dir = os.path.join(ori_dir, os.listdir(ori_dir, 'SR_'+str(i)))
            tmp_model_state = torch.load(os.path.join(tmp_dir, os.listdir(tmp_dir)[0], 'best.pth.tar'),
                                         map_location=self.params.device)
            self.k_list1[i].weight = Parameter(tmp_model_state['structure_dict']['GGC.k1.weight'])
            self.k_list1[i].bias = Parameter(tmp_model_state['structure_dict']['GGC.k1.bias'])
            self.k_list2[i].weight = Parameter(tmp_model_state['structure_dict']['GGC.k2.weight'])
            self.k_list3[i].weight = Parameter(tmp_model_state['structure_dict']['GGC.k3.weight'])
            self.k_list4[i].weight = Parameter(tmp_model_state['structure_dict']['GGC.k4.weight'])
            self.k_list5[i].weight = Parameter(tmp_model_state['structure_dict']['GGC.k5.weight'])
            self.k_list5[i].bias = Parameter(tmp_model_state['structure_dict']['GGC.k5.bias'])
            self.gc_list[i].weight = Parameter(tmp_model_state['structure_dict']['GGC.gc.weight'])
            self.gc_list[i].bias = Parameter(tmp_model_state['structure_dict']['GGC.gc.bias'])

            self.readout0[i].weight = Parameter(tmp_model_state['structure_dict']['GGC.readout0.weight'])
            self.readout1[i].weight = Parameter(tmp_model_state['structure_dict']['GGC.readout1.weight'])
            self.readout1[i].bias = Parameter(tmp_model_state['structure_dict']['GGC.readout1.bias'])
            self.readout2[i].weight = Parameter(tmp_model_state['structure_dict']['GGC.readout2.weight'])
            self.readout3[i].weight = Parameter(tmp_model_state['structure_dict']['GGC.readout3.weight'])
            self.readout3[i].bias = Parameter(tmp_model_state['structure_dict']['GGC.readout3.bias'])


# TODO: still not practical
class NDSGCN(nn.Module):
    def __init__(self, params: Params):
        super(NDSGCN, self).__init__()
        self.params = params
        # now incidence is (nodes, edges)
        link = np.load(params.link_path)
        self.link = torch.tensor(link, device=params.device).float()
        # now edge feature matrix is (node_num, eq_ef_num, features)
        self.ef = torch.tensor(np.load(params.eq_ef_path), device=params.device).float()
        self.e2k_L1 = nn.Linear(in_features=params.edge_feature_dims, out_features=params.decf_mlp_hidden[0])
        self.t2k_L2 = nn.Linear(in_features=params.num_temp, out_features=params.decf_mlp_hidden[0], bias=False)
        self.k_L3 = nn.Linear(in_features=params.decf_mlp_hidden[0], out_features=params.weight_function_dim)
        self.k_act = nn.LeakyReLU(negative_slope=0.02)
        self.k_act1 = nn.Softmax(dim=-2)
        self.gc = nn.Linear(in_features=params.local_cnn_dims*params.weight_function_dim,
                            out_features=params.gnn_dims[0])
        self.gc_act = nn.ReLU()
        self.output_size = self.params.num_stats*self.params.gnn_dims[0]

    def forward(self, x, t):
        # x is (batch_size, num_nodes, num_features)
        torch.cuda.empty_cache()
        y = self.e2k_L1(self.ef)  # (node_num, eq_ef_num, hidden_dim)
        # (node_num, eq_ef_num, batch_size, hidden_dim)
        y = y.unsqueeze(dim=2).repeat(1, 1, t.shape[0], 1) + self.t2k_L2(t)
        # (batch_size, node_num, eq_ef_num, weight_dim)
        y = self.k_act1(self.k_L3(self.k_act(y.permute(2, 0, 1, 3))))
        # (batch_size, weight_dim, node_num, node_num)
        y = torch.mul(y.permute(0, 3, 1, 2).unsqueeze(dim=-1), self.link).sum(dim=-2)
        # (batch_size, node_num, weight_dim, hidden_feature_dim)
        y = torch.matmul(y.permute(2, 0, 1, 3).unsqueeze(dim=1), x).squeeze(dim=1).permute(1, 0, 2, 3)
        y = self.gc_act(self.gc(y.reshape(x.shape[0], self.params.num_stats, -1)))
        return y.reshape((x.size()[0], -1))

    def reinit(self):
        lim_e0 = 1./np.sqrt(self.e2k_L1.in_features)
        lim_e1 = 1./np.sqrt(self.t2k_L2.in_features)
        lim_e2 = 1./np.sqrt(self.k_L3.in_features)
        lim_gc = 1. / np.sqrt(self.gc.in_features)
        nn.init.uniform_(self.e2k_L1.weight, -lim_e0, lim_e0)
        nn.init.uniform_(self.e2k_L1.bias, -lim_e0, lim_e0)
        nn.init.uniform_(self.t2k_L2.weight, -lim_e1, lim_e1)
        nn.init.uniform_(self.k_L3.weight, -lim_e2, lim_e2)
        nn.init.uniform_(self.k_L3.bias, -lim_e2, lim_e2)
        nn.init.uniform_(self.gc.weight, -lim_gc, lim_gc)
        nn.init.uniform_(self.gc.bias, -lim_gc, lim_gc)


class Readout(nn.Module):
    def __init__(self, params: Params):
        super(Readout, self).__init__()
        self.params = params
        # now incidence is (nodes, edges)
        # now edge feature matrix is (node_num, eq_ef_num, features)
        self.partition = torch.tensor(np.load(params.partition_path), device=params.device).float()
        self.conv = nn.Conv1d(in_channels=self.params.gnn_dims[0], out_channels=self.params.l2_dim,
                              kernel_size=self.partition.shape[0])
        self.act = nn.LeakyReLU(negative_slope=0.02)
        self.output_size = self.params.l2_dim

    def forward(self, x):
        # x is (batch_size, num_nodes, num_features)
        y = self.act(self.conv(torch.matmul(self.partition, x).permute(0, 2, 1)))
        return y.reshape(x.shape[0], -1)

    def reinit(self):
        lim = np.sqrt(1. / (self.conv.kernel_size[0] * self.conv.in_channels))
        nn.init.uniform_(self.conv.weight, -lim, lim)
        nn.init.uniform_(self.conv.bias, -lim, lim)


class DReadout(nn.Module):
    def __init__(self, params: Params):
        super(DReadout, self).__init__()
        self.params = params
        # now incidence is (nodes, edges)
        # now edge feature matrix is (node_num, eq_ef_num, features)
        self.partition_index = torch.tensor(np.load(params.partition_index_path), device=params.device).float()
        self.partition_end = list(np.load(params.partition_border_path))
        self.partition_start = [0] + self.partition_end[:-1]
        self.cap_feature = torch.tensor(np.load(params.partition_cap_feature_path), device=params.device).float()
        module_list1 = []
        module_list2 = []
        module_list3 = []
        for i in range(len(self.partition_end)):
            input_dim = self.partition_end[i] - self.partition_start[i] + self.params.num_temp
            module_list1.append(nn.Linear(in_features=self.params.num_temp, out_features=2 * input_dim))
            module_list2.append(nn.Linear(in_features=input_dim-self.params.num_temp, out_features=2 * input_dim,
                                          bias=False))
            module_list3.append(nn.Linear(in_features=2 * input_dim, out_features=input_dim - self.params.num_temp))
            # module_list3.append(nn.Sequential(
            #     nn.LeakyReLU(),
            #     nn.Linear(in_features=2 * input_dim, out_features=input_dim - self.params.num_temp),
            #     # nn.ReLU()
            #     nn.Softmax(dim=-1)
            # ))
        self.readout1 = nn.ModuleList(module_list1)
        self.readout2 = nn.ModuleList(module_list2)
        self.readout3 = nn.ModuleList(module_list3)
        self.readout_act1 = nn.LeakyReLU()
        self.readout_act2 = nn.ReLU()
        # self.readout_act2 = nn.Softmax(dim=-1)
        self.conv = nn.Conv1d(in_channels=self.params.gnn_dims[0], out_channels=self.params.l2_dim,
                              kernel_size=len(self.partition_end))
        self.act = nn.LeakyReLU(negative_slope=0.02)
        self.output_size = self.params.l2_dim

    def forward(self, x, t):
        # x is (batch_size, num_nodes, num_features)
        weight = torch.randn(x.shape[0], 0, x.shape[1], device=x.device)
        for i in range(len(self.partition_end)):
            weight = torch.cat((weight, torch.matmul(self.partition_index[:,
                                                     self.partition_start[i]:self.partition_end[i]], self.readout_act2(
                self.readout3[i](self.readout_act1(self.readout1[i](t) + self.readout2[i](
                    self.cap_feature[self.partition_start[i]:self.partition_end[i]])))).unsqueeze(
                dim=-1)).permute(0, 2, 1)), dim=1)
        y = self.act(self.conv(torch.matmul(weight, x).permute(0, 2, 1)))
        return y.reshape(x.shape[0], -1)

    def reinit(self):
        lim_conv = np.sqrt(1. / (self.conv.kernel_size[0] * self.conv.in_channels))
        nn.init.uniform_(self.conv.weight, -lim_conv, lim_conv)
        nn.init.uniform_(self.conv.bias, -lim_conv, lim_conv)

        for i in range(len(self.partition_start)):
            lim1 = 1./np.sqrt(self.readout1[i].in_features)
            lim2 = 1./np.sqrt(self.readout2[i].in_features)
            lim3 = 1./np.sqrt(self.readout3[i].in_features)
            nn.init.uniform_(self.readout1[i].weight, -lim1, lim1)
            nn.init.uniform_(self.readout1[i].bias, -lim1, lim1)
            nn.init.uniform_(self.readout2[i].weight, -lim2, lim2)
            nn.init.uniform_(self.readout3[i].weight, -lim3, lim3)
            nn.init.uniform_(self.readout3[i].bias, -lim3, lim3)


class TDReadout(nn.Module):
    def __init__(self, params: Params):
        super(TDReadout, self).__init__()
        self.params = params
        # now incidence is (nodes, edges)
        # now edge feature matrix is (node_num, eq_ef_num, features)
        partition_index = np.load(params.partition_index_path)
        self.partition_index = torch.tensor(partition_index, device=params.device).float()
        self.partition_end = list(np.load(params.partition_border_path))
        self.partition_start = [0] + self.partition_end[:-1]
        self.cap_feature = torch.tensor(np.load(params.partition_cap_feature_path), device=params.device).float()
        self.partition_nodes = []
        for i in range(len(self.partition_start)):
            self.partition_nodes.append(np.where(partition_index[:,
                                                 self.partition_start[i]:self.partition_end[i]])[0].tolist())
        module_list0 = []
        module_list1 = []
        module_list2 = []
        module_list3 = []
        for i in range(len(self.partition_end)):
            input_dim = self.partition_end[i] - self.partition_start[i] + self.params.num_temp + \
                        self.params.gnn_dims[0] * len(self.partition_nodes[i])
            module_list0.append(nn.Linear(in_features=self.params.gnn_dims[0] * len(self.partition_nodes[i]),
                                          out_features=2 * input_dim, bias=False))
            module_list1.append(nn.Linear(in_features=self.params.num_temp, out_features=2 * input_dim))
            module_list2.append(nn.Linear(in_features=len(self.partition_nodes[i]), out_features=2 * input_dim,
                                          bias=False))
            module_list3.append(nn.Linear(in_features=2 * input_dim, out_features=len(self.partition_nodes[i])))
        self.readout0 = nn.ModuleList(module_list0)
        self.readout1 = nn.ModuleList(module_list1)
        self.readout2 = nn.ModuleList(module_list2)
        self.readout3 = nn.ModuleList(module_list3)
        self.readout_act1 = nn.LeakyReLU()
        self.readout_act2 = nn.ReLU()
        # self.readout_act2 = nn.Softmax(dim=-1)
        self.conv = nn.Conv1d(in_channels=self.params.gnn_dims[0], out_channels=self.params.l2_dim,
                              kernel_size=len(self.partition_end))
        self.act = nn.LeakyReLU(negative_slope=0.02)
        self.output_size = self.params.l2_dim

    def forward(self, x, t):
        # x is (batch_size, num_nodes, num_features)
        weight = torch.randn(x.shape[0], 0, x.shape[1], device=x.device)
        for i in range(len(self.partition_end)):
            weight = torch.cat((weight, torch.matmul(self.partition_index[:,
                                                     self.partition_start[i]:self.partition_end[i]], self.readout_act2(
                self.readout3[i](self.readout_act1(self.readout0[i](x[:, self.partition_nodes[i], :].reshape(
                    x.shape[0], -1)) + self.readout1[i](t) + self.readout2[i](
                    self.cap_feature[self.partition_start[i]:self.partition_end[i]])))).unsqueeze(dim=-1)).permute(
                0, 2, 1)), dim=1)
        y = self.act(self.conv(torch.matmul(weight, x).permute(0, 2, 1)))
        return y.reshape(x.shape[0], -1)

    def reinit(self):
        lim_conv = np.sqrt(1. / (self.conv.kernel_size[0] * self.conv.in_channels))
        nn.init.uniform_(self.conv.weight, -lim_conv, lim_conv)
        nn.init.uniform_(self.conv.bias, -lim_conv, lim_conv)

        for i in range(len(self.partition_start)):
            lim0 = 1./np.sqrt(self.readout0[i].in_features)
            lim1 = 1./np.sqrt(self.readout1[i].in_features)
            lim2 = 1./np.sqrt(self.readout2[i].in_features)
            lim3 = 1./np.sqrt(self.readout3[i].in_features)
            nn.init.uniform_(self.readout0[i].weight, -lim0, lim0)
            nn.init.uniform_(self.readout1[i].weight, -lim1, lim1)
            nn.init.uniform_(self.readout1[i].bias, -lim1, lim1)
            nn.init.uniform_(self.readout2[i].weight, -lim2, lim2)
            nn.init.uniform_(self.readout3[i].weight, -lim3, lim3)
            nn.init.uniform_(self.readout3[i].bias, -lim3, lim3)


# bonding
# global graphical convolutional network
class GGCMap(nn.Module):
    def __init__(self, params: Params):
        super(GGCMap, self).__init__()
        self.params = params
        self.LC = LocalConv1D(self.params)
        # self.GGC = GConv1C(self.params)
        self.GGC = DSGCN(self.params)
        self.OL = QuantileOutputLayer(input_size=self.GGC.output_size,
                                      output_size=self.params.mlp_output_dims, q=self.params.q)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1 is (batch_size, num_nodes, num_nwp_features), x2 is (batch_size, num_temporal_features)
        # return self.OL(torch.cat((self.GGC(self.LC(x1), x2), x2), dim=1))
        return self.OL(self.GGC(self.LC(x1), x2))

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

    def crps_qc(self, pred: torch.Tensor, label: torch.Tensor, alpha=1.1):
        # pred is (batch_size, num_qs), label is (batch_size)
        # this func use crps as loss, note that this crps is calculated from several quantiles.
        crps = torch.abs((pred - label.unsqueeze(dim=1))).sum() - \
               (torch.diff(pred) * torch.tensor(range(1, len(self.OL.q), 1), device=pred.device) *
                torch.tensor(range(len(self.OL.q)-1, 0, -1), device=pred.device)).sum()/pred.shape[1]
        qc_score = (nn.functional.relu(-torch.diff(pred))).sum()
        return (crps + 0.8*qc_score) / pred.numel()

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

    def transfer_params(self):
        self.LC.transfer_params()

    def detect_trans_params(self):
        return self.LC.detect_trans_params()

    def freeze_params(self):
        self.LC.freeze_params()


# global graphical convolutional network with readout operation
class GGC2L(nn.Module):
    def __init__(self, params: Params):
        super(GGC2L, self).__init__()
        self.params = params
        self.LC = LocalConv1D(self.params)
        # self.GGC = GConv1C(self.params)
        self.GGC = DSGCN2(self.params)
        self.Read = TDReadout(self.params)
        self.OL = QuantileOutputLayer(input_size=self.Read.output_size+self.params.num_temp,
                                      output_size=self.params.mlp_output_dims, q=self.params.q)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1 is (batch_size, num_nodes, num_nwp_features), x2 is (batch_size, num_temporal_features)
        return self.OL(torch.cat((self.Read(self.GGC(self.LC(x1), x2), x2), x2), dim=1))

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

    def crps_qc(self, pred: torch.Tensor, label: torch.Tensor, alpha=1.1):
        # pred is (batch_size, num_qs), label is (batch_size)
        # this func use crps as loss, note that this crps is calculated from several quantiles.
        crps = torch.abs((pred - label.unsqueeze(dim=1))).sum() - \
               (torch.diff(pred) * torch.tensor(range(1, len(self.OL.q), 1), device=pred.device) *
                torch.tensor(range(len(self.OL.q)-1, 0, -1), device=pred.device)).sum()/pred.shape[1]
        qc_score = (nn.functional.relu(-torch.diff(pred))).sum()
        return (crps + 0.8*qc_score) / pred.numel()

    # def crps_trans_loss(self, pred: torch.Tensor, label: torch.Tensor, alpha=0.1):
    #     crps = torch.abs((pred - label.unsqueeze(dim=1))).sum() - \
    #            (torch.diff(pred) * torch.tensor(range(1, len(self.OL.q), 1), device=pred.device) *
    #             torch.tensor(range(len(self.OL.q)-1, 0, -1), device=pred.device)).sum()/pred.shape[1]
    #     self.LC.weight**2
    #     return crps / pred.numel()

    def reinit(self):
        self.LC.reinit()
        self.GGC.reinit()
        self.Read.reinit()
        self.OL.reinit()

    def transfer_params(self):
        self.LC.transfer_params()

    def detect_trans_params(self):
        return self.LC.detect_trans_params()

    def freeze_params(self):
        self.LC.freeze_params()


class GGC2LW(nn.Module):
    def __init__(self, params: Params):
        super(GGC2LW, self).__init__()
        self.params = params
        self.LC = LocalConv1D(self.params)
        self.params.gnn_input_dims = 6
        # self.GGC = GConv1C(self.params)
        self.GGC = TWDSGCN2(self.params)
        self.OL = QuantileOutputLayer(input_size=self.GGC.output_size+self.params.num_temp,
                                      output_size=self.params.mlp_output_dims, q=self.params.q)

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

    def reinit(self):
        self.LC.reinit()
        self.GGC.reinit()
        self.OL.reinit()

    def transfer_params(self):
        self.LC.transfer_params()

    def detect_trans_params(self):
        return self.LC.detect_trans_params()

    def freeze_params(self):
        self.LC.freeze_params()


# G-CNN
class GCCMap(nn.Module):
    def __init__(self, params: Params):
        super(GCCMap, self).__init__()
        self.params = params
        self.LC = LocalConv1D(self.params)
        self.params.gcn_input_dims = self.LC.output_size
        self.GCN = GlobalConv1D(self.params)
        self.OL = QuantileOutputLayer(input_size=self.GCN.output_size+self.params.num_temp,
                                      output_size=self.params.mlp_output_dims, q=self.params.q, bq=self.params.weights)

    def transfer_params(self):
        self.LC.transfer_params()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # x1 is (batch_size, num_nodes, num_nwp_features), x2 is (batch_size, num_temporal_features)
        return self.OL(torch.cat((self.GCN(self.LC(x1)), x2), dim=1))

    def qs_loss(self, pred: torch.Tensor, label: torch.Tensor):
        # pred is (batch_size, num_qs), label is (batch_size)
        ind = pred >= label.unsqueeze(dim=1)
        q_score = ((pred - label.unsqueeze(dim=1)) * torch.tensor(self.OL.weights, device=pred.device))[ind].sum() - \
                  ((pred - label.unsqueeze(dim=1)) * torch.tensor(self.OL.q, device=pred.device) *
                   torch.tensor(self.OL.weights, device=pred.device)).sum()
        return q_score / pred.numel()

    def crps_loss(self, pred: torch.Tensor, label: torch.Tensor):
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
        loss = qs.sum() + alpha*qs.max(dim=1).values.sum()
        return loss / pred.numel()

    def reinit(self):
        self.LC.reinit()
        self.GCN.reinit()
        self.OL.reinit()

    def transfer_params(self):
        self.LC.transfer_params()

    def freeze_params(self):
        self.LC.freeze_params()


# probabilistic models
class QuantileOutputLayer(nn.Module):
    def __init__(self, input_size, output_size, q=5, bq=False):
        super(QuantileOutputLayer, self).__init__()
        self.input_size = input_size
        if isinstance(q, list):
            self.q = q
        else:
            self.q = [i/(q+1.) for i in range(1, q + 1)]
        if bq:
            prop = [np.exp(abs(0.5-tmp_q)) for tmp_q in self.q]
            scale = 1./sum(prop)
            self.weights = [scale*weight for weight in prop]
        else:
            self.weights = [1. for tmp_q in self.q]

        self.l1 = nn.Linear(in_features=input_size, out_features=output_size)
        self.l2 = nn.Linear(in_features=output_size, out_features=len(self.q))
        self.act = nn.LeakyReLU()
        self.sp = nn.Softplus()

    def forward(self, x):
        # return self.sp(self.l2(self.act(self.l1(x)))).cumsum(dim=1)
        return self.l2(self.act(self.l1(x)))

        # y = self.l2(self.act(self.l1(x)))
        # for i in range(len(self.q)-1):
        #     y[:, i+1] = y[:, i] + nn.functional.relu(y[:, i+1]-y[:, i])
        # return y

        # return self.sp(self.l2(self.act(self.l1(x))))

    def reinit(self):
        lim1 = 1./np.sqrt(self.l1.in_features)
        lim2 = 1./np.sqrt(self.l2.in_features)
        nn.init.uniform_(self.l1.weight, -lim1, lim1)
        nn.init.uniform_(self.l1.bias, -lim1, lim1)
        nn.init.uniform_(self.l2.weight, -lim2, lim2)
        nn.init.uniform_(self.l2.bias, -lim2, lim2)

