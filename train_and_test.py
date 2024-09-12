import math
import torch
import os
import logging
import joblib

import numpy as np
import torch.nn as nn

import utils
from utils import Params, plot_in_train, SampleSet, GraphSampleSet
from network import GCMap, CMap, GGCMap, GCCMap
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator(object):
    def __init__(self, q=5):
        if isinstance(q, list):
            self.q = q
        else:
            self.q = [i / (q + 1.) for i in range(1, q + 1)]
        self.itv_num = (len(self.q) - 1)//2
        self.size = 0
        self.agg = 0.
        self.naps = 0.
        self.mre = 0.
        self.mqs = 0.
        self.qs = [0.] * len(self.q)
        self.coverage = [0.] * len(self.q)
        self.ivt_width = [0.] * self.itv_num

    def qs_eval(self, pred: torch.Tensor, label: torch.Tensor):
        # pred is (batch_size, num_qs), label is (batch_size)
        ind = pred >= label.unsqueeze(dim=1)
        q_score = torch.matmul((label.unsqueeze(dim=1) - pred), torch.diag(torch.tensor(self.q, device=label.device))) \
                  + ind * (pred - label.unsqueeze(dim=1))
        q_score = q_score.sum(dim=0).tolist()
        for count in range(len(self.q)):
            self.qs[count] += q_score[count]

    def rel_eval(self, pred: torch.Tensor, label: torch.Tensor):
        # pred is (batch_size, num_qs), label is (batch_size)
        ind = pred.T > label
        ind_sum = ind.sum(axis=1)
        for count in range(len(self.q)):
            self.coverage[count] += int(ind_sum[count])

    def sha_eval(self, pred: torch.Tensor):
        for i in range(self.itv_num):
            self.ivt_width[i] = self.ivt_width[i] + float((pred[:, -(i+1)] - pred[:, i]).sum())

    def batch_eval(self, pred: torch.Tensor, label: torch.Tensor):
        self.size = self.size + label.numel()
        self.agg = self.agg + float(label.sum())
        self.qs_eval(pred, label)
        self.rel_eval(pred, label)
        self.sha_eval(pred)

    def sum_up(self):
        self.qs = [float('{:.4f}'.format(x/self.size)) for x in self.qs]
        self.coverage = [float('{:.4f}'.format(self.coverage[i]/self.size-self.q[i])) for i in range(len(self.q))]
        self.ivt_width = [float('{:.4f}'.format(x/self.agg)) for x in self.ivt_width]
        ivt_norm = [(self.itv_num-i)/(self.itv_num+1) for i in range(self.itv_num)]
        self.naps = sum([self.ivt_width[i]/ivt_norm[i] for i in range(len(ivt_norm))])/len(ivt_norm)
        self.mre = max([abs(x) for x in self.coverage])
        self.mqs = sum(self.qs)/len(self.qs)
        self.report()

    def report(self):
        logger.info(f"*** Quantiles: {self.q} ***")
        logger.info(f'Size: {self.size}')
        logger.info(f'Q Score: {self.qs}')
        logger.info('Mean score: {:.4f}'.format(self.mqs))
        logger.info(f'Reliability Error: {self.coverage}')
        logger.info('MRE: {:.4f}'.format(self.mre))
        logger.info(f'PINAW: {self.ivt_width}')
        logger.info('NAPS: {:.4f}'.format(self.naps))


class PointFcst(object):
    def __init__(self, params: Params, model_dir):
        self.params = params
        self.model_dir = model_dir
        logger.info("*** Using Conv1D ***")
        # logger.info("*** Using GCN ***")
        logger.info("*** Using CRPS Loss ***")
        self.struc = CMap(params).to(self.params.device)
        self.optimizer = torch.optim.Adam(self.struc.parameters(), lr=self.params.lr)

    def __train__(self, data_loader: DataLoader):
        """training function for one epoch"""
        self.struc.train()
        ret_loss = 0.0
        for (nwp_tensor, temp_tensor, label_tensor) in data_loader:
            nwp_tensor = nwp_tensor.to(self.params.device).float()
            label_tensor = label_tensor.to(self.params.device).float()
            temp_tensor = temp_tensor.to(self.params.device).float()
            loss = torch.zeros(1, device=self.params.device, requires_grad=True)

            self.optimizer.zero_grad()

            loss = loss + self.struc.crps_loss(self.struc(nwp_tensor, temp_tensor), label_tensor)
            loss.backward()
            self.optimizer.step()
            ret_loss = ret_loss + loss.item()
        return ret_loss / len(data_loader)

    def test(self, data_loader: DataLoader, restore=True):
        model_state = torch.load(os.path.join(self.model_dir, 'best.pth.tar'), map_location=self.params.device)
        # print(model_state)
        if restore:
            self.struc.load_state_dict(model_state['structure_dict'])
        self.struc.eval()
        evaluator = Evaluator(self.params.q)
        with torch.no_grad():
            for (nwp_tensor, temp_tensor, label_tensor) in data_loader:
                nwp_tensor = nwp_tensor.to(self.params.device).float()
                label_tensor = label_tensor.to(self.params.device).float()
                temp_tensor = temp_tensor.to(self.params.device).float()
                evaluator.eval(self.struc(nwp_tensor, temp_tensor), label_tensor)

        evaluator.report()

    def evolve(self, data_loader: DataLoader, plot=False):
        train_loss = np.zeros(shape=self.params.num_epochs)
        best_loss = float('inf')

        for epoch in tqdm(range(self.params.num_epochs)):
            train_loss[epoch] = self.__train__(data_loader)
            # plot training process
            if plot:
                plot_in_train(train_loss[:epoch + 1], 'train_loss')

            # Save model
            if train_loss[epoch] <= best_loss:
                best_loss = train_loss[epoch]
                dict_to_save = {'structure_dict': self.struc.state_dict()}
                torch.save(dict_to_save, os.path.join(self.model_dir, 'best.pth.tar'))

                if (epoch > 0) & ((train_loss[epoch-1] - best_loss) < self.params.gap):
                    logger.info("Now the loss stops declining")
                    break


class GlobalFcst(object):
    def __init__(self, params: Params, model_dir: str, method: str, loss_fc='CRPS'):
        self.params = params
        self.mdir = model_dir
        self.loss_fc = loss_fc
        if method == "CNN":
            logger.info("*** Using Conv1D ***")
            self.struc = CMap(params).to(self.params.device)
        elif method == "GCN":
            logger.info("*** Using GCN ***")
            self.struc = GGCMap(params).to(self.params.device)
        elif method == "G-CNN":
            logger.info("*** Using G-CNN ***")
            self.struc = GCCMap(params).to(self.params.device)
        logger.info(f"*** Using {loss_fc} Loss ***")
        self.optimizer = torch.optim.Adam(self.struc.parameters(), lr=self.params.lr)

    def __train__(self, data_loader: DataLoader):
        """training function for one epoch"""
        self.struc.train()
        ret_loss = 0.0
        for (nwp_tensor, temp_tensor, label_tensor) in data_loader:
            nwp_tensor = nwp_tensor.to(self.params.device).float()
            label_tensor = label_tensor.to(self.params.device).float()
            temp_tensor = temp_tensor.to(self.params.device).float()
            loss = torch.zeros(1, device=self.params.device, requires_grad=True)

            self.optimizer.zero_grad()
            if self.loss_fc == 'QS':
                loss = loss + self.struc.qs_loss(self.struc(nwp_tensor, temp_tensor), label_tensor)
            elif self.loss_fc == 'CRPS':
                loss = loss + self.struc.crps_loss(self.struc(nwp_tensor, temp_tensor), label_tensor)
            elif self.loss_fc == 'MQS':
                loss = loss + self.struc.mqs_loss(self.struc(nwp_tensor, temp_tensor), label_tensor)
            loss.backward()
            self.optimizer.step()
            ret_loss = ret_loss + loss.item()
        return ret_loss / len(data_loader)

    def test(self, data_loader: DataLoader, restore=True):
        model_state = torch.load(os.path.join(self.mdir, 'best.pth.tar'), map_location=self.params.device)
        # print(model_state)
        if restore:
            self.struc.load_state_dict(model_state['structure_dict'])
        self.struc.eval()
        evaluator = Evaluator(self.params.q)
        with torch.no_grad():
            for (nwp_tensor, temp_tensor, label_tensor) in data_loader:
                nwp_tensor = nwp_tensor.to(self.params.device).float()
                label_tensor = label_tensor.to(self.params.device).float()
                temp_tensor = temp_tensor.to(self.params.device).float()
                evaluator.batch_eval(self.struc(nwp_tensor, temp_tensor), label_tensor)

        evaluator.sum_up()

    def evolve(self, data_loader: DataLoader, plot=False):
        train_loss = np.zeros(shape=self.params.num_epochs)
        best_loss = float('inf')

        for epoch in tqdm(range(self.params.num_epochs)):
            train_loss[epoch] = self.__train__(data_loader)
            # plot training process
            if plot:
                plot_in_train(train_loss[:epoch + 1], 'train_loss')

            # Save model
            if train_loss[epoch] <= best_loss:
                best_loss = train_loss[epoch]
                dict_to_save = {'structure_dict': self.struc.state_dict()}
                torch.save(dict_to_save, os.path.join(self.mdir, 'best.pth.tar'))

                if (epoch > 0) & ((train_loss[epoch-1] - best_loss) < self.params.gap):
                    logger.info("Now the loss stops declining")
                    break


if __name__ == '__main__':
    dec_nwp_dir = os.path.join('data', 'single_station_experiment', 'nwp')

    task_id = 0
    method = "G-CNN"
    # method = "GCN"
    loss_fc = "QS"
    # lap_type = "random_norm_lap"
    lap_type = 'norm_lap'
    data_dir = os.path.join('data', 'task'+str(task_id))
    model_dir = os.path.join("models", 'task'+str(task_id))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    params = Params(os.path.join(model_dir, 'params.json'))
    if 'norm_lap2' not in lap_type:
        params.lap_path = os.path.join('graph', 'info', 'global_graph', 'task'+str(task_id), lap_type+'.npy')
    else:
        params.lap_path = [os.path.join('graph', 'info', 'global_graph', 'task'+str(task_id), 'norm_lap.npy'),
                           os.path.join('graph', 'info', 'global_graph', 'task'+str(task_id), 'norm_lap2.npy')]
    utils.set_logger(os.path.join(model_dir, 'train.log'))
    logger = logging.getLogger('RPPF.Train')
    logger.info('***** Now we make global experiment *****')
    logger.info(f"**** Task {task_id} ****")

    # now we make single station experiment
    # st_id = 22
    # logger.info('***** Now we make single station experiment *****')

    if method == "CNN":
        st_id = None
        train_set = SampleSet(data_dir, 'train', st_id)
        test_set = SampleSet(data_dir, 'test', st_id)
    else:
        train_set = GraphSampleSet(data_dir, dec_nwp_dir, usage='train')
        test_set = GraphSampleSet(data_dir, dec_nwp_dir, usage='test')

    train_loader = DataLoader(train_set, batch_size=params.batch_size, shuffle=True,
                              pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=params.batch_size, shuffle=True,
                             pin_memory=True, num_workers=4)
    # params.lap_path = os.path.join(params.lap_path, str(st_id).rjust(4, '0') + '.npy')
    cuda_exist = torch.cuda.is_available()
    # Set random seeds for reproducible experiments if necessary
    if cuda_exist:
        params.device = torch.device('cuda:0')
        logger.info('Using Cuda...')
    else:
        params.device = torch.device('cpu')
        logger.info('Not using Cuda...')

    params.ef_path = os.path.join(data_dir, 'edge_feature.npy')

    # model = PointFcst(params, model_dir)
    model = GlobalFcst(params, model_dir, method=method, loss_fc=loss_fc)
    model.evolve(train_loader, plot=True)

    logger.info("#### test the training set ####")
    model.test(train_loader)
    logger.info("#### test the testing set ####")
    model.test(test_loader)



