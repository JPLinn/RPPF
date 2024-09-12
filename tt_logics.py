import math
import torch
import os
import logging
import joblib

import numpy as np
import torch.nn as nn

import utils
from utils import Params, plot_in_train, SampleSet, GraphSampleSet, Evaluator
from new_network import GGC2LW, GCCMap, GGCMap, GGC2L
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm


class GlobalFcst(object):
    def __init__(self, params: Params, model_dir: str, method: str, loss_fc: str, logger):
        self.params = params
        self.mdir = model_dir
        self.loss_fc = loss_fc
        self.logger = logger
        if method == "GCN":
            self.logger.info("*** Using GCN ***")
            self.struc = GGCMap(params).to(self.params.device)
        elif method == "G-CNN":
            self.logger.info("*** Using G-CNN ***")
            self.struc = GCCMap(params).to(self.params.device)
        elif method == "GCN2L":
            self.logger.info("*** Using GCN2L ***")
            self.struc = GGC2L(params).to(self.params.device)
        elif method == "GCN2LW":
            self.logger.info("*** Using GGC2LW ***")
            self.struc = GGC2LW(params).to(self.params.device)
        self.logger.info(f"*** Using {loss_fc} Loss ***")
        self.optimizer = torch.optim.Adam(self.struc.parameters(), lr=self.params.lr, amsgrad=True)

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
            elif self.loss_fc == 'QS-QC':
                loss = loss + self.struc.qs_qc(self.struc(nwp_tensor, temp_tensor), label_tensor)
            elif self.loss_fc == 'CRPS-QC':
                loss = loss + self.struc.crps_qc(self.struc(nwp_tensor, temp_tensor), label_tensor)
            loss.backward()
            self.optimizer.step()
            ret_loss = ret_loss + loss.item()
        return ret_loss / len(data_loader)

    def reinit(self):
        self.struc.reinit()

    def transfer_params(self):
        self.struc.transfer_params()
        self.optimizer = torch.optim.Adam(self.struc.parameters(), lr=self.params.lr, amsgrad=True)

    def freeze_params(self):
        self.struc.freeze_params()

    def detect_trans_params(self):
        return self.struc.detect_trans_params()

    def test(self, data_loader: DataLoader, restore=True, dict_prefix='', evaluate=True, save_forecasts=False):
        model_state = torch.load(os.path.join(self.mdir, dict_prefix+'best.pth.tar'), map_location=self.params.device)
        # print(model_state)
        if restore:
            self.struc.load_state_dict(model_state['structure_dict'])
        self.struc.eval()
        evaluator = Evaluator(self.logger, self.params.q)
        forecasts = np.empty((0, self.params.q))
        with torch.no_grad():
            for (nwp_tensor, temp_tensor, label_tensor) in data_loader:
                nwp_tensor = nwp_tensor.to(self.params.device).float()
                label_tensor = label_tensor.to(self.params.device).float()
                temp_tensor = temp_tensor.to(self.params.device).float()
                res = self.struc(nwp_tensor, temp_tensor)
                res[res < 0.] = 0.
                if evaluate:
                    evaluator.batch_eval(res, label_tensor)
                if save_forecasts:
                    forecasts = np.concatenate((forecasts, res.sort().values.cpu().numpy()))
        if evaluate & save_forecasts:
            return evaluator.sum_up(), forecasts
        elif evaluate:
            return evaluator.sum_up()
        elif save_forecasts:
            return forecasts

    def evolve(self, data_loader: DataLoader, plot=False, show=True, dict_prefix=''):
        train_loss = np.zeros(shape=self.params.num_epochs)
        best_loss = float('inf')
        best_epoch = int(0)
        end_ind = int(0)
        if show:
            for epoch in tqdm(range(self.params.num_epochs)):
                train_loss[epoch] = self.__train__(data_loader)
                # plot training process
                if plot:
                    plot_in_train(train_loss[:epoch + 1], 'train_loss', self.mdir)
                # Save model
                if train_loss[epoch] <= best_loss:
                    if (best_loss - train_loss[epoch]) < self.params.gap:
                        end_ind += 1
                    else:
                        end_ind = 0
                    best_loss = train_loss[epoch]
                    best_epoch = epoch
                    dict_to_save = {'structure_dict': self.struc.state_dict()}
                    torch.save(dict_to_save, os.path.join(self.mdir, dict_prefix+'best.pth.tar'))
                if (end_ind >= 3) | ((epoch - best_epoch) > self.params.max_delay_epochs):
                    self.logger.info("Now the loss stops declining")
                    break
        else:
            for epoch in range(self.params.num_epochs):
                train_loss[epoch] = self.__train__(data_loader)
                # plot training process
                if plot:
                    plot_in_train(train_loss[:epoch + 1], 'train_loss', self.mdir)
                # Save model
                if train_loss[epoch] <= best_loss:
                    if (best_loss - train_loss[epoch]) < self.params.gap:
                        end_ind += 1
                    else:
                        end_ind = 0
                    best_loss = train_loss[epoch]
                    best_epoch = epoch
                    dict_to_save = {'structure_dict': self.struc.state_dict()}
                    torch.save(dict_to_save, os.path.join(self.mdir, dict_prefix + 'best.pth.tar'))
                if (end_ind >= 3) | ((epoch - best_epoch) > self.params.max_delay_epochs):
                    self.logger.info("Now the loss stops declining")
                    break


class SingleOperator(object):
    def __init__(self, method: str, loss_fc: str):
        self.dec_nwp_dir = os.path.join('data', 'single_station_experiment', 'nwp')
        self.data_dir = os.path.join("data")
        self.method = method
        self.loss_fc = loss_fc
        self.model_dir = os.path.join("models", 'task0')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.logger = self._set_logger()
        self.params = self._set_params()
        self.model = GlobalFcst(self.params, self.model_dir, self.method, self.loss_fc, self.logger)

    def _set_params(self):
        params = Params(os.path.join('models', 'task0', 'params.json'))
        cuda_exist = torch.cuda.is_available()
        # Set random seeds for reproducible experiments if necessary
        if cuda_exist:
            params.device = torch.device('cuda:0')
            self.logger.info('Using Cuda...')
        else:
            params.device = torch.device('cpu')
            self.logger.info('Not using Cuda...')
        return params

    def _set_logger(self):
        logger = logging.getLogger(f'SingleOperator.log')
        logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.model_dir, 'singleOperator.log'))
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _reinit_model(self):
        self.model.reinit()

    def transfer_params(self):
        self.model.transfer_params()

    def freeze_params(self):
        self.model.freeze_params()

    def detect_trans_params(self):
        return self.model.detect_trans_params()

    def train(self):
        vali_set = GraphSampleSet(self.data_dir, usage='vali')
        train_set = GraphSampleSet(self.data_dir, usage='train')
        test_set = GraphSampleSet(self.data_dir, usage='test')

        train_loader = DataLoader(train_set, batch_size=self.params.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=4)
        vali_loader = DataLoader(vali_set, batch_size=self.params.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=4)
        self.logger.warning(f'$$$$$ Single Operator $$$$$')
        self.model.evolve(train_loader, plot=True)

        self.logger.info("#### test the training set ####")
        self.model.test(train_loader)
        self.logger.info("#### test the validation set ####")
        self.model.test(vali_loader)
        self.logger.info("#### test the testing set ####")
        self.model.test(test_loader)

    def save_forecasts(self, sdir: str):
        if not os.path.exists(sdir):
            os.makedirs(sdir)
        train_set = GraphSampleSet(self.data_dir, usage='train')
        test_set = GraphSampleSet(self.data_dir, usage='test')
        train_loader = DataLoader(train_set, batch_size=self.params.batch_size, sampler=SequentialSampler(train_set),
                                  pin_memory=True, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, sampler=SequentialSampler(test_set),
                                 pin_memory=True, num_workers=4)

        self.logger.warning(f'***** Save forecasts  *****')
        train_res = self.model.test(train_loader, evaluate=True, save_forecasts=True)
        # np.save(os.path.join(sdir, 'train_set_forecasts'), train_res)
        test_res = self.model.test(test_loader, evaluate=True, save_forecasts=True)

        # np.save(os.path.join(sdir, 'test_set_forecasts'), test_res)


class Trials(object):
    def __init__(self, method, loss_fc, n_trials=10):
        self.dec_nwp_dir = os.path.join('data', 'single_station_experiment', 'nwp')
        self.data_dir = os.path.join("data")
        self.method = method
        self.loss_fc = loss_fc
        self.model_dir = os.path.join("models", 'task0')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.n_trials = n_trials
        self.decided = None
        self.logger = self._set_logger()
        self.logger.warning(f'$$$$$$$$ Trials of {self.n_trials} folds start $$$$$$$$')
        self.params = self._set_params()
        self.model = GlobalFcst(self.params, self.model_dir, self.method, self.loss_fc, self.logger)
        self.res = np.zeros((self.n_trials, 2, 4))  # TODO: here is hard code.

    def _set_params(self):
        params = Params(os.path.join('models', 'task0', 'params.json'))
        cuda_exist = torch.cuda.is_available()
        # Set random seeds for reproducible experiments if necessary
        if cuda_exist:
            params.device = torch.device('cuda:0')
            # params.device = torch.device('cuda:' + str(id_thread % 2))
            self.logger.info('Using Cuda...')
        else:
            params.device = torch.device('cpu')
            self.logger.info('Not using Cuda...')
        return params

    def _set_logger(self):
        logger = logging.getLogger(f'Trials.log')
        logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.model_dir, 'trials.log'))
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _reinit_model(self):
        self.model.reinit()

    def transfer_params(self):
        self.model.transfer_params()

    def detect_trans_params(self):
        return self.model.detect_trans_params()

    def freeze_params(self):
        self.model.freeze_params()

    def start(self, trans_params=False):
        if self.method == "CNN":
            train_set = SampleSet(self.data_dir, 'train')
            test_set = SampleSet(self.data_dir, 'test')
            vali_set = SampleSet(self.data_dir, 'vali')
        else:
            vali_set = GraphSampleSet(self.data_dir, usage='vali')
            train_set = GraphSampleSet(self.data_dir, usage='train')
            test_set = GraphSampleSet(self.data_dir, usage='test')

        train_loader = DataLoader(train_set, batch_size=self.params.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=4)
        vali_loader = DataLoader(vali_set, batch_size=self.params.batch_size, shuffle=True, pin_memory=True,
                                 num_workers=4)
        for k in range(self.n_trials):
            self.logger.warning(f'***** the {k + 1}_th trial *****')
            self._reinit_model()
            if trans_params:
                self.transfer_params()
            # self.freeze_params()
            prefix = str(k+1)
            self.model.evolve(train_loader, plot=True, dict_prefix=prefix, show=False)

            self.logger.info("#### test the training set ####")
            self.res[k, 0, :] = self.model.test(train_loader, dict_prefix=prefix)
            self.logger.info("#### test the validation set ####")
            self.res[k, 0, :] = self.model.test(vali_loader, dict_prefix=prefix)
            self.logger.info("#### test the testing set ####")
            self.res[k, 1, :] = self.model.test(test_loader, dict_prefix=prefix)

        return self.res

    def report(self):
        self.logger.warning('!!!  QS  !!!')
        self.logger.info(f'{self.res[:, 0, 0].tolist()}')
        self.logger.info(f'{self.res[:, 1, 0].tolist()}')
        self.logger.warning('!!!  MRE  !!!')
        self.logger.info(f'{self.res[:, 0, 1].tolist()}')
        self.logger.info(f'{self.res[:, 1, 1].tolist()}')
        self.logger.warning('!!!  NAPS  !!!')
        self.logger.info(f'{self.res[:, 0, 2].tolist()}')
        self.logger.info(f'{self.res[:, 1, 2].tolist()}')
        self.logger.warning('!!!  QCS  !!!')
        self.logger.info(f'{self.res[:, 0, 3].tolist()}')
        self.logger.info(f'{self.res[:, 1, 3].tolist()}')

    def decide(self):
        ind = self.res[:, 0, 1].argsort()[:5]
        decided = ind[self.res[ind, 0, 0].argsort()][0]
        self.decided = decided
        self.logger.warning(f'#### The {decided+1}-th model is selected ####')
        self.logger.warning(f'{self.res[decided, 0, :].tolist()}')
        self.logger.warning(f'{self.res[decided, 1, :].tolist()}')
        return decided

    def del_useless(self):
        for k in range(self.n_trials):
            if k != self.decided:
                os.remove(os.path.join(self.model_dir, str(k+1)+'best.pth.tar'))
            else:
                os.rename(os.path.join(self.model_dir, str(k+1)+'best.pth.tar'),
                          os.path.join(self.model_dir, 'best.pth.tar'))


class Trials_for_hps(object):
    def __init__(self, params, method, loss_fc, n_trials=10):
        self.dec_nwp_dir = os.path.join('data', 'single_station_experiment', 'nwp')
        self.data_dir = os.path.join("data")
        self.method = method
        self.loss_fc = loss_fc
        self.model_dir = params.model_dir
        self.params = params
        params.save(os.path.join(params.model_dir, 'params.json'))
        self.params.device = torch.device('cuda:' + str(params.pro_id % 2))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.n_trials = n_trials
        self.decided = None
        self.logger = self._set_logger()
        self.logger.warning(f'$$$$$$$$ Trials of {self.n_trials} folds start $$$$$$$$')
        self.model = GlobalFcst(self.params, self.model_dir, self.method, self.loss_fc, self.logger)
        self.res = np.zeros((self.n_trials, 2, 4))  # TODO: here is hard code.

    def _set_logger(self):
        logger = logging.getLogger(f'Trials.log')
        logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.model_dir, 'trials.log'))
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _reinit_model(self):
        self.model.reinit()

    def transfer_params(self):
        self.model.transfer_params()

    def detect_trans_params(self):
        return self.model.detect_trans_params()

    def freeze_params(self):
        self.model.freeze_params()

    def start(self, trans_params=False):
        if self.method == "CNN":
            train_set = SampleSet(self.data_dir, 'train')
            test_set = SampleSet(self.data_dir, 'test')
            vali_set = SampleSet(self.data_dir, 'vali')
        else:
            vali_set = GraphSampleSet(self.data_dir, usage='vali')
            train_set = GraphSampleSet(self.data_dir, usage='train')
            test_set = GraphSampleSet(self.data_dir, usage='test')

        train_loader = DataLoader(train_set, batch_size=self.params.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=4)
        vali_loader = DataLoader(vali_set, batch_size=self.params.batch_size, shuffle=True, pin_memory=True,
                                 num_workers=4)
        for k in range(self.n_trials):
            self.logger.warning(f'***** the {k + 1}_th trial *****')
            self._reinit_model()
            if trans_params:
                self.transfer_params()
            # self.freeze_params()
            prefix = str(k+1)
            self.model.evolve(train_loader, plot=True, dict_prefix=prefix, show=False)

            self.logger.info("#### test the training set ####")
            self.res[k, 0, :] = self.model.test(train_loader, dict_prefix=prefix)
            self.logger.info("#### test the validation set ####")
            self.res[k, 0, :] = self.model.test(vali_loader, dict_prefix=prefix)
            self.logger.info("#### test the testing set ####")
            self.res[k, 1, :] = self.model.test(test_loader, dict_prefix=prefix)

        return self.res

    def report(self):
        self.logger.warning('!!!  QS  !!!')
        self.logger.info(f'{self.res[:, 0, 0].tolist()}')
        self.logger.info(f'{self.res[:, 1, 0].tolist()}')
        self.logger.warning('!!!  MRE  !!!')
        self.logger.info(f'{self.res[:, 0, 1].tolist()}')
        self.logger.info(f'{self.res[:, 1, 1].tolist()}')
        self.logger.warning('!!!  NAPS  !!!')
        self.logger.info(f'{self.res[:, 0, 2].tolist()}')
        self.logger.info(f'{self.res[:, 1, 2].tolist()}')
        self.logger.warning('!!!  QCS  !!!')
        self.logger.info(f'{self.res[:, 0, 3].tolist()}')
        self.logger.info(f'{self.res[:, 1, 3].tolist()}')

    def decide(self):
        ind = self.res[:, 0, 1].argsort()[:5]
        decided = ind[self.res[ind, 0, 0].argsort()][0]
        self.decided = decided
        self.logger.warning(f'#### The {decided+1}-th model is selected ####')
        self.logger.warning(f'{self.res[decided, 0, :].tolist()}')
        self.logger.warning(f'{self.res[decided, 1, :].tolist()}')
        return decided

    def del_useless(self):
        for k in range(self.n_trials):
            if k != self.decided:
                os.remove(os.path.join(self.model_dir, str(k+1)+'best.pth.tar'))
            else:
                os.rename(os.path.join(self.model_dir, str(k+1)+'best.pth.tar'),
                          os.path.join(self.model_dir, 'best.pth.tar'))


if __name__ == '__main__':
    mode = 'S'
    if mode == 'S':
        so = SingleOperator(method='GCN2L', loss_fc='QS')
        # so.transfer_params()
        # so.freeze_params()
        # so.train()
        so.save_forecasts('results')
        # err = so.detect_trans_params()
    else:
        tri = Trials(method='GCN2LW', loss_fc='CRPS-QC')
        tri.start(trans_params=True)
        tri.decide()