import math
import time
import torch
import os
import logging
import joblib

import numpy as np
import torch.nn as nn

import utils
from utils import Params, plot_in_train, SampleSet, Evaluator
from sr_network import GGC2LW
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from multiprocessing import Process


class Fcst(object):
    def __init__(self, params: Params, model_dir, logger):
        self.params = params
        self.logger = logger
        self.model_dir = model_dir
        self.struc = GGC2LW(params).to(self.params.device)
        self.optimizer = torch.optim.Adam(self.struc.parameters(), lr=self.params.lr, amsgrad=True)

    def __train__(self, data_loader: DataLoader):
        """training function for one epoch"""
        self.struc.train()
        ret_loss = 0.0
        for (nwp_tensor, temp_tensor, label_tensor) in data_loader:
            nwp_tensor = nwp_tensor.to(self.params.device).float()
            label_tensor = label_tensor.to(self.params.device).float()
            temp_tensor = temp_tensor.to(self.params.device).float()
            nwp_tensor = nwp_tensor.permute(0, 2, 1)
            loss = torch.zeros(1, device=self.params.device, requires_grad=True)

            self.optimizer.zero_grad()

            loss = loss + self.struc.crps_qc(self.struc(nwp_tensor, temp_tensor), label_tensor)
            loss.backward()
            self.optimizer.step()
            ret_loss = ret_loss + loss.item()
        return ret_loss / len(data_loader)

    def test(self, data_loader: DataLoader, restore=True, dict_prefix='', evaluate=True, save_forecasts=False):
        model_state = torch.load(os.path.join(self.model_dir, dict_prefix+'best.pth.tar'),
                                 map_location=self.params.device)
        # print(model_state)
        if restore:
            self.struc.load_state_dict(model_state['structure_dict'])
        self.struc.eval()
        if evaluate:
            evaluator = Evaluator(self.logger, self.params.q)
        if save_forecasts:
            forecasts = np.empty((0, len(self.params) if isinstance(self.params.q, list) else self.params.q))
        with torch.no_grad():
            for (nwp_tensor, temp_tensor, label_tensor) in data_loader:
                nwp_tensor = nwp_tensor.to(self.params.device).float()
                label_tensor = label_tensor.to(self.params.device).float()
                temp_tensor = temp_tensor.to(self.params.device).float()
                nwp_tensor = nwp_tensor.permute(0, 2, 1)
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

    def evolve(self, data_loader: DataLoader, plot=False, dict_prefix='', show=True):
        train_loss = np.zeros(shape=self.params.num_epochs)
        best_loss = float('inf')
        if show:
            for epoch in tqdm(range(self.params.num_epochs)):
                train_loss[epoch] = self.__train__(data_loader)
                # plot training process
                if plot:
                    plot_in_train(train_loss[:epoch + 1], 'train_loss', self.model_dir)

                # Save model
                if train_loss[epoch] <= best_loss:
                    best_loss = train_loss[epoch]
                    dict_to_save = {'structure_dict': self.struc.state_dict()}
                    torch.save(dict_to_save, os.path.join(self.model_dir, dict_prefix + 'best.pth.tar'))

                    if (epoch > 0) & ((train_loss[epoch - 1] - best_loss) < self.params.gap):
                        self.logger.info("Now the loss stops declining")
                        break
        else:
            for epoch in range(self.params.num_epochs):
                train_loss[epoch] = self.__train__(data_loader)
                # plot training process
                if plot:
                    plot_in_train(train_loss[:epoch + 1], 'train_loss', self.model_dir)

                # Save model
                if train_loss[epoch] <= best_loss:
                    best_loss = train_loss[epoch]
                    dict_to_save = {'structure_dict': self.struc.state_dict()}
                    torch.save(dict_to_save, os.path.join(self.model_dir, dict_prefix + 'best.pth.tar'))

                    if (epoch > 0) & ((train_loss[epoch - 1] - best_loss) < self.params.gap):
                        self.logger.info("Now the loss stops declining")
                        break

    def reinit(self):
        self.struc.reinit()

    def transfer_params(self, stats):
        self.struc.transfer_params(stats)
        self.optimizer = torch.optim.Adam(self.struc.parameters(), lr=self.params.lr, amsgrad=True)

    def freeze_params(self):
        self.struc.freeze_params()

    def detect_trans_params(self, stats):
        return self.struc.detect_trans_params(stats)


class SingleOperator(object):
    def __init__(self, subreg_id, thread_id=0):
        self.data_dir = os.path.join('..', 'data', 'transfer_params', 'subregions', 'SR_' + str(subreg_id))
        self.model_dir = os.path.join("sr_models", 'SR_' + str(subreg_id))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.subreg_id = subreg_id
        self.logger = self._set_logger()
        self.params = self._set_params(thread_id)
        self.model = Fcst(self.params, self.model_dir, self.logger)

    def _set_logger(self):
        logger = logging.getLogger(f'SingleOperator_Subregion-{self.subreg_id}.log')
        logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.model_dir, 'so.log'))
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _set_params(self, thread_id):
        params = Params(os.path.join('sr_models', 'params.json'))
        cuda_exist = torch.cuda.is_available()
        # Set random seeds for reproducible experiments if necessary
        if cuda_exist:
            params.device = torch.device('cuda:'+str(thread_id % 2))
            self.logger.info('Using Cuda...')
        else:
            params.device = torch.device('cpu')
            self.logger.info('Not using Cuda...')
        params.partition_id = self.subreg_id
        params.num_stats = np.load(os.path.join(self.data_dir, 'partition_nodes.npy')).size
        return params

    def _reinit_model(self):
        self.model.reinit()

    def transfer_params(self):
        stats = joblib.load(os.path.join(self.data_dir, 'stats_involved.jl'))
        self.model.transfer_params(stats)

    def freeze_params(self):
        self.model.freeze_params()

    def detect_trans_params(self):
        stats = joblib.load(os.path.join(self.data_dir, 'stats_involved.jl'))
        return self.model.detect_trans_params(stats)

    def train(self):
        train_set = SampleSet(self.data_dir, 'train')
        vali_set = SampleSet(self.data_dir, 'vali')
        test_set = SampleSet(self.data_dir, 'test')

        train_loader = DataLoader(train_set, batch_size=self.params.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=4)
        vali_loader = DataLoader(vali_set, batch_size=self.params.batch_size, shuffle=True, pin_memory=True,
                                 num_workers=4)
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=4)
        self.logger.warning(f'***** Training and test of Subregion No.{self.subreg_id} *****')
        self.model.evolve(train_loader, plot=True, show=False)

        self.logger.info("#### test the training set ####")
        self.model.test(vali_loader)
        self.logger.info("#### test the testing set ####")
        self.model.test(test_loader)

    def save_forecasts(self, sdir: str):
        save_dir = os.path.join(sdir, str(self.sta_id).rjust(4, '0'))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        train_set = SampleSet(self.data_dir, 'train', self.sta_id)
        test_set = SampleSet(self.data_dir, 'test', self.sta_id)
        train_loader = DataLoader(train_set, batch_size=self.params.batch_size, sampler=SequentialSampler(train_set),
                                  pin_memory=True, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, sampler=SequentialSampler(test_set),
                                 pin_memory=True, num_workers=4)

        self.logger.warning(f'***** Save forecasts of Station No.{self.sta_id} *****')

        train_res = self.model.test(train_loader, evaluate=False, save_forecasts=True)
        np.save(os.path.join(save_dir, 'train_set_forecasts'), train_res)
        test_res = self.model.test(test_loader, evaluate=False, save_forecasts=True)
        np.save(os.path.join(save_dir, 'test_set_forecasts'), test_res)


class Trials(object):
    def __init__(self, subreg_id, id_thread=0, n_trials=10):
        self.data_dir = os.path.join('..', 'data', 'transfer_params', 'subregions', 'SR_' + str(subreg_id))
        self.model_dir = os.path.join("sr_models", 'SR_' + str(subreg_id))
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.n_trials = n_trials
        self.subreg_id = subreg_id
        self.decided = None
        self.logger = self._set_logger()
        self.logger.warning(f'############ Trial of Subregion {subreg_id} starts ############')
        self.params = self._set_params(id_thread)
        self.model = Fcst(self.params, self.model_dir, self.logger)
        self.res = np.zeros((self.n_trials, 2, 4))  # TODO: here is hard code.

    def _set_params(self, id_thread):
        params = Params(os.path.join('sr_models', 'params.json'))
        cuda_exist = torch.cuda.is_available()
        # Set random seeds for reproducible experiments if necessary
        if cuda_exist:
            params.device = torch.device('cuda:'+str(id_thread%2))
            self.logger.info('Using Cuda...')
        else:
            params.device = torch.device('cpu')
            self.logger.info('Not using Cuda...')
        params.partition_id = self.subreg_id
        params.num_stats = np.load(os.path.join(self.data_dir, 'partition_nodes.npy')).size
        return params

    def _set_logger(self):
        logger = logging.getLogger(f'Trials_Subregion-{self.subreg_id}.log')
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
        stats = joblib.load(os.path.join(self.data_dir, 'stats_involved.jl'))
        self.model.transfer_params(stats)

    def freeze_params(self):
        self.model.freeze_params()

    def detect_trans_params(self):
        stats = joblib.load(os.path.join(self.data_dir, 'stats_involved.jl'))
        return self.model.detect_trans_params(stats)

    def start(self, trans=False, debug=False):
        train_set = SampleSet(self.data_dir, 'train')
        test_set = SampleSet(self.data_dir, 'test')
        vali_set = SampleSet(self.data_dir, 'vali')

        train_loader = DataLoader(train_set, batch_size=self.params.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=4)
        vali_loader = DataLoader(vali_set, batch_size=self.params.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=4)
        for k in range(self.n_trials):
            self.logger.warning(f'***** the {k + 1}_th time of Subregion {self.subreg_id} *****')
            self._reinit_model()
            if trans:
                self.transfer_params()
                if debug:
                    print(self.detect_trans_params())
            prefix = str(k+1)+'_'
            self.model.evolve(train_loader, plot=True, dict_prefix=prefix, show=False)
            if debug:
                print(self.detect_trans_params())
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
        self.logger.warning(f'#### The {decided+1}-th model is selected for Subregion {self.subreg_id} ####')
        self.logger.warning(f'{self.res[decided, 0, :].tolist()}')
        self.logger.warning(f'{self.res[decided, 1, :].tolist()}')
        return decided

    def del_useless(self):
        for k in range(self.n_trials):
            if k != self.decided:
                os.remove(os.path.join(self.model_dir, str(k+1)+'_best.pth.tar'))
            else:
                os.rename(os.path.join(self.model_dir, str(k+1)+'_best.pth.tar'),
                          os.path.join(self.model_dir, 'best.pth.tar'))


class Trials_for_hps(object):
    def __init__(self, params, n_trials=10):
        self.data_dir = os.path.join('..', 'data', 'transfer_params', 'subregions', 'SR_' + str(params.partition_id))
        self.model_dir = params.model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.n_trials = n_trials
        self.params = params
        params.save(os.path.join(params.model_dir, 'params.json'))
        self.params.device = torch.device('cuda:' + str(params.pro_id % 2))
        self.subreg_id = params.partition_id
        self.decided = None
        self.logger = self._set_logger()
        self.logger.warning(f'############ Trial of Subregion {self.subreg_id} starts ############')
        self.model = Fcst(self.params, self.model_dir, self.logger)
        self.res = np.zeros((self.n_trials, 2, 4))  # TODO: here is hard code.

    def _set_logger(self):
        logger = logging.getLogger(f'Trials_Subregion-{self.subreg_id}.log')
        logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.model_dir, 'trials.log'))
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def _reinit_model(self):
        self.model.reinit()

    def start(self):
        train_set = SampleSet(self.data_dir, 'train')
        test_set = SampleSet(self.data_dir, 'test')
        vali_set = SampleSet(self.data_dir, 'vali')

        train_loader = DataLoader(train_set, batch_size=self.params.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=0)
        vali_loader = DataLoader(vali_set, batch_size=self.params.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=self.params.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=0)
        for k in range(self.n_trials):
            self.logger.warning(f'***** the {k + 1}_th time of Subregion {self.subreg_id} *****')
            self._reinit_model()
            prefix = str(k+1)+'_'
            self.model.evolve(train_loader, plot=False, dict_prefix=prefix, show=False)

            self.logger.info("#### test the training set ####")
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
        self.logger.warning(f'#### The {decided+1}-th model is selected for Subregion {self.subreg_id} ####')
        self.logger.warning(f'{self.res[decided, 0, :].tolist()}')
        self.logger.warning(f'{self.res[decided, 1, :].tolist()}')
        return decided

    def del_useless(self):
        for k in range(self.n_trials):
            if k != self.decided:
                os.remove(os.path.join(self.model_dir, str(k+1)+'_best.pth.tar'))
            else:
                os.rename(os.path.join(self.model_dir, str(k+1)+'_best.pth.tar'),
                          os.path.join(self.model_dir, 'best.pth.tar'))


def traverse_all_stats(sta_ids: list, save_dir: str, id_thread: int, n_trials=20):
    sta_ids.sort()
    process_save = []
    decided_save = []
    for count, sta_id in enumerate(sta_ids):
        trials = Trials(sta_id, id_thread, n_trials=n_trials)
        process_save = process_save + [trials.start()]
        trials.report()
        decided_save = decided_save + [trials.decide()]
        trials.del_useless()
    print(f"*** The {id_thread}-th process ended. ***")
    joblib.dump(process_save, os.path.join(save_dir, str(id_thread)+'_process_save.jl'))
    joblib.dump(decided_save, os.path.join(save_dir, str(id_thread)+'_decided_save.jl'))
    # return process_save, decided_save


def join_res(n_res: int, res_dir):
    file_suffix1 = '_process_save.jl'
    file_suffix2 = '_decided_save.jl'
    process = []
    decided = []
    for i in range(1, n_res+1):
        process = process + joblib.load(os.path.join(res_dir, str(i)+file_suffix1))
        decided = decided + joblib.load(os.path.join(res_dir, str(i)+file_suffix2))
    acc_res = np.zeros((len(process), 2, 4))
    for i in range(len(process)):
        acc_res[i, :, :] = process[i][decided[i], :, :]
    np.save(os.path.join(res_dir, 'final_res'), acc_res)
    return process, decided, acc_res


def mp_traverse_trials(sta_ids: list, save_dir: str, n_threads=20):
    delta = len(sta_ids)//n_threads
    residual = (len(sta_ids)) % n_threads
    proc_pool = []
    start = 0
    time_start = time.time()
    for id_thread in range(1, n_threads+1):
        if residual != 0:
            end = start + delta + 1
            residual -= 1
        else:
            end = start + delta
        proc = Process(target=traverse_all_stats,
                       args=(sta_ids[start:end], save_dir, id_thread, 20))
        start = end
        proc.start()
        proc_pool.append(proc)
    for p in proc_pool:
        p.join()
    time_end = time.time()
    print(f'Running time: {(time_end - time_start)/60} mins')
    return join_res(n_threads, save_dir)


def sp_save_forecasts(sta_ids: list, save_dir: str, thread_id=0):
    for sta_id in sta_ids:
        so = SingleOperator(sta_id, thread_id)
        so.save_forecasts(save_dir)


def mp_save_forecasts(sta_ids: list, save_dir: str, n_threads=20):
    delta = len(sta_ids)//n_threads
    residual = (len(sta_ids)) % n_threads
    proc_pool = []
    start = 0
    time_start = time.time()
    for id_thread in range(1, n_threads+1):
        if residual != 0:
            end = start + delta + 1
            residual -= 1
        else:
            end = start + delta
        proc = Process(target=sp_save_forecasts,
                       args=(sta_ids[start:end], save_dir, id_thread))
        start = end
        proc.start()
        proc_pool.append(proc)
    for p in proc_pool:
        p.join()
    time_end = time.time()
    print(f'Running time: {(time_end - time_start)/60} mins')


if __name__ == '__main__':
    subreg_id = 7
    # so = SingleOperator(subreg_id)
    # so.train()
    trial = Trials(0, n_trials=20)
    trial.start(trans=True, debug=True)
    # trial.res[trial.decide()]
