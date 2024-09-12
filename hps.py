import os
import sys
import json
import logging
import argparse
import time
import torch
import joblib
import shutil

import multiprocessing_on_dill as multiprocessing
import numpy as np

from copy import copy
from itertools import product
from subprocess import check_call

import utils
from utils import Params, plot_in_train, SampleSet, Evaluator
from tt_logics import Trials


def launch_training_job(params):
    trial = Trials(params)
    trial.start()
    trial.decide()
    np.save(os.path.join(params.model_dir, 'best_res'), trial.res[trial.decided])
    trial.del_useless()


class HPS(object):
    def __init__(self, search_params):
        self.data_dir = self.data_dir = os.path.join("data", 'task0')
        self.ori_model_dir = os.path.join("models", "hyperparams_search")
        if not os.path.exists(self.ori_model_dir):
            os.makedirs(self.ori_model_dir)
        self.ori_params = Params(os.path.join('models', 'params.json'))
        self.search_params = search_params
        keys = sorted(search_params.keys())
        self.search_grids = list(product(*[[*range(len(search_params[i]))] for i in keys]))
        self.param_list = []
        self._prepare_param_list()
        self.decided = 0
        self.res = np.empty(1)

    def _prepare_param_list(self):
        for i, grid_point in enumerate(self.search_grids):
            params = {k: self.search_params[k][grid_point[idx]] for idx, k in
                      enumerate(sorted(self.search_params.keys()))}
            model_param_list = '-'.join('_'.join((k, f'{v:.2f}')) for k, v in params.items())
            params['model_dir'] = os.path.join(self.ori_model_dir, model_param_list)
            params['pro_id'] = i
            model_params = copy(self.ori_params)
            for k, v in params.items():
                setattr(model_params, k, v)
            self.param_list.append(copy(model_params))

    def start_hps(self, n_procs=6):
        running_times = (len(self.param_list) - 1) // n_procs + 1
        print(f'### The Whole Region: {len(self.param_list)} search points, {n_procs} parallel processes, '
              f'{running_times} running times ###')
        time_start = time.time()
        residual = len(self.param_list) % n_procs
        for id_running_time in range(1, running_times + 1):
            proc_pool = []
            if id_running_time == running_times:
                real_n_procs = n_procs if residual == 0 else residual
            else:
                real_n_procs = n_procs
            for idx_proc in range(real_n_procs):
                count_proc = idx_proc + (id_running_time - 1) * n_procs
                proc = multiprocessing.Process(target=launch_training_job, args=(self.param_list[count_proc],))
                proc.start()
                proc_pool.append(proc)
            for p in proc_pool:
                p.join()
            print(f"*** {id_running_time}-th running is over ***")
        time_end = time.time()
        print(f'$$$ Running time for the Whole Region: {(time_end - time_start) / 60} mins $$$')

    def comp_res(self):
        dirs = os.listdir(self.ori_model_dir)
        dirs.sort()
        for idx, dir_i in enumerate(dirs):
            if idx == 0:
                res = np.load(os.path.join(self.ori_model_dir, dir_i, 'best_res.npy'))
                res = res.reshape(1, *res.shape)
            else:
                tmp_res = np.load(os.path.join(self.ori_model_dir, dir_i, 'best_res.npy'))
                tmp_res = tmp_res.reshape(1, *tmp_res.shape)
                res = np.concatenate((res, tmp_res), axis=0)

        ind = res[:, 0, 1].argsort()[:5]
        decided = ind[res[ind, 0, 0].argsort()][0]
        print(f'#### The Whole Region: The {decided+1}-th model is selected as best ####')
        print(f'{res[decided, 0, :].tolist()}')
        print(f'{res[decided, 1, :].tolist()}')
        self.res = res.copy()
        self.decided = decided
        best_params = Params(os.path.join(self.ori_model_dir, dirs[decided], 'params.json'))
        return best_params, res[decided]

    def del_useless(self):
        dirs = os.listdir(self.ori_model_dir)
        dirs.sort()
        for idx, dir_i in enumerate(dirs):
            if idx != self.decided:
                shutil.rmtree(os.path.join(self.ori_model_dir, dir_i), ignore_errors=True)


if __name__ == '__main__':
    search_params = {
        # 'lstm_dropout': np.arange(0.1, 0.3, 0.05, dtype=np.float32).tolist(),
        # 'lstm_hidden_dim': np.arange(6, 30, 5, dtype=np.int).tolist()
        # 'num_spline': np.arange(53, 80, 3, dtype=int).tolist()
        'alpha': np.arange(0.05, 0.31, 0.1).tolist(),
        'mlp_output_dims': np.arange(40, 100, 20).tolist(),
        # 'mlp_hid_dim2': np.arange(40, 51, 10, dtype=np.int_).tolist()
    }
    # hps = SR_HPS(subreg_id=0, search_params=search_params)
    # hps.start_hps()
