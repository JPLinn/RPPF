import math
import os
import json
import joblib
import torch
import logging

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from tqdm import tqdm


class Params:
    """
    Loads hyper-parameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """Saves parameters into json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by params.dict['learning_rate']"""
        return self.__dict__


class SampleSet(Dataset):
    def __init__(self, data_dir, usage, st_id: None):
        if st_id is None:
            self.nwp = np.load(os.path.join(data_dir, f'{usage}_nwp.npy'))
            self.label = np.load(os.path.join(data_dir, f'{usage}_power.npy'))
            self.temp = np.load(os.path.join('data', f'{usage}_time.npy'))
        else:
            self.nwp = np.load(os.path.join(data_dir, 'nwp', str(st_id).rjust(4, '0') + '_' + f'{usage}_nwp.npy'))
            self.label = np.load(os.path.join(data_dir, str(st_id).rjust(4, '0') + '_' + f'{usage}_power.npy'))
            self.temp = np.load(os.path.join('data', f'{usage}_time.npy'))
        self.length = self.label.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.nwp[index], self.temp[index], self.label[index]


class GraphSampleSet(Dataset):
    def __init__(self, power_data_dir, usage='train'):
        self.nwp = np.load(os.path.join(power_data_dir, f'{usage}_nwp.npy'))
        self.label = np.load(os.path.join(power_data_dir, f'{usage}_power.npy'))
        self.temp = np.load(os.path.join('data', f'{usage}_temp2.npy'))
        self.length = self.label.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.nwp[index], self.temp[index], self.label[index]


def calc_sunrise(date_of_year: int, latitude: float):
    c1 = 0.98565 * math.pi / 180
    c2 = 1.914 * math.pi / 180
    sun_decl = -math.asin(0.39779 * math.cos((date_of_year + 9) * c1 + c2 * math.sin(c1 * (date_of_year - 3))))
    ha_sunrise = math.acos(-math.tan(sun_decl) * math.tan(latitude * math.pi / 180))
    interval = ha_sunrise * 180 / math.pi / 15
    sunrise = 12. - interval
    sunset = 12. + interval
    return sunrise, sunset


def plot_in_train(data, save_name, dir='./figures/'):
    if not os.path.exists(dir):
        os.mkdir(dir)
    length = data.shape[0]
    start = 1 if length - 10 < 0 else length - 9
    x_tick = np.arange(start=start, stop=length + 1)
    f = plt.figure()
    plt.plot(x_tick, data[start-1:length])
    f.savefig(os.path.join(dir, save_name + '.png'))
    plt.close()


def set_logger(log_path):
    '''Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    '''
    _logger = logging.getLogger('RPPF')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    # _logger.addHandler(TqdmHandler(fmt))


class Evaluator(object):
    def __init__(self, logger, q=5):
        self.logger = logger
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
        self.crps = 0.
        self.qs = [0.] * len(self.q)
        self.coverage = [0.] * len(self.q)
        self.ivt_width = [0.] * self.itv_num
        self.qcs = 0.
        self.calibrator = [0.] * len(self.q)

    def qs_eval(self, pred: torch.Tensor, label: torch.Tensor):
        # pred is (batch_size, num_qs), label is (batch_size)
        ind = pred >= label.unsqueeze(dim=1)
        q_score = torch.matmul((label.unsqueeze(dim=1) - pred), torch.diag(torch.tensor(self.q, device=label.device))) \
                  + ind * (pred - label.unsqueeze(dim=1))
        q_score = q_score.sum(dim=0).tolist()
        for count in range(len(self.q)):
            self.qs[count] += q_score[count]

    def crps_eval(self, pred: torch.Tensor, label: torch.Tensor):
        crps = torch.abs((pred - label.unsqueeze(dim=1))).sum() - \
               (torch.diff(pred) * torch.tensor(range(1, len(self.q), 1), device=pred.device) *
                torch.tensor(range(len(self.q)-1, 0, -1), device=pred.device)).sum()/pred.shape[1]
        self.crps += crps.tolist()

    def rel_eval(self, pred: torch.Tensor, label: torch.Tensor):
        # pred is (batch_size, num_qs), label is (batch_size)
        ind = pred.T > label
        ind_sum = ind.sum(axis=1)
        for count in range(len(self.q)):
            self.coverage[count] += int(ind_sum[count])

    def sha_eval(self, pred: torch.Tensor):
        for i in range(self.itv_num):
            self.ivt_width[i] = self.ivt_width[i] + float((pred[:, -(i+1)] - pred[:, i]).sum())

    def batch_eval(self, pred: torch.Tensor, label: torch.Tensor, usage='train'):
        self.size = self.size + label.numel()
        self.agg = self.agg + float(label.sum())
        if usage == 'test':
            pred = self.calibrate(pred)
        self.check_qcs(pred)
        pred = pred.sort().values
        self.qs_eval(pred, label)
        self.rel_eval(pred, label)
        self.sha_eval(pred)
        self.crps_eval(pred, label)

    def check_qcs(self, pred: torch.Tensor):
        self.qcs = self.qcs + ((pred.diff().sum(dim=1) + 0.001) / (pred.diff().abs().sum(dim=1) + 0.001)).sum()
        # self.qcs = \
        #     self.qcs + float(((ind - torch.tensor(range(len(self.q)), device=pred.device))**2).sum(dim=1).sqrt().sum())

    def sum_up(self, usage='train'):
        self.qs = [float('{:.4f}'.format(x/self.size)) for x in self.qs]
        if usage == 'train':
            self.calibrator = [self.coverage[i]/self.size for i in range(len(self.q))]
        self.coverage = [float('{:.4f}'.format(self.coverage[i]/self.size-self.q[i])) for i in range(len(self.q))]
        self.ivt_width = [float('{:.4f}'.format(x/self.agg)) for x in self.ivt_width]
        ivt_norm = [(self.itv_num-i)/(self.itv_num+1) for i in range(self.itv_num)]
        self.naps = \
            sum([self.ivt_width[i]/ivt_norm[i] for i in range(len(ivt_norm))])/len(ivt_norm) if self.itv_num != 0 else 0.
        self.mre = max([abs(x) for x in self.coverage])
        self.mqs = sum(self.qs)/len(self.qs)
        self.qcs = (1. - self.qcs/self.size) / 2.
        self.crps = self.crps / self.size / len(self.q)
        return self.report(usage=usage)

    def report(self, usage='train'):
        # self.logger.info(f"#### the {usage} set ####")
        self.logger.info(f"*** Quantiles: {self.q} ***")
        self.logger.info(f'Size: {self.size}')
        self.logger.info(f'Q Score: {self.qs}')
        self.logger.info('Mean score: {:.4f}'.format(self.mqs))
        self.logger.info(f'Reliability Error: {self.coverage}')
        self.logger.info('MRE: {:.4f}'.format(self.mre))
        self.logger.info(f'PINAW: {self.ivt_width}')
        self.logger.info('NAPS: {:.4f}'.format(self.naps))
        self.logger.info('CRPS: {:.4f}'.format(self.crps))
        self.logger.info('QCS: {:.4f}'.format(self.qcs))
        return np.array([float('{:.4f}'.format(self.mqs)), float('{:.4f}'.format(self.mre)),
                         float('{:.4f}'.format(self.naps)), float('{:.4f}'.format(self.crps)),
                         float('{:.4f}'.format(self.qcs))])

    def clear(self):
        self.size = 0
        self.agg = 0.
        self.naps = 0.
        self.mre = 0.
        self.mqs = 0.
        self.crps = 0.
        self.qs = [0.] * len(self.q)
        self.coverage = [0.] * len(self.q)
        self.ivt_width = [0.] * self.itv_num
        self.qcs = 0.

    def calibrate(self, pred: torch.Tensor):
        # new = (new_level - old_level) * k + b
        calib = torch.tensor(self.calibrator, device=pred.device)
        k = pred.diff() / calib.diff()
        k = torch.cat((k, k[:, -1].unsqueeze(dim=1)), dim=1)

        level_diff = torch.zeros(len(self.q), device=pred.device)
        level_diff[:-1] = torch.tensor(self.q, device=pred.device)[:-1] - calib[1:]
        level_diff[-1] = self.q[-1] - calib[-2]

        b = torch.zeros_like(pred)
        b[:, :-1] = pred[:, 1:]
        b[:, -1] = pred[:, -2]

        new_pred = k * level_diff + b
        return new_pred


if __name__ == '__main__':
    # print(calc_sunrise(112, 33.43))

    power_dir = os.path.join('data', 'task_0')
    nwp_dir = os.path.join('data', 'single_station_experiment', 'nwp')
    train_set = GraphSampleSet(power_dir, nwp_dir)