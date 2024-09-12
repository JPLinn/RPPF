import os
import joblib
import matplotlib

import numpy as np
import pandas as pd
import minepy as mp

import matplotlib.pyplot as plt
import networkx as nx

from geopy.distance import geodesic
from math import sin, cos, atan2, pi
from sklearn.feature_selection import mutual_info_regression

from scipy.stats import entropy

matplotlib.use('TKAgg')


def cal_pcc(stats: pd.DataFrame):
    data_dir = os.path.join('..', 'data', 'single_station_experiment')
    st_ids = stats.index.to_list()
    st_ids.sort()
    st_num = len(st_ids)
    for st_count, st_id in enumerate(st_ids):
        if st_count == 0:
            all_power = np.load(os.path.join(data_dir, str(st_id).rjust(4, '0')+'_train_power.npy')).reshape(-1, 1)
        else:
            tmp_power = np.load(os.path.join(data_dir, str(st_id).rjust(4, '0')+'_train_power.npy')).reshape(-1, 1)
            all_power = np.append(all_power, tmp_power, axis=1)
        print(f"{st_count + 1} station involved")
    pcc = pd.DataFrame(all_power).corr(method='pearson').to_numpy()
    np.save(os.path.join('info', 'global_graph', 'task0', 'pcc'), pcc)
    return pcc


def cal_pcc_by_group(stats: pd.DataFrame, timeline: pd.DatetimeIndex):
    data_dir = os.path.join('..', 'data', 'single_station_experiment')
    st_ids = stats.index.to_list()
    st_ids.sort()
    st_num = len(st_ids)
    pcc = np.zeros((st_num, st_num))
    for st_count, st_id in enumerate(st_ids):
        if st_count == 0:
            all_power = np.load(os.path.join(data_dir, str(st_id).rjust(4, '0')+'_train_power.npy')).reshape(-1, 1)
        else:
            tmp_power = np.load(os.path.join(data_dir, str(st_id).rjust(4, '0')+'_train_power.npy')).reshape(-1, 1)
            all_power = np.append(all_power, tmp_power, axis=1)
        print(f"{st_count + 1} station involved")
    power_df = pd.DataFrame(all_power, index=timeline, columns=st_ids)
    power_df['mark'] = ((power_df.index.month % 12) // 3) * 24 + power_df.index.hour
    gs = power_df.groupby(by='mark').agg(lambda x: x.to_list())  # samples by group

    group_num = gs.shape[0]
    for _, row in gs.iterrows():
        samples = np.array(row.to_list())
        tmp_pcc = np.corrcoef(samples)
        if True in np.isnan(tmp_pcc):
            group_num -= 1
        else:
            pcc = pcc + tmp_pcc
    pcc = pcc / group_num
    np.save(os.path.join('info', 'global_graph', 'task0', 'pcc_bg'), pcc)
    return pcc


def cal_mi_stats(stats: pd.DataFrame):
    data_dir = os.path.join('..', 'data', 'single_station_experiment')
    st_ids = stats.index.to_list()
    st_ids.sort()
    st_num = len(st_ids)
    mi = np.zeros((st_num, st_num))
    for st_count, st_id in enumerate(st_ids):
        if st_count == 0:
            all_power = np.load(os.path.join(data_dir, str(st_id).rjust(4, '0')+'_train_power.npy')).reshape(-1, 1)
        else:
            tmp_power = np.load(os.path.join(data_dir, str(st_id).rjust(4, '0')+'_train_power.npy')).reshape(-1, 1)
            all_power = np.append(all_power, tmp_power, axis=1)
        print(f"{st_count + 1} station involved")
    for count in range(st_num-1):
        if count == st_num - 2:
            mi[count, count+1] = mutual_info_regression(all_power[:, count+1].reshape(-1, 1), all_power[:, count],
                                                        n_neighbors=7)
        else:
            mi[count, count+1:] = mutual_info_regression(all_power[:, count+1:], all_power[:, count], n_neighbors=7)
        print(f"{count + 1}-th row calculated")
    mi = mi + mi.T
    return mi


def _cal_mi(samp_list: list, mine: mp.MINE):
    var_len = len(samp_list)
    samples = np.array(samp_list)
    cmp_nmi, _ = mp.pstats(samples, est='mic_e')
    nmi = np.zeros((var_len, var_len))
    # for i in range(var_len):
    #     for j in range(var_len):
    #         mine.compute_score(samples[i, :], samples[j, :])
    #         nmi[i, j] = mine.mic()
    #         nmi[j, i] = nmi[i, j]
    start = 0
    for i in range(var_len-1):
        nmi[i, i+1:] = cmp_nmi[start:(start+var_len-i-1)]
        start = start + var_len - i - 1
    print(start)
    nmi = nmi + nmi.T
    return nmi


def cal_mi_by_group(stats: pd.DataFrame, timeline: pd.DatetimeIndex):
    data_dir = os.path.join('..', 'data', 'single_station')
    st_ids = stats.index.to_list()
    st_ids.sort()
    st_num = len(st_ids)
    nmi = np.zeros((st_num, st_num))
    for st_count, st_id in enumerate(st_ids):
        if st_count == 0:
            all_power = np.load(os.path.join(data_dir, str(st_id).rjust(4, '0'), 'train_power.npy')).reshape(-1, 1)
        else:
            tmp_power = np.load(os.path.join(data_dir, str(st_id).rjust(4, '0'), 'train_power.npy')).reshape(-1, 1)
            all_power = np.append(all_power, tmp_power, axis=1)
        print(f"{st_count + 1} station involved")
    power_df = pd.DataFrame(all_power, index=timeline.loc[timeline].index, columns=st_ids)
    power_df['mark'] = ((power_df.index.month % 12) // 3) * 24 + power_df.index.hour
    gs = power_df.groupby(by='mark').agg(lambda x: x.to_list())  # samples by group

    mine = mp.MINE()
    for i, row_i in enumerate(gs.iterrows()):
        nmi = nmi + _cal_mi(row_i[1].to_list(), mine)
        print(f'The {i+1}-st row has been dealed with')
    nmi = nmi / gs.shape[0]

    return nmi


# this calculates mi_score between a station and the cluster it belongs to.
def cal_mi_by_group_for_cluster(cluster_stats: pd.DataFrame, timeline: pd.DatetimeIndex):
    data_dir = os.path.join('..', 'data', 'single_station')
    st_ids = cluster_stats.index.to_list()
    st_ids.sort()
    st_num = len(st_ids)
    nmi = np.zeros((st_num, 1))
    for st_count, st_id in enumerate(st_ids):
        if st_count == 0:
            all_power = np.load(os.path.join(data_dir, str(st_id).rjust(4, '0'), 'train_power.npy')).reshape(-1, 1)
        else:
            tmp_power = np.load(os.path.join(data_dir, str(st_id).rjust(4, '0'), 'train_power.npy')).reshape(-1, 1)
            all_power = np.append(all_power, tmp_power, axis=1)
        print(f"{st_count + 1} station involved")
    power_df = pd.DataFrame(all_power, index=timeline, columns=st_ids)
    power_df['total'] = (power_df * cluster_stats['total_cap']).sum(axis=1) / cluster_stats['total_cap'].sum()
    power_df['mark'] = ((power_df.index.month % 12) // 3) * 24 + power_df.index.hour
    gs = power_df.groupby(by='mark').agg(lambda x: x.to_list())  # samples by group

    for i, row_i in enumerate(gs.iterrows()):
        x = np.array(row_i[1][:-1].to_list())
        y = np.array(row_i[1]['total']).reshape(1, -1)
        nmi = nmi + mp.cstats(x, y)[0]
        print(f'The {i+1}-st row has been dealed with')
    nmi = nmi / gs.shape[0]

    return nmi


if __name__ == '__main__':
    tr_ind, _, _ = joblib.load(os.path.join('..', 'data', 'new_time_index_for_sets.jl'))
    mcs = joblib.load(os.path.join('..', 'graph', 'merged_close_stats.jl'))
    mcs = mcs.sort_index()
    # partition = np.load(os.path.join('info', 'global_graph', 'task0', 'partition_midist_newman', 'partition.npy')

    mi = cal_mi_by_group(mcs, tr_ind)
    np.save(os.path.join('info', 'mi'), mi)
    # cal_pcc(mcs)
    # cal_pcc_by_group(mcs, tl)

    # mi = cal_mi_by_group_for_cluster(mcs.iloc[np.where(partition[0])], tl)