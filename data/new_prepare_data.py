import os
import joblib
import datetime
import random

import numpy as np
import pandas as pd

from utils import calc_sunrise
from math import sin, cos, pi
from sklearn.preprocessing import StandardScaler


def make_time_validity_table(latitude: float, start_date: datetime.date, end_date: datetime.date):
    time_mark = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='1H', closed='left'), columns=['time'])
    record = pd.DataFrame(pd.date_range(start=start_date, end=end_date, freq='1D', closed='left'), columns=['date'])
    record['date'] = record['date'].apply(lambda x: x.date())
    record['sun_time'] = record['date'].apply(lambda x: calc_sunrise(pd.to_datetime(x).dayofyear, latitude))
    record.set_index('date', inplace=True)
    time_mark['mark'] = time_mark['time'].apply(
        lambda x: record.loc[x.date(), 'sun_time'][0] <= x.hour <= record.loc[x.date(), 'sun_time'][1])
    return time_mark.set_index('time')['mark']


def extract_local_grid(sts: list, gps: dict, save_dir: str):
    if len(sts) == 0:
        sel_sts = list(gps.keys())
    else:
        sel_sts = sts
    sel_sts.sort()
    local_grid = []
    for st in sel_sts:
        local_grid = local_grid + [len(gps[st][0])]
    joblib.dump(local_grid, os.path.join(save_dir, 'local_grid.jl'))


def make_index_for_sets(time_mark, error_days, save_dir):
    all_time = time_mark & (~pd.Series(time_mark.index.date, index=time_mark.index).isin(error_days))
    at_index = all_time.index
    test_set_index = all_time.copy()
    test_start = pd.to_datetime('2021-08-01')
    test_set_index[at_index<test_start] = False
    train_set_index = all_time.copy()
    train_set_index[at_index>=test_start] = False

    train_set_tmp_index = train_set_index[train_set_index]
    train_set_tmp_index = pd.DataFrame(train_set_tmp_index.values, columns=['value'], index=train_set_tmp_index.index)
    by_date = train_set_tmp_index.groupby(by=train_set_tmp_index.index.date).agg(sum)
    by_date.index = pd.to_datetime(by_date.index)
    by_date['year'] = by_date.index.year
    by_date['month'] = by_date.index.month
    by_date['ym'] = by_date.apply(lambda x: pd.to_datetime(str(x.year) + '-' + str(x.month)), axis=1)
    by_date['day'] = by_date.index.day
    by_date['count'] = 1
    month_count = by_date.groupby(by='ym').agg(lambda x: x.to_list())

    vali_set_index = all_time.copy()
    vali_set_index[:] = False
    for index, row in month_count.iterrows():
        days = row.day
        days.sort()
        start_pos = random.sample(range(len(days)-3), 1)[0]
        sampled_days = days[start_pos:(start_pos+4)]
        # sampled_days = random.sample(row.day, 4)
        tmp_index = all_time & (at_index.year == index.year) & (at_index.month == index.month)
        for sampled_day in sampled_days:
            vali_set_index[tmp_index & (at_index.day == sampled_day)] = True
            train_set_index[tmp_index & (at_index.day == sampled_day)] = False

    joblib.dump((train_set_index, vali_set_index, test_set_index), os.path.join(save_dir, 'new_time_index_for_sets.jl'))
    return train_set_index, vali_set_index, test_set_index


def prepare_temporal_feature(ind_for_sets, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ind_tn, ind_vd, ind_tt = ind_for_sets
    temp_features = pd.Series(ind_tn.index, index=ind_tn.index, name='time').to_frame()
    temp_features['day'] = temp_features['time'].apply(lambda x: x.dayofyear) / 365.
    temp_features['hour'] = temp_features['time'].apply(lambda x: x.hour) / 24.
    temp_features['dcos'] = temp_features['day'].apply(lambda x: cos(2*pi*x))
    temp_features['dsin'] = temp_features['day'].apply(lambda x: sin(2 * pi * x))
    temp_features['hcos'] = temp_features['hour'].apply(lambda x: cos(2*pi*x))
    temp_features['hsin'] = temp_features['hour'].apply(lambda x: sin(2 * pi * x))

    np.save(os.path.join(save_dir, 'train' + '_' + 'temp1' + '.npy'), temp_features[ind_tn][['day', 'hour']].to_numpy())
    np.save(os.path.join(save_dir, 'vali' + '_' + 'temp1' + '.npy'), temp_features[ind_vd][['day', 'hour']].to_numpy())
    np.save(os.path.join(save_dir, 'test' + '_' + 'temp1' + '.npy'), temp_features[ind_tt][['day', 'hour']].to_numpy())

    np.save(os.path.join(save_dir, 'train' + '_' + 'temp2' + '.npy'),
            temp_features[ind_tn][['dcos', 'dsin', 'hcos', 'hsin']].to_numpy())
    np.save(os.path.join(save_dir, 'vali' + '_' + 'temp2' + '.npy'),
            temp_features[ind_vd][['dcos', 'dsin', 'hcos', 'hsin']].to_numpy())
    np.save(os.path.join(save_dir, 'test' + '_' + 'temp2' + '.npy'),
            temp_features[ind_tt][['dcos', 'dsin', 'hcos', 'hsin']].to_numpy())

    return temp_features


# pack up nwp data for one given station
def prepare_nwp_for_one_st(gps: list, features: list, ind_for_sets, save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ind_tn, ind_vd, ind_tt = ind_for_sets
    for count, point in enumerate(gps):
        tmp_nwp = joblib.load(os.path.join('../raw_data', 'new_nwp', str(point).rjust(4, '0') + '.jl'))
        tmp_nwp = tmp_nwp.loc[:, '19:00'][features]
        tmp_nwp_tn = tmp_nwp[ind_tn]
        tmp_nwp_vd = tmp_nwp[ind_vd]
        tmp_nwp_tt = tmp_nwp[ind_tt]
        if count == 0:
            nwp_tn = tmp_nwp_tn.values.reshape([*tmp_nwp_tn.shape, 1])
            nwp_vd = tmp_nwp_vd.values.reshape([*tmp_nwp_vd.shape, 1])
            nwp_tt = tmp_nwp_tt.values.reshape([*tmp_nwp_tt.shape, 1])
        else:
            nwp_tn = np.concatenate((nwp_tn, tmp_nwp_tn.values.reshape([*tmp_nwp_tn.shape, 1])), axis=2)
            nwp_vd = np.concatenate((nwp_vd, tmp_nwp_vd.values.reshape([*tmp_nwp_vd.shape, 1])), axis=2)
            nwp_tt = np.concatenate((nwp_tt, tmp_nwp_tt.values.reshape([*tmp_nwp_tt.shape, 1])), axis=2)

    nwp_tn = nwp_tn.transpose(0, 2, 1)
    nwp_vd = nwp_vd.transpose(0, 2, 1)
    nwp_tt = nwp_tt.transpose(0, 2, 1)

    np.save(os.path.join(save_dir, 'train' + '_' + 'nwp' + '.npy'), nwp_tn)
    np.save(os.path.join(save_dir, 'vali' + '_' + 'nwp' + '.npy'), nwp_vd)
    np.save(os.path.join(save_dir, 'test' + '_' + 'nwp' + '.npy'), nwp_tt)


# just a shell
def prepare_nwp_for_many_single_st(sts: list, gps: dict, features: list, ind_for_sets, save_dir: str):
    if len(sts) == 0:
        sel_gps = gps
    else:
        sel_gps = {st_id: gp for st_id, gp in gps.items() if st_id in sts}
    for st_id, gp in sel_gps.items():
        true_save_dir = os.path.join(save_dir, str(st_id).rjust(4, '0'))
        prepare_nwp_for_one_st(gp[0], features, ind_for_sets, true_save_dir)


#################################################
# # hereafter wrapper routine
# prepare power data for many single stats
def prepare_single_st_data(mcs: pd.DataFrame, ind_for_sets, save_dir: str):
    count = 0
    tn_id, vd_id, tt_id = ind_for_sets
    for merged_id, merged_info in mcs.iterrows():
        x = pd.DataFrame()
        inner_count = 0
        tmp_dir = os.path.join(save_dir, str(merged_id).rjust(4, '0'))
        for stat_id in merged_info['stats']:
            # if stat_id == 1675:
            #     print('debug')
            if inner_count == 0:
                x = pd.read_csv(os.path.join('../raw_data', 'power', str(stat_id).rjust(4, '0') + '.csv'), index_col=0,
                                parse_dates=True)
            else:
                x = x + pd.read_csv(os.path.join('../raw_data', 'power', str(stat_id).rjust(4, '0') + '.csv'),
                                    index_col=0, parse_dates=True)
            inner_count = inner_count + 1
        norm = x / merged_info['total_cap']
        train_power = norm[tn_id]
        vali_power = norm[vd_id]
        test_power = norm[tt_id]
        train_power = train_power.values.reshape(train_power.size)
        # print(f'Max p.u. value is {train_power.max()} at No.{merged_id}')
        if train_power.max() < 0.65 or train_power.max() > 1.15:
            print(f'Max p.u. value is {train_power.max()} at No.{merged_id}')
        vali_power = vali_power.values.reshape(vali_power.size)
        test_power = test_power.values.reshape(test_power.size)

        np.save(os.path.join(tmp_dir, 'train' + '_' + 'power' + '.npy'), train_power)
        np.save(os.path.join(tmp_dir, 'vali' + '_' + 'power' + '.npy'), vali_power)
        np.save(os.path.join(tmp_dir, 'test' + '_' + 'power' + '.npy'), test_power)
        count = count + 1
        print(f'Sta {merged_id} ended')
    print(f'{count} stations are processed')


# cal the aggregate power of selected stations and pack it up.
def prepare_agg_power(merged_close_stats: pd.DataFrame, ind_for_sets, save_dir: str):
    tmp_power = pd.DataFrame()
    count = 0
    cap = 0
    tn_id, vd_id, tt_id = ind_for_sets
    for merged_id, merged_info in merged_close_stats.iterrows():
        for stat_id in merged_info['stats']:
            if count == 0:
                tmp_power = pd.read_csv(os.path.join('../raw_data', 'power', str(stat_id).rjust(4, '0') + '.csv'),
                                      index_col=0, parse_dates=True)
            else:
                tmp_power = tmp_power + pd.read_csv(
                    os.path.join('../raw_data', 'power', str(stat_id).rjust(4, '0') + '.csv'), index_col=0,
                    parse_dates=True)
            count = count + 1
        cap = cap + merged_info['total_cap']
    norm = tmp_power / cap
    train_power = norm[tn_id]
    vali_power = norm[vd_id]
    test_power = norm[tt_id]
    train_power = train_power.values.reshape(-1)
    vali_power = vali_power.values.reshape(-1)
    test_power = test_power.values.reshape(-1)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'train' + '_' + 'power' + '.npy'), train_power)
    np.save(os.path.join(save_dir, 'vali' + '_' + 'power' + '.npy'), vali_power)
    np.save(os.path.join(save_dir, 'test' + '_' + 'power' + '.npy'), test_power)


# prepare nwp data for the local stage of the GNN method
def prepare_local_nwp(sts: list, extract_dir: str, save_dir: str):
    dirs = os.listdir(extract_dir)
    if len(sts) != 0:
        dirs = [dir_i for dir_i in dirs if int(dir_i) in sts]
    dirs.sort()
    nwp_count = 0
    for dir_i in dirs:
        if nwp_count == 0:
            tn_nwp = np.load(os.path.join('single_station', dir_i, 'train_nwp.npy'))
            vd_nwp = np.load(os.path.join('single_station', dir_i, 'vali_nwp.npy'))
            tt_nwp = np.load(os.path.join('single_station', dir_i, 'test_nwp.npy'))
            # self.nwp = np.expand_dims(self.nwp, axis=-1)
        else:
            tn_nwp = np.append(tn_nwp, np.load(os.path.join('single_station', dir_i, 'train_nwp.npy')), axis=2)
            vd_nwp = np.append(vd_nwp, np.load(os.path.join('single_station', dir_i, 'vali_nwp.npy')), axis=2)
            tt_nwp = np.append(tt_nwp, np.load(os.path.join('single_station', dir_i, 'test_nwp.npy')), axis=2)
        nwp_count = nwp_count + 1
        if nwp_count == len(dirs):
            np.save(os.path.join(save_dir, 'train_nwp.npy'), tn_nwp)
            np.save(os.path.join(save_dir, 'vali_nwp.npy'), vd_nwp)
            np.save(os.path.join(save_dir, 'test_nwp.npy'), tt_nwp)


# prepare edge feature data for dynamic edge-conditioned filter
def prepare_edge_feature(adj: np.ndarray, stats: pd.DataFrame, xy_dist: np.ndarray, save_dir: str):
    stats.sort_index(inplace=True)
    cap = stats['total_cap'].to_numpy()
    node_num = adj.shape[0]
    adj_ = adj.copy()
    adj_[range(node_num), range(node_num)] = True
    edge_num = adj_.sum()
    ef = np.zeros((6, edge_num))  # 6 features in all
    ef[0, :] = np.real(xy_dist[adj_])  # x-dist feature
    ef[1, :] = np.imag(xy_dist[adj_])  # y-dist feature
    ef[2, :] = np.sqrt(ef[0, :]**2 + ef[1, :]**2)  # distance
    ef[3, :] = ef[0, :] / (ef[2, :] + 0.01)  # cos feature
    ef[4, :] = ef[1, :] / (ef[2, :] + 0.01)  # sin feature
    col_start = 0
    for stat_i in range(node_num):
        edge_i_num = adj_[stat_i, :].sum()
        ef[5, col_start:(col_start+edge_i_num)] = cap[adj_[stat_i, :]] / cap[stat_i]
        col_start += edge_i_num
    scaler = StandardScaler()
    scaler.fit(ef.T)
    ef = scaler.transform(ef.T).T
    np.save(os.path.join(save_dir, 'edge_feature'), ef)
    print(ef.shape)
    return ef


# prepare edge feature data for dynamic edge-conditioned filter and for graph partition
def prepare_edge_feature_for_partition(adj: np.ndarray, stats: pd.DataFrame, xy_dist: np.ndarray, partition: np.ndarray,
                                       save_dir: str):
    stats.sort_index(inplace=True)
    node_num = adj.shape[0]
    partition = partition.astype(np.int_)
    adj_ = adj.copy()
    adj_[range(node_num), range(node_num)] = True
    ef_list = []
    scaler = StandardScaler()
    for part_i, part in enumerate(partition):
        part = np.array(part, dtype=bool)
        edge_num = adj_[part, :].sum()
        ef = np.zeros((6, edge_num))  # 6 features in all
        ef[0, :] = np.real(xy_dist[part, :][adj_[part, :]])  # x-dist feature
        ef[1, :] = np.imag(xy_dist[part, :][adj_[part, :]])  # y-dist feature
        ef[2, :] = np.sqrt(ef[0, :]**2 + ef[1, :]**2)  # distance
        ef[3, :] = ef[0, :] / (ef[2, :] + 0.01)  # cos feature
        ef[4, :] = ef[1, :] / (ef[2, :] + 0.01)  # sin feature
        scaler.fit(ef.T)
        ef = scaler.transform(ef.T).T
        ef_list.append(ef)
    joblib.dump(ef_list, os.path.join(save_dir, 'edge_feature_for_partition.jl'))
    return ef_list


# prepare edge features for graph with each nodes having edges of equal number.
def prepare_eq_ef(adj: np.ndarray, stats: pd.DataFrame, xy_dist: np.ndarray, save_dir: str):
    stats.sort_index(inplace=True)
    cap = stats['total_cap'].to_numpy()
    node_num = adj.shape[0]
    adj_ = adj.copy()
    adj_[range(node_num), range(node_num)] = True
    edge_num = adj_.sum()
    # (node_num, edge_num_for_each_node+itself, 5)
    ef = np.zeros((node_num, adj_[0, :].sum(), 5))  # 5 features in all
    for i in range(node_num):
        ef[i, :, 0] = np.real(xy_dist[i, :][adj_[i, :]])  # x-dist feature
        ef[i, :, 1] = np.imag(xy_dist[i, :][adj_[i, :]])  # y-dist feature
    ef[i, :, 2] = np.sqrt(ef[i, :, 0]**2 + ef[i, :, 1]**2)  # distance
    ef[i, :, 3] = ef[i, :, 0] / (ef[i, :, 2] + 0.01)  # cos feature
    ef[i, :, 4] = ef[i, :, 1] / (ef[i, :, 2] + 0.01)  # sin feature
    col_start = 0
    # for stat_i in range(node_num):
    #     edge_i_num = adj_[stat_i, :].sum()
    #     ef[5, col_start:(col_start+edge_i_num)] = cap[adj_[stat_i, :]] / cap[stat_i]
    #     col_start += edge_i_num
    scaler = StandardScaler()
    scaler.fit(ef.reshape(-1, 5))
    ef = scaler.transform(ef.reshape(-1, 5)).reshape(node_num, adj_[0, :].sum(), 5)
    np.save(os.path.join(save_dir, 'eq_edge_feature'), ef)
    print(ef.shape)
    return ef


if __name__ == '__main__':
    error_days = joblib.load(os.path.join('../raw_data', 'error_days.jl'))
    time_mark = joblib.load(os.path.join('../raw_data', 'time_mark.jl'))
    stats = joblib.load(os.path.join('..', 'graph', 'merged_stats.jl'))  # note these stats have NOT been merged closely
    merged_cs = joblib.load(os.path.join('..', 'graph', 'merged_close_stats.jl'))
    gps_for_st = joblib.load(os.path.join('..', 'graph', 'info', 'gps_for_st.jl'))
    valid_features = ['SC', 'CR', 'T2M', 'RH2M', 'TCC', 'DSWR', 'CS']
    graph_dir = os.path.join('..', 'graph', 'info', 'global_graph', 'task0')

    # tn_ind, vd_ind, tt_ind = make_index_for_sets(time_mark, error_days, '.')
    tn_ind, vd_ind, tt_ind = joblib.load('new_time_index_for_sets.jl')

    prepare_agg_power(merged_cs, (tn_ind, vd_ind, tt_ind), '.')
    # prepare_nwp_for_many_single_st([], gps_for_st, valid_features, (tn_ind, vd_ind, tt_ind), 'single_station')
    # prepare_local_nwp([], 'single_station', '.')
    # prepare_temporal_feature((tn_ind, vd_ind, tt_ind), '.')
    # prepare_single_st_data(merged_cs, (tn_ind, vd_ind, tt_ind), 'single_station')



