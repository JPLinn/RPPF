import os
import joblib
import numpy
import matplotlib

import numpy as np
import pandas as pd
import igraph as ig

import matplotlib.pyplot as plt
import networkx as nx

from geopy.distance import geodesic
from math import sin, cos, atan2, pi
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy

matplotlib.use('TKAgg')


# calculate the x-directional (from west to east) and y-directional (from south to north) distance of two points
def cal_xy_dist(lat1, lon1, lat2, lon2):
    earthR = 6371.393
    lat1_rad = lat1 * pi / 180
    lon1_rad = lon1 * pi / 180
    lat2_rad = lat2 * pi / 180
    lon2_rad = lon2 * pi / 180
    x = earthR * cos((lat1_rad+lat2_rad)/2) * (lon2_rad - lon1_rad)
    y = earthR * (lat2_rad - lat1_rad)
    return complex(x, y)


# hereafter complex graph
def make_comp_dist_mat(stats: pd.DataFrame):
    stats.sort_index(inplace=True)
    stats_num = stats.shape[0]
    dist = np.mat(np.zeros([stats_num, stats_num]), dtype=complex)
    for st_count in range(stats_num-1):
        lat, lon = stats.iloc[st_count][['lat', 'lon']]
        tmp_dist = stats.iloc[(st_count+1):, :].apply(lambda x: cal_xy_dist(lat, lon, x.lat, x.lon), axis=1).tolist()
        dist[st_count, (st_count+1):] = tmp_dist
    dist = dist + dist.T.conj()
    return dist


# calculate the distance between two stations
def cal_stat_dists(stats: pd.DataFrame):
    stats_ = stats.sort_index()
    dists = np.zeros((len(stats), len(stats)))
    count = 0
    for _, stat in stats_.iterrows():
        dists[count, :] = stats_.apply(lambda x: geodesic((x.lat, x.lon), (stat.lat, stat.lon)).km, axis=1).values
        count = count + 1
    return dists


def cal_cnc_subgraph(bool_ajc: np.ndarray):
    pass
    subgraphs = {}
    old_inds = []
    for count, row in enumerate(bool_ajc):
        if count == len(bool_ajc) - 1:
            break
        elif count in old_inds:
            continue
        old_inds.append(count)
        start = count + 1
        inds = np.where(row[start:])[0] + start
        inds = inds.tolist()
        if len(inds) == 0:
            continue
        tmp_subgrh = [set([count] + inds), set([(count, x) for x in inds])]
        while len(inds) != 0:
            cur_index = inds.pop(0)
            if cur_index == len(bool_ajc) - 1:
                continue
            old_inds.append(cur_index)
            cur_start = cur_index + 1
            cur_inds = (np.where(row[cur_start:])[0] + cur_start).tolist()
            if len(cur_inds) != 0:
                inds = inds + cur_inds
                inds = list(set(inds))
                inds.sort()
                tmp_subgrh[0] = tmp_subgrh[0].union(set(cur_inds))
                tmp_subgrh[1] = tmp_subgrh[1].union(set([(cur_index, x) for x in cur_inds]))
        subgraphs[count] = tmp_subgrh
    return subgraphs


def merge_close_stats(subgraphs: dict, stats: pd.DataFrame):
    merged_stats = stats.loc[:, ['lat', 'lon', 'total_cap']].sort_index()
    merged_stats['stats'] = merged_stats.index
    merged_stats['stats'] = merged_stats['stats'].apply(lambda x: [int(x)])
    key_list = []
    value_list = []
    old_stat_ids = []
    merged_stats_info = {}
    for key, value in subgraphs.items():
        key_list.append(key)
        merged_stats_info[merged_stats.index[key]] = merged_stats.index[list(value[0])].tolist()
        cur_value = merged_stats.iloc[list(value[0]), :2].mean().to_list() + \
                    [merged_stats.iloc[list(value[0]), 2].sum()] + [merged_stats.index[list(value[0])].tolist()]
        value_list.append(cur_value)
        old_stat_ids = old_stat_ids + list(value[0])
    old_stat_ids = list(set(old_stat_ids))
    new_stats = pd.DataFrame(value_list, index=list(merged_stats_info.keys()),
                             columns=['lat', 'lon', 'total_cap', 'stats'])
    merged_stats.drop(merged_stats.index[old_stat_ids], inplace=True)
    merged_stats = pd.concat((merged_stats, new_stats)).sort_index()
    joblib.dump(merged_stats_info, 'merged_close_stats_info.jl')
    joblib.dump(merged_stats, 'merged_close_stats.jl')

    return merged_stats


# hereafter some one-time functions
def merge_cs(stat_dists: np.ndarray, stats: pd.DataFrame, gap=2):
    bool_dists = stat_dists < gap
    sub_graphs = cal_cnc_subgraph(bool_dists)
    return merge_close_stats(sub_graphs, stats)


if __name__ == '__main__':
    sts = joblib.load('merged_stats.jl')
    stat_dists = cal_stat_dists(sts)
    merged_stats = merge_cs(stat_dists, sts)
    stat_dists = cal_stat_dists(merged_stats)
    np.save(os.path.join('info', 'stat_dists'), stat_dists)
    np.save(os.path.join('info', 'stat_loc'), merged_stats.loc[:, ['lat', 'lon']].to_numpy())

    xy_dist = make_comp_dist_mat(merged_stats)
    np.save(os.path.join('info', 'stat_xy_comp_dist'), xy_dist)
    x_pos = np.real(xy_dist[0, :]).tolist()[0]
    y_pos = np.imag(xy_dist[0, :]).tolist()[0]
    pos = np.array(list(zip(x_pos, y_pos)))
    np.save(os.path.join('info', 'stat_pos'), pos)