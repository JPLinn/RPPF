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


# determine the connectivity of any two stats
def det_connect_by_midist(mi_mat: np.ndarray, dist_mat: np.ndarray, k1=6, k2=6):
    # thd_mi = np.percentile(mi_mat[mi_mat > 0.], gap)
    # thd_dist = np.percentile(dist_mat[dist_mat > 0.], 100-gap)
    # ind = (mi_mat > thd_mi) | (dist_mat < thd_dist)
    s_mi = np.argsort(mi_mat, axis=0)
    s_dist = np.argsort(dist_mat, axis=0)
    mixed = np.append(s_mi[-k1:, :], s_dist[1:k2, :], axis=0)
    set(mixed[:, 0])
    adj = np.zeros_like(s_mi, dtype=bool)
    ind = [list(set(mixed[:, i])) for i in range(mixed.shape[1])]
    for count, ind_i in enumerate(ind):
        adj[count, ind_i] = True
        adj[ind_i, count] = True
    np.save(os.path.join(main_dir, 'midist_adj'), adj)
    return adj


# determine the connectivity of any two stats, the version of directed graph
def det_directed_connect_by_midist(mi_mat: np.ndarray, dist_mat: np.ndarray, k1=6, k2=6):
    # thd_mi = np.percentile(mi_mat[mi_mat > 0.], gap)
    # thd_dist = np.percentile(dist_mat[dist_mat > 0.], 100-gap)
    # ind = (mi_mat > thd_mi) | (dist_mat < thd_dist)
    s_mi = np.argsort(mi_mat, axis=0)
    s_dist = np.argsort(dist_mat, axis=0)
    mixed = np.append(s_mi[-k1:, :], s_dist[1:k2, :], axis=0)
    set(mixed[:, 0])
    adj = np.zeros_like(s_mi, dtype=bool)
    ind = [list(set(mixed[:, i])) for i in range(mixed.shape[1])]
    for count, ind_i in enumerate(ind):
        adj[count, ind_i] = True
    np.save(os.path.join(main_dir, 'mi_dist_adj'), adj)
    return adj


def det_connect_by_mi(mi_mat: np.ndarray, k=6, sym=True):
    s_mi = np.argsort(mi_mat, axis=0)
    adj = np.zeros_like(s_mi, dtype=bool)
    ind = [list(s_mi[-k:, i]) for i in range(s_mi.shape[1])]
    for count, ind_i in enumerate(ind):
        adj[count, ind_i] = True
        if sym:
            adj[ind_i, count] = True
    save_name = 'mi_adj' if sym else 'mi_adj_eq'
    np.save(os.path.join(main_dir, save_name), adj)
    return adj


# make link matrix
def make_link_mat(adj: np.ndarray):
    node_num = adj.shape[0]
    adj_ = adj.copy()
    adj_[range(node_num), range(node_num)] = True
    link_mat = np.zeros((node_num, adj_[0,:].sum(), node_num))
    for i, row_i in enumerate(adj_):
        link_mat[i, range(link_mat.shape[1]), np.nditer(np.where(row_i)[0])] = 1
    np.save(os.path.join(main_dir, 'link_mat'), link_mat)
    return link_mat


def det_connect_by_pcc(pcc_mat: np.ndarray, frac=0.25):
    npcc = pcc_mat.copy()
    sta_num = pcc_mat.shape[0]
    npcc[range(sta_num), range(sta_num)] = 1000.
    trius = pcc_mat[np.triu_indices(pcc_mat.shape[0], 1)]
    sepline = np.percentile(trius, q=100*(1-frac))
    adj = (npcc > 0.) & (npcc < 10.)
    adj[npcc < sepline] = False
    npcc[npcc < sepline] = 1000.
    return adj, npcc


def det_connect_by_pccdist(pcc_mat: np.ndarray, dist_mat: np.ndarray, k1=6, k2=7):
    # thd_mi = np.percentile(mi_mat[mi_mat > 0.], gap)
    # thd_dist = np.percentile(dist_mat[dist_mat > 0.], 100-gap)
    # ind = (mi_mat > thd_mi) | (dist_mat < thd_dist)
    s_pcc = np.argsort(pcc_mat, axis=0)
    s_dist = np.argsort(dist_mat, axis=0)
    mixed = np.append(s_pcc[-(k1+1):-1, :], s_dist[1:k2, :], axis=0)
    set(mixed[:, 0])
    adj = np.zeros_like(s_pcc, dtype=bool)
    ind = [list(set(mixed[:, i])) for i in range(mixed.shape[1])]
    for count, ind_i in enumerate(ind):
        adj[count, ind_i] = True
        adj[ind_i, count] = True
    np.save(os.path.join(main_dir, 'pccdist_adj'), adj)
    return adj


def draw_graph(adj: np.ndarray, pos: np.ndarray):
    G = nx.Graph()
    G.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0]-1):
        for j in range(i, adj.shape[0]):
            if adj[i, j]:
                G.add_edge(i, j)
    nx.draw(G, nx.rescale_layout(pos), node_size=200, width=0.75, with_labels=True, font_size=6)
    print(nx.number_connected_components(G))
    plt.show()


def draw_dgraph(adj: np.ndarray, pos: np.ndarray):
    G = nx.DiGraph()
    G.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0]-1):
        for j in range(adj.shape[0]):
            if adj[i, j]:
                G.add_edge(j, i)
    nx.draw(G, nx.rescale_layout(pos), node_size=60, width=0.75, with_labels=False, font_size=6)
    # print(nx.number_connected_components(G))
    plt.show()


if __name__ == '__main__':
    main_dir = "info"

    mi = np.load(os.path.join(main_dir, 'mi.npy'))
    dist = np.load(os.path.join(main_dir, 'stat_dists.npy'))
    pos = np.load(os.path.join(main_dir, 'stat_pos.npy'))
    adj = det_directed_connect_by_midist(mi, dist, k1=6, k2=8)
    # adj = det_connect_by_midist(mi, dist, k1=6, k2=6)
    # adj = det_connect_by_mi(mi, k=6, sym=True)
    # adj = det_connect_by_pccdist(pcc, dist)
    # link_mat = make_link_mat(adj)

    # adj, new_pcc = det_connect_by_pcc(pcc, frac=0.16)

    # adj = np.load(os.path.join(main_dir, 'mi_adj.npy'))
    draw_dgraph(adj, pos=pos)
