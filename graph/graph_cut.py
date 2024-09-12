import os
import joblib
import numpy
import matplotlib

import random_color

import numpy as np
import pandas as pd
import igraph as ig

import matplotlib.pyplot as plt
import networkx as nx

matplotlib.use('TKAgg')


def edge_to_remove(graph):
    G_dict = nx.edge_betweenness_centrality(graph, weight="weight")
    edge = ()

    # extract the edge with highest edge betweenness centrality score
    for key, value in sorted(G_dict.items(), key=lambda item: item[1], reverse=True):
        edge = key
        break
    return edge


def girvan_newman(graph):
    # find number of connected components
    sg = nx.connected_components(graph)
    sg_count = nx.number_connected_components(graph)

    while sg_count == 1:
        graph.remove_edge(edge_to_remove(graph)[0], edge_to_remove(graph)[1])
        sg = nx.connected_components(graph)
        sg_count = nx.number_connected_components(graph)

    return [graph.subgraph(c).copy() for c in sg]


def first_graph_cut(graph, stat_info, n=8):
    stat_info.sort_index(inplace=True)
    caps = stat_info['total_cap'].to_numpy()
    tmp_sgs = girvan_newman(graph)
    count = 1
    sg_cap = [caps[list(sg_i.nodes())].sum() for sg_i in tmp_sgs]
    sgs = tmp_sgs
    tmp_n = len(tmp_sgs)
    print(f'### after {count}-st cut, subgraph cap is {sg_cap}')
    while tmp_n < n:
        count += 1
        cur_id = sg_cap.index(max(sg_cap))
        tmp_sg = sgs.pop(cur_id)
        sg_cap.pop(cur_id)
        tmp_sgs = girvan_newman(tmp_sg)
        sg_cap = sg_cap + [caps[list(sg_i.nodes())].sum() for sg_i in tmp_sgs]
        sgs = sgs + tmp_sgs
        tmp_n = tmp_n + len(tmp_sgs) - 1
        print(f'### after {count}-st cut, subgraph cap is {sg_cap}')

    return sgs, sg_cap


def make_graph(adj: np.ndarray, stat_info: pd.DataFrame, mi):
    g = nx.Graph()
    stat_info.sort_index(inplace=True)
    g.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0] - 1):
        for j in range(adj.shape[0]):
            if adj[i, j]:
                g.add_edge(i, j, weight=mi[i, j])
    return g


def make_partition(G, sg_list, save_dir):
    partition = np.zeros((len(sg_list), len(G.nodes)))
    partition_index = np.zeros((len(G.nodes), len(G.nodes)))
    partition_border = np.zeros(len(sg_list), dtype=int)
    pointer = 0
    for count, sg_i in enumerate(sg_list):
        nodes_in_subg = list(sg_i.nodes())
        nodes_in_subg.sort()
        partition[count, list(nodes_in_subg)] = 1
        for node_i in nodes_in_subg:
            partition_index[node_i, pointer] = 1
            pointer += 1
    partition_border = partition.sum(axis=1).cumsum().astype(np.int_)
    np.save(os.path.join(save_dir, 'partition_index'), partition_index)
    np.save(os.path.join(save_dir, 'partition_border'), partition_border)
    np.save(os.path.join(save_dir, 'partition'), partition)
    return partition, partition_index, partition_border


def make_partition_with_cap(G, sg_list, caps, save_dir):
    partition = np.zeros((len(sg_list), len(G.nodes)))
    for count, sg_i in enumerate(sg_list):
        partition[count, list(sg_i.nodes())] = caps[list(sg_i.nodes())] / caps[list(sg_i.nodes())].sum()
    np.save(os.path.join(save_dir, 'partition_with_cap'), partition)
    cap_feature = partition[np.where(partition != 0)]
    np.save(os.path.join(save_dir, 'cap_feature'), cap_feature)
    return partition, cap_feature


# this needs to be updated along with the partition matrix
def make_readout_index(partition: np.ndarray, save_dir):
    readout_index = np.zeros((partition.shape[1], partition.shape[1]))
    total_count = 0
    for row_i in partition:
        entries = np.where(row_i)[0]
        for entry in entries:
            readout_index[entry, total_count] = 1.
            total_count += 1
    np.save(os.path.join(save_dir, 'readout_index'), partition)
    return readout_index


def draw_dgraph_partition(adj: np.ndarray, pos: np.ndarray, sgl, ns=100, wid=0.5):
    G = nx.DiGraph()
    G.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0]-1):
        for j in range(adj.shape[0]):
            if adj[i, j]:
                G.add_edge(j, i)

    # color_list = np.array(list(map(lambda x: random_color.color(tuple(x)), random_color.ncolors(len(sgl)))))
    # idx = np.zeros(len(G.nodes), dtype=int)
    # for count, sg_i in enumerate(sgl):
    #     idx[list(sg_i.nodes())] = count
    # color_map = color_list[idx]

    cc = ['#00b050', '#7030a0', '#0070c0', '#f4b183', '#b4c7e7', '#c5e0b4']
    color_list = np.array(['#000000']*len(G.nodes))
    for count, sg_i in enumerate(sgl):
        color_list[list(sg_i.nodes())] = cc[count]
    nx.draw(G, nx.rescale_layout(pos), node_size=ns, width=wid,  font_size=6, node_color=color_list, arrowstyle='->', arrowsize=10, arrows=True, alpha=0.5)
    # print(nx.number_connected_components(G))
    plt.show()


def rescale_pos(pos: np.ndarray):
    rescaled = pos.copy()
    sort_res = np.argsort(pos, axis=0)
    ind = np.array(range(pos.shape[0]))
    rescaled[sort_res[:, 0], 0] = ind
    rescaled[sort_res[:, 1], 1] = ind
    return rescaled


if __name__ == '__main__':
    main_dir = "info"
    partition_save_dir = os.path.join(main_dir, 'partition_midist_newman')

    pos = np.load(os.path.join(main_dir, 'stat_pos.npy'))
    adj = np.load(os.path.join(main_dir, 'mi_dist_adj.npy'))
    mcs = joblib.load('merged_close_stats.jl')  # merged close stats info
    mi = np.load(os.path.join(main_dir, 'mi.npy'))

    G = make_graph(adj, mcs, mi)

    n_clusters = 6
    sg_list, sg_caps = first_graph_cut(G.copy(), mcs, n=n_clusters)
    # find communities in the graph
    # c = girvan_newman(G.copy())
    # find the nodes forming the communities


    # nx.draw(G, nx.rescale_layout(pos), node_size=200, width=0.75, with_labels=True, font_size=6)
    # plt.show()

    draw_dgraph_partition(adj, rescale_pos(pos), sg_list)
    # nx.draw(G, nx.rescale_layout(pos), node_size=100, width=0.5,  font_size=6, node_color=color_map)
    # plt.show()
    # partition, pindex, pborder = make_partition(G, sg_list, partition_save_dir)
    # partition_with_cap, cap_feature = make_partition_with_cap(G, sg_list, mcs['total_cap'].to_numpy(), partition_save_dir)



