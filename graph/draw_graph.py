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

from netgraph import Graph

matplotlib.use('TKAgg')


def rescale_pos(pos: np.ndarray):
    rescaled = pos.copy()
    sort_res = np.argsort(pos, axis=0)
    ind = np.array(range(pos.shape[0]))
    rescaled[sort_res[:, 0], 0] = ind
    rescaled[sort_res[:, 1], 1] = ind
    return rescaled


def truncate_colormap(cmapIn='jet', minval=0.0, maxval=1.0, n=100):
    cmapIn = plt.get_cmap(cmapIn)

    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f}'.format(n=cmapIn.name, a=minval, b=maxval),
        cmapIn(np.linspace(minval, maxval, n))
    )

    return new_cmap


if __name__ == '__main__':
    main_dir = "info"
    pos = np.load(os.path.join(main_dir, 'stat_pos.npy'))
    adj = np.load(os.path.join(main_dir, 'mi_dist_adj.npy'))
    mi = np.load(os.path.join(main_dir, 'mi.npy'))
    sg_list = joblib.load('sg_list.jl')
    rpos = rescale_pos(pos)
    rpos_dict = {}
    for count in range(rpos.shape[0]):
        rpos_dict[count] = (rpos[count, 0]/127*5 - 2.5, rpos[count, 1]/127*4 - 2)
    G = nx.DiGraph()
    mi_list = []
    G.add_nodes_from(range(adj.shape[0]))
    for i in range(adj.shape[0] - 1):
        for j in range(adj.shape[0]):
            if adj[i, j]:
                G.add_edge(j, i)
                mi_list.append(mi[i, j])

    cc = ['#ff0000',  # green
          '#7030a0',  # purple
          '#7b68ee',  # blue
          '#f4b183',  # light orange
          '#c0c0c0',  # light blue
          '#db7093']  # red
    color_list = np.array(['#000000'] * len(G.nodes))
    color_dict = {}
    for count, sg_i in enumerate(sg_list):
        color_list[list(sg_i.nodes())] = cc[count]
    for count, col in enumerate(color_list):
        color_dict[count] = col
    edge_alphas = [(300 + i) / (G.number_of_edges() + 299) for i in range(G.number_of_edges())]
    edge_alphas = [0.7] * G.number_of_edges()
    cmap = plt.cm.viridis
    # my_cmap = matplotlib.colors.ListedColormap(colors=['red', 'black', 'orange', 'seagreen', 'royalblue', 'purple'])
    # new_cmap = truncate_colormap(cmap, minval=0.0, maxval=0.5)
    new_cmap = cmap
    nodes = nx.draw_networkx_nodes(G, rpos, node_size=60, node_color=color_list, linewidths=1)
    edges = nx.draw_networkx_edges(
        G,
        rpos,
        node_size=60,
        arrowstyle="->",
        edge_color=mi_list,
        arrowsize=15,
        width=1.2,
        arrows=True,
        alpha=0.8,
        edge_cmap=new_cmap,
        edge_vmin=0.265,
        edge_vmax=0.779
    )
    for i in range(G.number_of_edges()):
        edges[i].set_alpha(edge_alphas[i])
    pc = matplotlib.collections.PatchCollection(edges, cmap=new_cmap)
    pc.set_array(mi_list)

    ax = plt.gca()
    ax.set_axis_off()
    ax.collections[0].set_edgecolor('#000000')
    plt.rc('font', size=24, weight='book')
    plt.colorbar(pc, ax=ax, label='Edge Weight')
    plt.show()
    # Graph(G, node_color=color_dict, node_layout=rpos_dict, arrows=True, node_size=4, edge_width=1.5)
    # plt.show()
    # plt.savefig('fig1.png', dpi=1000)