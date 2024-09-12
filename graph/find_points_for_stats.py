import joblib
import os

import numpy as np
import pandas as pd

from geopy.distance import geodesic


def sel_nodes_for_sta(grid: pd.DataFrame, stats: pd.DataFrame, k):
    ret = {}
    for stat_id, stat in stats.iterrows():
        lat, lon = stat.lat, stat.lon
        dist = grid.apply(lambda x: geodesic((x.lat, x.lon), (lat, lon)).km, axis=1)
        smallest = dist.nsmallest(k)
        selected_ids = smallest.index.to_list()
        selected_ids.sort()
        dists = np.diag(smallest.sort_index())
        for count, p_id in enumerate(selected_ids):
            p_lat, p_lon = grid.loc[p_id, ['lat', 'lon']]
            for i in range(count+1, len(selected_ids)):
                dists[count, i] = geodesic((p_lat, p_lon), grid.loc[selected_ids[i]].to_list()).km
                dists[i, count] = dists[count, i]
        ret[stat_id] = (selected_ids, dists)
    return ret


if __name__ == '__main__':
    main_dir = 'info'
    grid = pd.read_csv('reduced_grid.csv')
    stats = joblib.load('merged_close_stats.jl')

    gps_for_st = sel_nodes_for_sta(grid, stats, k=8)
    joblib.dump(gps_for_st, os.path.join(main_dir, 'gps_for_st.jl'))