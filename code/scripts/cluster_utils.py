from collections import namedtuple

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as hac
import scipy.stats as stats
from scripts.enums import Clusters, Columns
from scripts import preprocessing as sprocessing
from scipy.spatial.distance import pdist

ClusterInfo = namedtuple('ClusterInfo', 'average users result')


def user_to_cluster(users_to_clusters, cluster_nums_to_names, user, botdict=None):
    if botdict and True in botdict.keys() and user in botdict[True]:
        return Clusters.BOT.value
    return cluster_nums_to_names[users_to_clusters.get(user, None)]


def get_user_rhythms(df, posts_per_user, cutoff_post_nr=240):
    users_over_threshold = posts_per_user.loc[posts_per_user[Columns.NUM_POSTS.value] > cutoff_post_nr].index
    return sprocessing.group_by_with_total(
        df.loc[df[Columns.USER.value].isin(users_over_threshold)],
        group_by_cols=[Columns.USER, Columns.MIN_BINS15]
    )


def get_user_stats(df):
    stats = pd.DataFrame(df[Columns.CLUSTER.value].value_counts(dropna=False))
    stats.columns = ['posts']
    stats['users'] = df.groupby([Columns.CLUSTER.value])[Columns.USER.value].nunique()
    stats['posts per user'] = df[Columns.CLUSTER.value].value_counts(dropna=False) / stats['users']
    return stats


def get_distance_stats(cluster_to_num, wma, cluster_col, c0):
    stats = pd.DataFrame(index=list(Clusters)[1:], columns=list(Clusters))
    for i, c1 in enumerate(list(Clusters)[1:]):
        for c2 in list(Clusters)[i + 1:]:
            to_compare = wma.T.loc[(cluster_col == cluster_to_num[c1]) | (cluster_col == cluster_to_num[c2])]
            stats.loc[c1, c2] = stats.loc[c2, c1] = max(hac.ward(pdist(to_compare))[:, 2])
        to_compare = pd.concat([wma.T.loc[(cluster_col == cluster_to_num[c1])], c0], axis=0)
        max_dist = max(hac.ward(pdist(to_compare))[:, 2])
        stats.loc[c1, Clusters.INFREQUENT.value] = max_dist
    return stats


def get_avg_for_cluster(df, wma, cluster, window=6, win_type='gaussian', center=True, std=3):
    avg_cluster = df.loc[df[Columns.CLUSTER.value] == cluster] \
        .groupby(Columns.MIN_BINS15.value)[Columns.TWEET_ID.value] \
        .count()
    avg_cluster /= avg_cluster.sum()
    avg_cluster_wma = pd.concat([avg_cluster.iloc[-3:], avg_cluster, avg_cluster.iloc[:3]]) \
                          .rolling(window=window, win_type=win_type, center=center).mean(std=std).iloc[3:-3]
    return pd.DataFrame(avg_cluster_wma.values, columns=[cluster], index=wma.index).transpose()


class Clusterer:

    def __init__(self, df, posts_per_user, cutoff_post_nr=240, ratio_column=Columns.RATIO_BY_USER, window=6,
                 smoother="wma"):
        self.ratio_column = ratio_column.value
        self.window = window
        self.smoothed_rhythms = {'raw': get_user_rhythms(df, posts_per_user, cutoff_post_nr=cutoff_post_nr)}
        self.cluster_results = {}
        self.smoother = smoother

    def average_over_column(self, padding=3, wma_std=3, wma=True, sma=True, ewa=True):
        tmp = self.smoothed_rhythms['raw'][self.ratio_column].unstack(level=0).fillna(0).iloc[:-1]
        tmp = pd.concat([tmp.iloc[-padding:], tmp, tmp.iloc[:padding]])  # loop
        if wma and 'wma' not in self.smoothed_rhythms:
            self.smoothed_rhythms['wma'] = (tmp
                                            .rolling(window=self.window, win_type='gaussian', center=True)
                                            .mean(std=wma_std)
                                            .iloc[padding:-padding])

        if sma and 'sma' not in self.smoothed_rhythms:
            self.smoothed_rhythms['sma'] = tmp.rolling(self.window, min_periods=1).mean().iloc[padding:-padding]

        if ewa and 'ewa' not in self.smoothed_rhythms:
            self.smoothed_rhythms['ewa'] = tmp.ewm(span=self.window).mean().iloc[padding:-padding]

        return self.smoothed_rhythms

    def process_clusters(self, d, smooth_df, cut_off_level, criterion="maxclust"):
        result = pd.Series(hac.fcluster(d, cut_off_level, criterion=criterion))
        clusters = result.unique()
        users_in_cluster = dict()
        cluster_averages = pd.DataFrame(index=smooth_df.columns, columns=clusters)

        for c in clusters:
            cluster_index = result[result == c].index
            cluster = smooth_df.T.iloc[:, cluster_index]
            cluster_averages[c] = cluster.mean(axis=1)
            for user in cluster.columns:
                users_in_cluster[user] = c

        self.cluster_results[cut_off_level] = ClusterInfo(cluster_averages, users_in_cluster, result)
        return cluster_averages, users_in_cluster, result


def get_rename_map(label, cluster_info, num_clusters):
    rename_map = {}
    df_avg = cluster_info[label][num_clusters].average
    for cluster in df_avg.columns:
        cl = get_cluster_label(df_avg[cluster])
        if cl in rename_map.values():  # two of the same type?
            cl += ".1"
        rename_map[cluster] = cl
    return rename_map


def get_cluster_label(s_avg):
    max_idx = s_avg.idxmax()
    labels = ["night owls", "early birds", "morning risers", "10 am posters", "noon posters", "early afternoon posters",
              "afternoon posters", "evening posters", "night owls"]
    cuts = [3.5, 6, 9, 10.5, 14, 16, 18, 22, 25]
    for label, cut in zip(labels, cuts):
        if max_idx <= cut:
            print(label, max_idx)
            return label


def get_cluster_method(df, vclust):
    cvi_val = vclust.fit_predict(df)
    # lower is better for these indices
    for i in ['davies', 'cop', 'distortion']:
        for m in cvi_val.index.levels[0]:
            cvi_val.loc[(m, i)] *= 1
    df_res = pd.DataFrame(
        (cvi_val.loc['hierarchical'] - cvi_val.loc['kmeans']) > 0) \
        .applymap(lambda x: 'hierarchical' if x else 'kmeans') \
        .apply(pd.value_counts, axis=1).fillna(0)
    df_res.loc['total'] = df_res.apply(sum, axis=0)
    if df_res.loc["total", "hierarchical"] > df_res.loc["total", "kmeans"]:
        print("Use hierarchical clustering.")
    else:
        print("Use kmeans clustering.")
    return cvi_val, df_res


def get_cluster_size_borda(norm):
    norm_ranked = norm.loc['hierarchical'].apply(lambda s: (stats.rankdata(s) - 1).astype(int), axis=1)
    rank_df = pd.DataFrame.from_dict(dict(zip(norm_ranked.index, norm_ranked.values))).transpose()
    rank_df.columns = norm.columns
    return rank_df.apply(sum, axis=0).idxmax()


def get_user_rhythm_wma(user_rhythm, std=3, window=6, win_type='gaussian'):
    tmp = user_rhythm["ratio"].unstack(level=0).fillna(0).iloc[:-1]
    tmp = pd.concat([tmp.iloc[-3:], tmp, tmp.iloc[:3]])  # loop during the day
    return tmp.rolling(window=window, win_type=win_type, center=True).mean(std=std).iloc[3:99]


def plot_results(time_series, d, cut_off_level, criterion="maxclust"):
    result = pd.Series(hac.fcluster(d, cut_off_level, criterion=criterion))
    clusters = result.unique()
    users_in_cluster = dict()
    cluster_averages = pd.DataFrame(index=time_series.columns, columns=clusters)

    fig, axes = plt.subplots(1, cut_off_level, figsize=(30, 5), tight_layout=True)

    for c in clusters:
        cluster_index = result[result == c].index
        cluster = time_series.T.iloc[:, cluster_index]
        cluster_averages[c] = cluster.mean(axis=1)
        for user in cluster.columns:
            users_in_cluster[user] = c
        ax = axes[c - 1]
        ax.plot(cluster)
        ax.set_title(f'Cluster {c}: {len(cluster_index)} users')
        ax.xaxis.set_major_locator(plt.MaxNLocator(12))

    plt.show()
    return cluster_averages, users_in_cluster, result


def plot_dendogram(z):
    plt.figure(figsize=(15, 5))
    plt.title(f'Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    dd = hac.dendrogram(z,
                        truncate_mode='lastp',
                        p=15,
                        show_contracted=True,
                        leaf_rotation=90.,
                        leaf_font_size=8)

    for i, d, c in zip(dd['icoord'], dd['dcoord'], dd['color_list']):
        x = 0.5 * sum(i[1:3])
        y = d[1]
        plt.plot(x, y, 'o', c=c)
        plt.annotate(f"{y:.3g}", (x, y), xytext=(0, -5),
                     textcoords='offset points',
                     va='top', ha='center')
    plt.show()
    return dd
