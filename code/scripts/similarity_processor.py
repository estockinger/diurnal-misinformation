import statistics
import sys
import matplotlib.colors as mcolors
import pymannkendall as mk
import seaborn as sns
import numpy as np
import pandas as pd
import similaritymeasures
from scripts.enums import Columns
from matplotlib import pyplot as plt
from scripts.plotting import *
from scripts.utils import *
from sklearn.metrics import mean_squared_error, mean_absolute_error

ALIGNMNENT_LIST = [
    "min_bins15",
    "min_distance",
    "min_activity",
    "max_activity",
    "first_inflection",
    "first_peak",
    "steepest_ascent"
]

SIMILARITY_MEASURES = {
    "pcm": similaritymeasures.pcm,
    "frechet": similaritymeasures.frechet_dist,
    "area_between_two_curves": similaritymeasures.area_between_two_curves,
    "curve_length_measure": similaritymeasures.curve_length_measure,
    "dtw": lambda a, b: similaritymeasures.dtw(a, b)[0],
    "mae": mean_absolute_error,
    "mse": mean_squared_error
}


def _plot_per_i_range(df, ii, palette, ax, legend=False, transparent=False, **kwargs):
    df[ii].plot.area(
        color=[mcolors.rgb2hex(palette[i]) for i in ii],
        ax=ax, legend=legend, **kwargs
    )
    if transparent:
        for c in ax.get_children()[1::2]:
            c.set_alpha(0)


def plot_daily_activities(
        df, c_col,
        alignment_list=None,
        y_col='activity',
        palette=None,
        **kwargs
):
    if alignment_list is None:
        alignment_list = [i for i in ALIGNMNENT_LIST if i in df.columns]
    df_melted = (df
    .reset_index()
    .melt(
        value_name="aligned",
        var_name="alignment_type",
        value_vars=alignment_list,
        id_vars=[c_col, y_col])
    )
    g = sns.FacetGrid(
        data=df_melted.loc[df_melted[Columns.CLUSTER.value] != "total"],
        hue=Columns.CLUSTER.value, col="alignment_type", legend_out=True, sharey="row", palette=palette, **kwargs)
    g.map(sns.lineplot, "aligned", y_col)
    g.set_axis_labels("time")
    g.set_titles(col_template="{col_name}")
    create_legend(g.fig, palette, df_melted[c_col].unique(), blob="line")
    plt.suptitle("Daily Activities aligned by Curve Features", y=1.02)


def shift(df_tmp, cluster, alignment, c_col, col="num_posts_norm"):
    return np.c_[df_tmp.xs(cluster, level=c_col).index.values, np.roll(df_tmp.xs(cluster, level=c_col)[col], alignment)]


def find_min_distance_shifts(df, r_col, c_col):
    ccs = df.index.unique(c_col)
    shift_by = pd.DataFrame(0, index=ccs, columns=SIMILARITY_MEASURES.keys())
    c1 = ccs[0]
    for measure, func in SIMILARITY_MEASURES.items():
        for c2 in ccs:
            min_distance = sys.maxsize
            for i, b in enumerate(df.index.levels[0]):
                dist = func(
                    shift(df, c2, i, c_col, col=r_col),
                    df.xs(c1, level=c_col)[r_col].reset_index().values)
                if dist < min_distance:
                    min_distance = dist
                    shift_by.loc[c2, measure] = -b
    shift_by["majority"] = shift_by.mode(axis=1)[0]
    return shift_by


def get_cluster_stats(df, r_col, c_col, waking_times):
    cluster_stats = df.groupby(level=c_col).agg(
        min_bins15=(r_col, lambda r: 0),
        min_activity=(r_col, lambda r: r.index[r.argmin()][0]),
        max_activity=(r_col, lambda r: r.index[r.argmax()][0]),
        first_inflection=(r_col, lambda r: find_first_inflection(r[r.index[r.argmin()][0]:])),
        first_peak=(r_col, lambda r: find_first_peak(r[r.index[r.argmin()][0]:])),
        steepest_ascent=(r_col, lambda r: r.index[np.argmax(np.diff(r))][0])
    )
    cluster_stats["waking_time"] = cluster_stats.apply(lambda r: waking_times[r.name][0], axis=1)
    return cluster_stats


def get_similarities(
        df, c_col, r_col, cluster_stats, shift_by
):
    similarities = pd.DataFrame(None,
                                index=pd.MultiIndex.from_arrays([
                                    ["features"] * len(cluster_stats.columns) +
                                    ["distance metrics"] * len(shift_by.columns),
                                    list(cluster_stats.columns) + list(shift_by.columns)]),
                                columns=SIMILARITY_MEASURES.keys())

    c_order = df.index.unique(c_col)
    c1 = c_order[0]
    for measure, func in SIMILARITY_MEASURES.items():
        for alignment in cluster_stats.columns:
            measures = []
            c1shift = shift(df, c1, int(cluster_stats[alignment].loc[c1] * 4), c_col, col=r_col)
            for c2 in c_order[1:]:
                measures.append(func(
                    c1shift,
                    shift(df, c2, int(cluster_stats[alignment].loc[c2] * 4), c_col, col=r_col)
                ))
            similarities.loc[("features", alignment), measure] = statistics.mean(measures)

        for alignment in shift_by.columns:
            measures = [
                func(
                    df.xs(c1, level=c_col)[r_col].reset_index().values,
                    shift(df, c2, int(-shift_by[alignment].loc[c2] * 4), c_col, col=r_col)
                ) for c2 in c_order[1:]]
            similarities.loc[("distance metrics", alignment), measure] = statistics.mean(measures)
    return similarities


class SimilarityProcessor:

    def __init__(self, activity, c_col, a_col, cluster_order, waking_times):
        self.activity = activity.loc[activity.index.get_level_values(level=c_col).isin(cluster_order)].copy()
        self.shift_by = None
        self.similarities = None
        self.cluster_stats = None
        self.c_col = c_col
        self.a_col = a_col
        self.cluster_order = cluster_order
        self.waking_times = waking_times

    def prep(self):
        self.cluster_stats = get_cluster_stats(
            self.activity,
            self.a_col, self.c_col, self.waking_times)

        find_hours_spent_awake(self.activity, self.cluster_stats, c=self.c_col)
        self.shift_by = find_min_distance_shifts(self.activity, r_col=self.a_col, c_col=self.c_col)
        self.similarities = get_similarities(self.activity, self.c_col, self.a_col, self.cluster_stats, self.shift_by)
        self.shift_by_min_distance(self.activity)

    def shift_by_min_distance(self, df):
        df["min_distance"] = \
            realign_df_by_cluster_stats_column(
                df,
                self.shift_by["majority"],
                c=self.c_col
            ).astype(float)


class ColSimilarityProcessor:

    def __init__(self, sp, ratios, i_palette, i_order, i_col="FactType2"):
        self.ratio_similarities = None
        self.ratios = ratios.loc[ratios.index.get_level_values(level=sp.c_col).isin(sp.cluster_order)].copy()
        self.beyond_threshold_mask = None
        self.prolonged_waking_mask = None
        self.sp = sp
        self.i_palette = i_palette
        self.i_order = i_order
        self.i_col = i_col

    def prep(self):
        self.sp.shift_by_min_distance(self.ratios)
        self.ratio_similarities = get_similarities(self.ratios, self.sp.c_col, self.i_col, self.sp.cluster_stats,
                                                   self.sp.shift_by)
