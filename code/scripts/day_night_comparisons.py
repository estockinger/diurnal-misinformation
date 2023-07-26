import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
import numpy as np
import scripts.constants as constants
import scripts.utils as sutils
from scripts.enums import Clusters, Columns

COMPARE_COLS = {i: ('day', 'night') for i in tuple(('daytime', 'suntime', 'waking_time'))}

def day_night_safety(t, daystart, dayend, safety_margin=(1, 1)):
    if (safety_margin[0] > 0 and sutils.within_n_hours_before_or_after_t(t, daystart, safety_margin[0],
                                                                         include_edges=True)) or \
            (safety_margin[1] > 0 and sutils.within_n_hours_before_or_after_t(t, dayend, safety_margin[1],
                                                                              include_edges=True)):
        return 'safety'
    elif sutils.t_between(t, daystart, dayend, include_edges=False):
        return 'day'
    else:
        return 'night'


def get_waking_time(df, day_times, waking_times, sun_times, safety_margin=(1, 1)):
    df['daytime'] = df.apply(
        lambda row: day_night_safety(row.name[2], day_times['sunrise'], day_times['sunset'], safety_margin), axis=1)
    df['suntime'] = df.apply(lambda row: day_night_safety(row.name[2], sun_times.loc[row.name[1], 'sunrise'],
                                                          sun_times.loc[row.name[1], 'sunset'], safety_margin), axis=1)
    df['waking_time'] = df.apply(lambda row: day_night_safety(row.name[2], *waking_times[row.name[-1]], safety_margin),
                                 axis=1)


def pair_cols(
        df,
        for_col,
        ivals,
        compare_cols=None,
        ratio_col='ratio_norm',
        test=mannwhitneyu,
        plot=False,
):
    if compare_cols is None:
        compare_cols = COMPARE_COLS
    shared_args = dict(y=ratio_col, hue=Columns.CLUSTER.value, palette=Clusters.palette())

    subcols = ['Statistic', 'P-Value']
    if test == mannwhitneyu:
        subcols.append('less')

    statdf = pd.DataFrame(
        0,
        index=pd.MultiIndex.from_product([ivals, list(Clusters) + ['total']]),
        columns=pd.MultiIndex.from_product([compare_cols.keys(), subcols])
    )

    for i in ivals:
        df_i = df.xs(i, level=for_col)
        for c in list(Clusters) + ['total']:
            tmpdf = df_i.xs(c, level=Columns.CLUSTER.value).reset_index()
            for col, (val1, val2) in compare_cols.items():
                d1 = tmpdf.loc[tmpdf[col] == val1, ratio_col]
                d2 = tmpdf.loc[tmpdf[col] == val2, ratio_col]
                stat, p = test(d1, d2)
                if test == mannwhitneyu:
                    _, pless = test(d1, d2, alternative='less')
                    _, pgreater = test(d1, d2, alternative='greater')
                    statdf.loc[(i, c), col] = stat, p, val1 if pless < pgreater else val2
                else:
                    statdf.loc[(i, c), col] = stat, p

    if plot:
        fig, axes = plt.subplots(len(ivals), len(compare_cols), sharex='col', sharey='row',
                                 figsize=(3 * len(compare_cols), 3 * len(ivals)))
        if len(ivals) == 1:
            axes = np.expand_dims(axes, axis=0)
        axi = iter(axes.flat)

        for i in ivals:
            df_i = df.xs(i, level=for_col)
            for col in compare_cols.keys():
                if plot:
                    ax = next(axi)
                    sns.pointplot(data=df_i.reset_index(), x=col, ax=ax, **shared_args)
                    ax.set_ylabel(str(i).replace('/', '/\n') if ax in axes[:, 0] else '')
                    ax.set_xlabel(col.replace('_', ' ') if ax in axes[-1, :] else '')

        hh, ll = axes[0, 0].get_legend_handles_labels()
        for ax in axes.flat:
            ax.legend().remove()
        fig.legend(hh, ll, bbox_to_anchor=(1.1, 0.8))
    return statdf


def find_distribution_side(statdf, threshold=0.05):
    statdf[('lower', 'during')] = '-'
    # mask insignificant values
    less = statdf.xs('less', level=1, axis=1).mask(statdf.xs('P-Value', level=1, axis=1) > threshold)
    # Check if the values are all the same
    mask = less.apply(
        lambda row: len(row.dropna().values) > 0 and (row.dropna().values == row.dropna().values[0]).all(),
        axis=1)
    # Assign the value only if it's the same for all significant values of p
    statdf.loc[mask, ('lower', 'during')] = less.loc[mask].apply(lambda row: row.dropna().values[0], axis=1)
    return statdf[[i for i in statdf.columns if i[1] != 'less']]
