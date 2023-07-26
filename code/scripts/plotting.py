import math
import os
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def create_legend(
        ax, palette, huelabels, sortby_key=lambda t: t[0],
        loc=2, bbox_to_anchor=(1, 1), frameon=False, consider_axes=[],
        include_total=False, blob="patch"):
    handles = []
    labels = []
    for ax in consider_axes:
        old_handles, old_labels = ax.get_legend_handles_labels()
        for old_handle, old_label in zip(old_handles, old_labels):
            if old_label not in labels and old_label not in palette:
                labels.append(old_label)
                handles.append(old_handle)

    for k, v in palette.items():
        if k in huelabels and (include_total or k != "total"):
            labels.append(k)
            if blob == "patch":
                new_handle = mpatches.Patch(color=mcolors.rgb2hex(v), label=k)
            elif blob == "line":
                new_handle = mlines.Line2D([], [], color=mcolors.rgb2hex(v), label=k)
            else:
                break
            handles.append(new_handle)

    labels, handles = zip(*sorted(zip(labels, handles), key=sortby_key))
    ax.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_to_anchor, frameon=frameon)
    return labels, handles


def sort_legend(ax, sortby_key):
    # sort both labels and handles by labels
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 1:
        labels, handles = zip(*sorted(zip(labels, handles), key=sortby_key))
        ax.legend(handles, labels, loc="center right", bbox_to_anchor=(1, 1), frameon=False)


def theme_river_helper(facet, hue, df, palette=None, perline=5, sortby_key=lambda t: t[0], usecol="num_posts",
                       title=None, mul=4):
    facetlabels = [i for i in df.index.unique(facet) if i != "total"]
    flen = len(facetlabels)

    if flen > perline:
        fig, axstack = plt.subplots(math.ceil(flen / perline), perline, tight_layout=True,
                                    figsize=(math.ceil(perline * mul), math.ceil(flen / perline) * mul))
        axes = tuple(i for j in axstack for i in j)
    else:
        fig, axes = plt.subplots(1, flen, tight_layout=True, sharey='row', figsize=(14, mul))

    facet_dfs = OrderedDict(sorted({f: df.xs(f, level=facet) for f in facetlabels}.items()))

    for ax, (f, tmp) in zip(axes, facet_dfs.items()):
        i = tmp.index.levels[0]

        huelabels = [i for i in tmp.index.unique(hue)]
        ys = [tmp.xs(label, level=hue)[usecol].reindex(i, fill_value=0) for label in huelabels if label != "total"]

        args = {
            "baseline": "sym",
            "linewidth": 0
        }
        if palette:
            args["colors"] = [palette.get(i) for i in huelabels if i != "total"]
        ax.stackplot(
            i,
            *ys,
            labels=[label for label in huelabels if label != "total"],
            **args
        )
        ax.set_title(f"{f}")

        yticklabels = [int(abs(tick)) for tick in ax.get_yticks()]
        yticklocs = ax.get_yticks().tolist()
        ax.yaxis.set_major_locator(mticker.FixedLocator(yticklocs))
        ax.set_yticklabels(yticklabels)

    # sort both labels and handles by labels
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 1:
        labels, handles = zip(*sorted(zip(labels, handles), key=sortby_key))
        ax.legend(handles, labels, loc=2, bbox_to_anchor=(1, 1), frameon=False)

    if len(axes) > len(facet_dfs):
        for i in axes[len(facet_dfs):]:
            fig.delaxes(i)
    plt.suptitle(title if title else f"Number of posts by {hue} and {facet} over the course of a day")


def weighted_lineplot(data, y, *args, **kwargs):
    g = sns.lineplot(data=data, y=y, *args, **kwargs, err_style=None)
    g.fill_between(data.index, *get_lims(data, y, f"{y}_sem"), alpha=0.2)
    return g


def plot_and_save(config, country_config, func, pngname):
    save_dir = os.path.join(config.SAVE_ROOT_DIR, country_config.LABEL)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.clf()
        r = func()
        plt.savefig(os.path.join(save_dir, f"{pngname}.png"), bbox_inches='tight', dpi=300, transparent=False)
        plt.show()
        return r

def plot_correlation_pairs(df, label, config, country_config):
    df = df.select_dtypes(include=np.number)
    df.columns = df.columns.map(" ".join)
    set_title = np.vectorize(
        lambda ax, r, rho: ax.title.set_text("r = {:.2f} \n $\\rho$ = {:.2f}".format(r, rho))
        if ax is not None else None
    )
    r = plot_and_save(config, country_config, lambda: plot_correlation(df, method="spearman"), f"CORR_{label}_spearman")
    rho = plot_and_save(config, country_config, lambda: plot_correlation(df, method="pearson"), f"CORR_{label}_pearson")

    def pairplot():
        g = sns.PairGrid(df.astype(float), corner=True)
        g.map_diag(sns.histplot)
        g.map_lower(sns.scatterplot)
        set_title(g.axes, r, rho)
        plt.subplots_adjust(hspace=0.6)

    plot_and_save(config, country_config, pairplot, f"CORR_{label}_pairs")


def plot_correlation(df, method="spearman"):
    r = df.corr(method)
    plt.figure(figsize=(10, 6))
    sns.heatmap(r, vmin=-1, vmax=1, annot=True)
    plt.title(f"{method} correlation")
    return r


def plot_mean_and_std(x, y, ax, color, label, granularity, plot_mean=True):
    rmean, rstd = ((r := y.rolling(granularity, min_periods=1)).mean(std=1), r.std())
    rstd[0] = 0
    ci = 1.96 * rstd / np.sqrt(len(x))
    ax.fill_between(x, rmean - ci, rmean + ci, linestyle='-', color=color, interpolate=True, alpha=0.1)
    ax.plot(x, rmean if plot_mean else y, '-', c=color, label=label)
