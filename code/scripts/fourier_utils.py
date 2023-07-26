import pickle
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import similaritymeasures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import matplotlib.ticker as mticker

from scripts.enums import Columns, Clusters
from scripts.utils import hours_to_mins, cons_max
from scripts.path_utils import *


CountryInfo = namedtuple("CountryInfo", "activity activity_smoothed ratio_smoothed stats_activity stats_ratio waking_times")
SineParams = namedtuple('SineParams', 'amplitude phase offset')


class Decomposer():
    def __init__(self, y_col, cluster_col, calculate_clusters, display_clusters, curve_path_manager, freq=.25):
        self.y_col = y_col
        self.cluster_col = cluster_col
        self.freq = freq
        self.display_clusters = display_clusters
        self.calculate_clusters = calculate_clusters
        self.curve_path_manager = curve_path_manager
        self.signal_df = curve_path_manager.load('signal')[[y_col]]
        self.signal_df.index.names = ['clock time', cluster_col]

        for i in self.signal_df.index.unique(self.cluster_col):
            if i not in calculate_clusters:
                self.signal_df.drop(index=i, level=self.cluster_col, inplace=True)

    def smooth_over_signal(self, padding=3, std=3):
        tmp = self.signal_df.unstack(level=self.cluster_col)
        tmp = pd.concat([tmp.iloc[-padding:], tmp, tmp.iloc[:padding]])
        self.signal_df = (tmp
                          .rolling(window=2 * padding, win_type='gaussian', center=True)
                          .mean(std=std)
                          .iloc[padding:-padding]
                          .stack(level=self.cluster_col))

    def fit_single_freq(self, periods=7):
        return {
            c: sine_fit(self.signal_df.xs(c, level=self.cluster_col).values.flatten(), periods)[2]
            for c in self.calculate_clusters
        }

    def plot_single_fit(self, single_fit_params, periods=7):
        fig, axes = plt.subplots(len(self.display_clusters), 1, figsize=(15, 6), tight_layout=True, sharex='all',
                                 sharey='all')
        x = np.tile(self.signal_df.xs(self.display_clusters[0], level=self.cluster_col).values.flatten(), periods)
        t = np.linspace(0, 24 * periods, int(24 * periods / self.freq), endpoint=False)
        for ax, (c, params) in zip(axes, single_fit_params.items()):
            x = np.tile(self.signal_df.xs(c, level=self.cluster_col).values.flatten(), periods)
            ax.plot(t, x, '.')
            ax.plot(t, sin_model(t, *params, freq=self.freq))
            ax.set_ylabel(c)
            ax.set_xlim(0, 24 * periods)
            ax.set_xticks(list(range(0, 24 * periods + 1, 12)), [f'{(i % 2) * 12}:00' for i in range(periods * 2 + 1)])
        return fig, axes

    def plot_aligned_single_fit(self, cluster_params, periods=7):
        palette = Clusters.palette()
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), tight_layout=True, sharex='all', sharey='all')

        for c, p in cluster_params.items():
            if c in self.display_clusters:
                y = self.signal_df.xs(c, level=self.cluster_col).values.flatten()
                t = np.linspace(0, 24 * periods, int(24 * periods / self.freq), endpoint=False)
                axes[0].plot(t, np.tile(y, periods), '.', color=palette[c])
                axes[0].plot(t, sin_model(t, *p, freq=self.freq), color=palette[c], label=c)

                tshift = t - p.phase / self.freq
                axes[1].plot(tshift, np.tile(y, periods), '.', color=palette[c])
                axes[1].plot(t[-1] + tshift, np.tile(y, periods), '.', color=palette[c])
                axes[1].plot(t, sin_model(t, amplitude=p.amplitude, phase=0, offset=p.offset, freq=self.freq),
                             color=palette[c])
                axes[0].set_xlim(0, 24 * periods)
                axes[1].set_xlim(0, 24 * periods)
                axes[1].set_xticks(list(range(0, 24 * periods + 1, 12)),
                                   [f'{(i % 2) * 12}:00' for i in range(periods * 2 + 1)])

        fig.legend(loc='upper right', bbox_to_anchor=(1.1, 0.5))
        axes[0].set_ylabel(f'sampled {self.curve_path_manager.CURVE_TYPE} and fitted curve')
        axes[1].set_ylabel('shifted by fitted phase')
        return fig, axes

    def get_nmax_per_cluster(self, n=(2, 5)):
        distances = {}
        t = np.linspace(0, 24, 24 * 4, endpoint=False)
        for i, c in enumerate(self.calculate_clusters):
            y_test = self.signal_df.xs(c, level=self.cluster_col)[self.y_col].values.flatten()
            t1 = np.zeros((y_test.size, 2))
            t1[:, 0] = t
            t1[:, 1] = y_test
            fhat, psd, freq = decompose(y_test)

            from_to = (n[0], n[1])
            distances[c] = pd.DataFrame(index=range(n[0], n[1]), columns=similarity_measures.keys())
            for i in range(*from_to):
                fhat_r, psd_r, mis = get_fhat_reduced(i, psd, fhat)
                t2 = np.zeros((y_test.size, 2))
                t2[:, 0] = t
                t2[:, 1] = np.fft.irfft(fhat_r)
                for k, func in similarity_measures.items():
                    distances[c].loc[i, k] = func(t1, t2)
        return get_nmax_per_cluster(distances)

    def recombine_n_largest_frequencies(self, nmax):
        cluster_params = {}

        cluster_recomb = pd.DataFrame(
            index=np.linspace(0, 24, 24 * 4, endpoint=False),
            columns=self.calculate_clusters
        )
        fig, axes = plt.subplots(2, len(self.display_clusters), figsize=(20, 7), sharex='row', sharey='row',
                                 tight_layout=True)
        for i, c in enumerate(self.calculate_clusters):
            y_test = self.signal_df.xs(c, level=self.cluster_col)[self.y_col].values.flatten()
            if c in self.display_clusters:
                axes[0][i].set_title(c)
                recomb, sine_params = decompose_fft(y_test, axes[0][i], axes[1][i], nmax=nmax[c])
            else:
                recomb, sine_params = decompose_fft(y_test, None, None, nmax=nmax[c])
            cluster_params[c] = sine_params
            cluster_recomb[c] = recomb

        single_dict = {(outerKey, innerKey): values for outerKey, innerDict in cluster_params.items() for
                       innerKey, values in innerDict.items()}
        return pd.DataFrame.from_dict(single_dict), cluster_recomb

    def get_and_store_waking_times(self, cluster_recomb, store=True):
        waking_times = get_waking_times(cluster_recomb.stack(), 1)
        if store:
            self.curve_path_manager.to_latex(
                pd.DataFrame(waking_times, index=['onset of heightened activity', 'end of heightened activity']).T
                .applymap(hours_to_mins)
                .style,
                label='waking_times', caption='Onset and end of heightened activity per cluster')
        return waking_times

    def get_and_store_stats(self, cluster_recomb, waking_times, store=True):
        stats = find_peaks_and_valleys(cluster_recomb, waking_times)
        stats_sorted = stats_to_min_max(stats, self.cluster_col, self.curve_path_manager.CURVE_TYPE,
                                        self.calculate_clusters)

        if store:
            df_style = (stats_sorted.style
                        .format({(c, r): f for r, f in zip(
                ['clock time', 'hrs past waking', self.curve_path_manager.CURVE_TYPE],
                [lambda x: f'{x:.2f}', lambda x: f'{x:.2f}', lambda x: f'{x:.3f}']
            ) for c in ['max', 'min']}, na_rep='-')
                        .hide(axis='index', level='i', names=True))
            self.curve_path_manager.to_latex(
                df_style,
                f'{self.curve_path_manager.CURVE_TYPE}_stats',
                f"Times of maximum and minimum {self.curve_path_manager.CURVE_TYPE}",
                True)
        return stats_sorted

    def plot_aligned(self, cluster_recomb, stats, by, ampm=True,
                     factor_of_baseline=False, hatch_area=False, annotate=2):
        filter_df = lambda df: (df
                                .loc[df.index.get_level_values(level=self.cluster_col).isin(self.display_clusters)]
                                .reset_index()
                                .set_index([by, self.cluster_col]))
        g = align_and_plot_orig_and_recomp(
            filter_df(self.signal_df),
            filter_df(cluster_recomb),
            stats,
            t_col=by,
            xlabel=by,
            y=self.y_col,
            ylabel=self.curve_path_manager.CURVE_TYPE,
            ampm=ampm,
            maxylim=self.signal_df.groupby(self.cluster_col)[self.y_col].max().max(),
            annotate=annotate)

        if hatch_area:
            grp = stats['max'][by].groupby(level=self.cluster_col)

            mpl.rcParams['hatch.linewidth'] = 1.5  # previous svg hatch linewidth
            g.fill_between(list(np.arange(grp.min().min(), grp.min().max() + .25, .25)), *g.get_ylim(), fc="none",
                           ec='lightgray', alpha=1, lw=0,
                           zorder=0, hatch="////")
            g.fill_between(list(np.arange(grp.max().min(), grp.max().max() + .25, .25)), *g.get_ylim(), fc="none",
                           ec='lightgray', alpha=1, lw=0,
                           zorder=0, hatch="////")

        if factor_of_baseline:
            baseline = 1 / len(cluster_recomb.index.unique(level=0))
            g.axes.yaxis.set_major_locator(mticker.FixedLocator([baseline * i / 2 for i in range(6)]))
            g.axes.yaxis.set_major_formatter(mticker.FixedFormatter([i / 2 for i in range(6)]))
            g.set_ylabel(f'{self.curve_path_manager.CURVE_TYPE} (factor of mean)')

        g.set_xlabel("")
        plt.tight_layout()
        self.curve_path_manager.save_plot(g.get_figure(), f'{self.curve_path_manager.CURVE_TYPE}_fourier_{by}')
        return g


def after_waking(waking, t):
    return (t - waking) % 24


def align_and_plot_orig_and_recomp(df_original, df_recomb, stats,
                                   t_col="clock time", xlabel="time of day", y="activity", ylabel="activity",
                                   scatter_alpha=1,
                                   ampm=False, maxylim=None, annotate=2, **args
                                   ):
    def ticks_format(val, i):
        return f'{hours_to_mins(val, format_ampm=ampm)}{"" if ampm else "h"}'

    cluster_palette = Clusters.palette()
    g = sns.lineplot(data=df_recomb.reset_index(), x=t_col, y=y, hue=Columns.CLUSTER.value, palette=cluster_palette,
                     lw=2, legend=False, **args)

    sns.scatterplot(
        data=df_original.reset_index(), x=t_col, y=y,
        hue=Columns.CLUSTER.value, palette=cluster_palette,
        legend=False, alpha=scatter_alpha, s=12, **args)

    maxidxs = (stats
               .loc[(stats.index.get_level_values(level=Columns.CLUSTER.value) != "total") & (
            stats.index.get_level_values(level='i') < annotate), ('max', xlabel)]
               .sort_values())
    maxvals = []

    def get_xoffset(j, maxidxs, vals):
        if j < len(maxidxs) - 1 and maxidxs[j + 1] - maxidxs[j] < .5 and abs(
                vals.loc[maxidxs[j]] - vals.loc[maxidxs[j + 1]]) < .1:
            return -6
        elif j and maxidxs[j] - maxidxs[j - 1] <= .75 and abs(vals.loc[maxidxs[j]] - vals.loc[maxidxs[j - 1]]) < .1:
            return 3
        else:
            return -3

    if annotate:
        for j, ((c, i), idx) in enumerate(maxidxs.items()):
            val = df_recomb.loc[(idx, c), y]

            g.axvline(x=idx, ymin=.0, ymax=val / maxylim, color=cluster_palette[c], ls="--")
            xtext = get_xoffset(j, maxidxs, df_recomb.xs(c, level=Columns.CLUSTER.value)[y])
            g.annotate(
                f'{hours_to_mins(idx, format_ampm=ampm)}', xy=(idx, val),
                xytext=(xtext, 10), textcoords="offset points",
                rotation=90, zorder=5,
                bbox=dict(boxstyle="round,pad=0.1,rounding_size=0.2", fc="white", alpha=.6, zorder=4))
            maxvals.append(val)

    if annotate == "top":
        sns.scatterplot(x=maxidxs.sort_values(), y=maxvals, zorder=4, color='black', **args)

    elif annotate:
        sns.scatterplot(x=maxidxs.sort_values(), y=maxvals, zorder=4, ax=g, color='black')

    g.axes.xaxis.set_major_locator(mticker.MultipleLocator(6))
    g.axes.xaxis.set_major_formatter(mticker.FuncFormatter(ticks_format))

    g.axes.tick_params(axis='x', which='both', zorder=5)
    g.axes.tick_params(axis='x', which='minor', labelsize='small')

    if annotate == "bottom":
        g.axes.xaxis.set_minor_locator(mticker.FixedLocator(locs=maxidxs.values))
        g.axes.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, ii: ticks_format(v, ii)))
        plt.setp(g.axes.xaxis.get_minorticklabels(), rotation=90, va='bottom', y=0.05, backgroundcolor="white",
                 zorder=5)
    else:
        g.axes.xaxis.set_minor_locator(mticker.MultipleLocator(3))
        g.axes.xaxis.set_minor_formatter(mticker.NullFormatter())

    plt.grid(False)

    g.set_xlim(0, 24)
    g.set_ylim(0, maxylim)
    g.set_xlabel(xlabel)
    g.set_ylabel(ylabel)
    return g


def align_signal_by_waking(df, waking_times):
    df["hrs past waking"] = df.apply(lambda r: after_waking(waking_times[r.name[1]][0], r.name[0]), axis=1)


def get_waking_times(activity_levels, level, n=65):
    waking_times = {}
    for i, c in enumerate(activity_levels.index.unique(level)):
        tmp = activity_levels.xs(c, level=level)
        tmp = pd.concat([tmp, tmp.iloc[:n]])
        ts = cons_max(tmp, n)
        waking_times[c] = ts
    return waking_times


def find_peaks_and_valleys(recomb_dict, waking_times, t=np.linspace(0, 24, 24 * 4, endpoint=False)):
    stats = {}
    for c, recomb in recomb_dict.items():
        peak_is = argrelextrema(recomb.values, np.greater)
        valley_is = argrelextrema(recomb.values, np.less)
        for i, j in enumerate(peak_is[0]):
            stats[("max", i, c)] = [t[j], after_waking(waking_times[c][0], t[j]), recomb.iloc[j]]
        for i, j in enumerate(valley_is[0]):
            stats[("min", i, c)] = [t[j], after_waking(waking_times[c][0], t[j]), recomb.iloc[j]]
    return stats


def sort_by(df, c_col, col_name, by='max'):
    df_e = (df.xs(by, level='kind').reset_index()
            .sort_values([c_col, col_name], ascending=[True, by != 'max'])
            .set_index([c_col, 'i'])
            )
    df_e.index = pd.MultiIndex.from_arrays(
        [df_e.index.get_level_values(c_col), df_e.groupby(level=0).cumcount()], names=[c_col, 'i'])
    return df_e


def stats_to_min_max(stat_dict, c_col, col_name, order):
    df_stats = pd.DataFrame.from_dict(stat_dict)
    df_stats = df_stats.T.swaplevel(1, 2).sort_index()
    df_stats.columns = ['clock time', 'hrs past waking', col_name]
    df_stats.index.names = ['kind', Columns.CLUSTER.value, 'i']
    df_max = sort_by(df_stats, c_col, col_name, by="max")
    df_min = sort_by(df_stats, c_col, col_name, by="min")
    return pd.concat([df_max, df_min], axis=1, keys=['max', 'min']).loc[order]


def get_nmax_per_cluster(c_df_dict):
    return {c: df.pct_change().idxmin(axis=0).mode().min() for c, df in c_df_dict.items()}

similarity_measures = {
    "pcm": similaritymeasures.pcm,
    "frechet": similaritymeasures.frechet_dist,
    "area_between_two_curves": similaritymeasures.area_between_two_curves,
    "curve_length_measure": similaritymeasures.curve_length_measure,
    "dtw": lambda a, b: similaritymeasures.dtw(a, b)[0],
    "mae": mean_absolute_error,
    "mse": mean_squared_error
}


def sin_model(x, amplitude, phase, offset, freq):
    return np.sin(x * freq - phase) * amplitude + offset


def sine_fit(y, periods):
    x = np.tile(y, periods)
    t = np.linspace(0, 24 * periods, x.size, endpoint=False)

    guess_amplitude = (max(x) - min(x)) / 2
    guess_phase = 0
    guess_offset = np.mean(x)

    p0 = SineParams(guess_amplitude, guess_phase, guess_offset)
    fit = curve_fit(lambda x, a, p, o: sin_model(x, a, p, o, freq=t[1]), t, x, p0)

    return t, x, SineParams(*fit[0]), SineParams(*p0)


def decompose(y, n=24 * 4, dt=.25):
    fhat = np.fft.rfft(y, n)
    psd = fhat * np.conj(fhat) / n  # power spectrum
    freq = (1 / (dt * n)) * np.arange(n)
    return fhat, psd, freq[:psd.size]


def get_fhat_reduced(max_n, psd, fhat):
    psd_idxs = np.zeros(psd.size)
    mis = np.abs(psd).argsort()[-1:-max_n - 2:-1]  # ids_half starts at 1
    psd_idxs[mis] = True
    return psd_idxs * fhat, psd * psd_idxs, mis


def to_sinewave(amplitude, phase, freq, x):
    return 1 / len(x) * amplitude * np.cos(freq * 2 * np.pi * x + phase)
    # return 1/ len(x) * (fourier_coeff.real * np.cos(freq * 2 * np.pi * x) - fourier_coeff.imag * np.sin(freq * 2 *
    # np.pi * x))


def decompose_fft(y, ax0, ax1, nmax=1):
    x = np.linspace(0, 24, 24 * 4, endpoint=False)
    fft3, psd, freqs = decompose(y)
    recomb = np.zeros((len(x),))
    sine_params = {}

    mis = np.abs(psd).argsort()[-1:-nmax - 2:-1]
    for i in mis:
        amplitude, phase = abs(fft3[i]), np.angle(fft3[i])
        sinewave = to_sinewave(amplitude, phase, freqs[i], x)
        if 0 < i:
            if ax0:
                ax0.plot(x, sinewave, label=f'{i}')
            sinewave *= 2
            sine_params[i] = {'amplitude': amplitude, 'phase': phase}
        recomb += sinewave

    if ax0:
        ax0.legend()
    if ax1:
        ax1.plot(x, recomb, x, y)
    return recomb, sine_params

def plot_difference(scatter_df_A, line_df_A, stats_A, scatter_df_B, line_df_B, stats_B, usex, c_col, y_col, consider_clusters, kwargs_A, kwargs_B, buffer= 0, adjust_to_baseline=True):
        
    def prep_df(df, rename_dict=None):
        if rename_dict is None:
            rename_dict = {}
        return (df
                .loc[df.index.get_level_values(level=c_col).isin(consider_clusters)]
                .rename(columns=rename_dict)
                .reset_index()
                .set_index([usex, Columns.CLUSTER.value]))

    maxylim = max(
        scatter_df_A.groupby(c_col)[y_col].max().max(),
        scatter_df_B.groupby(c_col)[y_col].max().max()) + buffer

    g = align_and_plot_orig_and_recomp(
        prep_df(scatter_df_A), prep_df(line_df_A), stats_A,
        t_col=usex, xlabel=usex, ls=":", y=y_col, maxylim=maxylim, ampm = usex=='clock time', **kwargs_A)

    g = align_and_plot_orig_and_recomp(
        prep_df(scatter_df_B), prep_df(line_df_B), stats_B,
        t_col=usex, xlabel=usex, y=y_col, ax=g, maxylim=maxylim, ampm = usex=='clock time', **kwargs_B)

    if adjust_to_baseline:
        baseline = 1 / len(line_df_A.index.unique(level=0))
        g.axes.yaxis.set_major_locator(mticker.FixedLocator([baseline * i / 2 for i in range(6)]))
        g.axes.yaxis.set_major_formatter(mticker.FixedFormatter([i / 2 for i in range(6)]))
    plt.tight_layout()
    return g


def load_activity_ratio_type(config, country_config, data_type, activity_decomposer, ratio_decomposer, consider_clusters):
    with open(get_waking_time_path(config, country_config, data_type), "rb") as f:
        waking_times = pickle.load(f)
    apm = ActivityPathManager(config, country_config.LABEL, user_type=data_type)
    rpm = MachinatedPathManager(config, country_config.LABEL, user_type=data_type)
    activity = apm.load('signal')[[activity_decomposer.y_col]]
    activity_smoothed = apm.load('fourier')[[activity_decomposer.y_col]]
    ratio_smoothed = rpm.load('fourier')[[ratio_decomposer.y_col]]
    activity.index.names = ['clock time', 'cluster']
    ratio_smoothed.index.names = ['clock time', 'cluster']
    for df, c in zip((activity_smoothed, activity, ratio_smoothed),
                     (activity_decomposer.y_col, activity_decomposer.y_col, ratio_decomposer.y_col)):
        for i in df.index.unique(activity_decomposer.cluster_col):
            if i not in consider_clusters:
                df.drop(index=i, level=activity_decomposer.cluster_col, inplace=True)
        align_signal_by_waking(df, waking_times)
    stats_activity = activity_decomposer.get_and_store_stats(activity_smoothed[activity_decomposer.y_col].unstack(), waking_times, store=False)
    stats_ratio = ratio_decomposer.get_and_store_stats(ratio_smoothed[ratio_decomposer.y_col].unstack(), waking_times, store=False)
    return CountryInfo(activity, activity_smoothed, ratio_smoothed, stats_activity, stats_ratio, waking_times)