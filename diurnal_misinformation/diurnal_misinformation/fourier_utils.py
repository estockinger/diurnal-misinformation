import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import similaritymeasures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import argrelextrema, fftconvolve
from scipy.stats import mannwhitneyu
import matplotlib.ticker as mticker
from dataclasses import dataclass
from diptest import diptest

from .utils import smooth_looped, format_h_min, time_past_t, shift_rows_by
from .enums import Columns, Clusters
from .data_processor import DataProcessor
from .path_utils import save_plot

@dataclass
class FourierSignalRecomposition:
    signal: pd.DataFrame
    distances: pd.DataFrame
    frequencies: pd.Series
    sinewaves:pd.DataFrame
    recomposed_signal: pd.DataFrame
    increased_activity: dict = None


pd.set_option('future.no_silent_downcasting', True)


similarity_measures = {
    "pcm": similaritymeasures.pcm,
    "frechet": similaritymeasures.frechet_dist,
    "area_between_two_curves": similaritymeasures.area_between_two_curves,
    "curve_length_measure": similaritymeasures.curve_length_measure,
    "dtw": lambda a, b: similaritymeasures.dtw(a, b)[0],
    "mae": mean_absolute_error,
    "mse": mean_squared_error
}


def recompose(df, frequency_nrs=np.arange(2,5)):
    distances = df.apply(smoothed_signal_distances, result_type='expand', axis=1, frequency_nrs=frequency_nrs)
    nr_frequencies = min_pct_change(distances)
    sinwaves = df.apply(to_sinewaves, result_type='expand', axis=1, nr_frequencies=nr_frequencies)
    smoothed_signal = (sinwaves * 2).groupby(level=1, axis=1).sum().add(df.mean(axis=1), axis=0)
    return FourierSignalRecomposition(
        signal = df,
        distances=distances, 
        frequencies = min_pct_change(distances), 
        sinewaves = sinwaves, 
        recomposed_signal = smoothed_signal
    )


def smoothed_signal_distances(row, frequency_nrs):
    fhat, psd, _ = decompose(row, n=len(row), dt=row.index.diff()[1])
    t1 = np.vstack([row.index.to_numpy(), row.to_numpy()]).T
    t2 = t1.copy()
    distances = pd.Series(0, index=pd.MultiIndex.from_product([frequency_nrs, similarity_measures.keys()], names=["frequency", "metric"]))
    for i in frequency_nrs:
        fhat_r, _, _ = get_fhat_reduced(i, psd, fhat)
        t2[:, 1] = np.fft.irfft(fhat_r)
        distances[i] = [similarity_measures[j](t1, t2) for j in distances.index.levels[1]]
    return distances


def decompose(y, n, dt):
    fhat = np.fft.rfft(y, n)
    psd = fhat * np.conj(fhat) / n  # power spectrum
    freq = (1 / (dt * n)) * np.arange(n)
    return fhat, psd, freq[:psd.size]


def get_fhat_reduced(max_n, psd, fhat):
    psd_idxs = np.zeros(psd.size)
    mis = np.abs(psd).argsort()[-1:-max_n - 2:-1]  # ids_half starts at 1
    psd_idxs[mis] = True
    return psd_idxs * fhat, psd * psd_idxs, mis


def min_pct_change(df):
    return df.groupby(level=1, axis=1).agg(lambda x: x.pct_change(axis=1).idxmin(axis=1).map(lambda y: y[0])).mode(axis=1).iloc[:, 0]


def to_sinewaves(y, nr_frequencies):
    fft3, psd, freqs = decompose(y, len(y), y.index.diff()[1])
    mis = np.abs(psd).argsort()[-2:-nr_frequencies[y.name] - 2:-1]
    res = pd.Series(None, index=pd.MultiIndex.from_product([mis, y.index], names=['frequency', y.index.name]))
    for i in mis:
        res[i] = to_sinewave(abs(fft3[i]), np.angle(fft3[i]), freqs[i], y.index.to_numpy())
    return res


def to_sinewave(amplitude, phase, freq, x):
    return 1 / len(x) * amplitude * np.cos(freq * 2 * np.pi * x + phase)


def get_increased_activity(activity, n=64):
    max_locs = fftconvolve(np.hstack([activity, activity.iloc[:, :n+1]]).T, np.ones((n+1,1), dtype=float), mode="valid", axes=0).argmax(axis=0)
    return pd.Series(activity.columns[max_locs], index=activity.index, name="onset of increased activity")


def get_increased_activity_with_end(onset, t=8):
    return pd.DataFrame.from_dict({'onset': onset, 'end': time_past_t(onset, t)})


def find_extrema(df, comparator, name='activity'):
    idx_x, idx_y = argrelextrema(df.to_numpy(), comparator, mode="wrap", axis=1)
    return pd.Series(df.values[(idx_x, idx_y)], index=pd.MultiIndex.from_arrays([df.index[idx_x], df.columns[idx_y]], names=[df.index.name, df.columns.name]), name=name)


def extremum_with_hrs_past_waking(d, comparator, shift_dict, ascending=False):
    stats = find_extrema(d, comparator, name='activity').reset_index(level=Columns.MIN_BINS15.value)
    stats.columns=['clock time', 'activity/ratio']
    stats['hours past waking'] = stats[['clock time']].apply(lambda v: (v-shift_dict[v.name]) % 24, axis=1)
    stats.sort_values(by='activity/ratio', ascending=ascending, inplace=True)
    stats.set_index(stats.groupby('cluster')['activity/ratio'].transform(lambda v: list(range(0, len(v)))), append=True, inplace=True)
    stats.index.names=[stats.index.names[0], 'i']
    return stats[['clock time', 'hours past waking', 'activity/ratio']]


def signal_extrema(df, shift_dict):
    return pd.concat([
        extremum_with_hrs_past_waking(df, np.greater, shift_dict, ascending=False), 
        extremum_with_hrs_past_waking(df, np.less, shift_dict, ascending=True)
        ], keys=['max', 'min'], axis=1).sort_index()


def style_extrema_stats_df(shift_by, index_slice=pd.IndexSlice[:], rename_dict=dict(), **signalkwargs):
    return (pd.concat([signal_extrema(res.recomposed_signal, shift_by) for res in signalkwargs.values()], keys=signalkwargs.keys())
        .loc[index_slice]
        .rename(**rename_dict)
        .style
        .format(lambda time: format_h_min(time, type="digital"), na_rep="-", subset=pd.IndexSlice[:, pd.IndexSlice[:, "clock time"]])
        .format(lambda time: format_h_min(time, type="duration"), na_rep="-", subset=pd.IndexSlice[:, pd.IndexSlice[:, "hours past waking"]])
        .format('{:.3f}', na_rep="-", subset=pd.IndexSlice[:, pd.IndexSlice[:, "activity/ratio"]])
        .hide(level='i')
        )


def style_dip_df(heightened_activity, rename_dict=dict(), order=None, **signalkwargs):
    if order  is None:
        order = heightened_activity.index
    return (pd.concat([
            *(t.apply(diptest, axis=1, result_type='expand') for t in signalkwargs.values()), 
            get_increased_activity_with_end(heightened_activity)
        ], keys=[*signalkwargs.keys(), 'heightened activity'], axis=1)
        .loc[Clusters.order()]
        .rename(**rename_dict)
        .style
        .format('{:.3f}' )
        .format(format_h_min, na_rep="-", subset="heightened activity")
        .map(lambda v: 'font-weight: bold;' if (v <0.05)  else None, subset=pd.IndexSlice[:, pd.IndexSlice[:, "\pvalue"]])
    )

def apply_mwu(df, order=None):
    order=df.index if order is None else order
    return df.loc[order].apply(
        lambda x: df.loc[order].apply(
            lambda y: pd.Series(
                mannwhitneyu(x,y, alternative='less') if order.index(x.name) < order.index(y.name) else None, index=["Statistic", "\pvalue"]
            ), axis=1).stack(), 
        axis=1, result_type='expand').loc[order[:-1], order[1:]]


def style_mwu_comparison_df(order=None, rename_dict=dict(), **signalkwargs):
    return (pd.concat([apply_mwu(df=s, order=order) for s in signalkwargs.values()], keys=signalkwargs.keys())
        .rename(**rename_dict)
        .style
        .format('{:,.1e}', na_rep="-")
        .format('{:,.0f}', na_rep="-", subset=pd.IndexSlice[:, pd.IndexSlice[:, "Statistic"]])
        .map(lambda v: 'font-weight: bold;' if (v <0.05)  else None, subset=pd.IndexSlice[:, pd.IndexSlice[:, "\pvalue"]])
    )

def plot_sinewaves(sinwaves, annotate_with=None, col_order=Clusters.total_order()):
    g = sns.relplot(
        data=sinwaves.stack(level=(0,1)).reset_index(), 
        col=Columns.CLUSTER.value, col_order=col_order, 
        hue='frequency', palette=sns.color_palette("crest", sinwaves.columns.levels[0].nunique()), 
        x=Columns.MIN_BINS15.value, y=0, kind="line")
    g.set_titles("{col_name}")
    g.set_axis_labels("", "")
    if annotate_with is not None:
        for ax, c in zip(g.axes.flat, col_order):
            if c in annotate_with:
                ax.set_title(f"{c}: {annotate_with[c]} frequencies")
    return g


def annotate_points(ax, stat_df, ha_cutoff, move_cutoff, ticks_format):
    get_diff= lambda i: stat_df.index[i] - stat_df.index[i-1]
    def get_ha_offset(i):
        if i > 0 and (diff:=get_diff(i)) < ha_cutoff:
            return 'left', diff if diff <=move_cutoff else 0
        elif i < len(stat_df)-1 and (diff:=get_diff(i+1)) < ha_cutoff:
            return 'right', -diff if diff <=move_cutoff else 0
        else:
            return 'center', 0

    for i in range(len(stat_df)):
        ha, offset = get_ha_offset(i)
        ax.annotate(
            text=ticks_format(stat_df.index[i]), 
            xy=(stat_df.index[i], stat_df.iloc[i]*1.05), xytext=(stat_df.index[i]+offset, stat_df.iloc[i]*1.05),
            rotation=90, ha=ha, zorder=2,
            bbox=dict(boxstyle="round,pad=0.1,rounding_size=0.2", fc="white", alpha=.6, zorder=1))


def hatch_groups(ax, max_vals, group_nr, x, hue):
    sorted_groups = max_vals.sort_index(level=x, ascending=False).groupby(level=hue)
    i = 0
    while i < group_nr:
        x1, x2 = sorted_groups.apply(lambda r: r.index.get_level_values(x)[i] if len(r) > i else None).agg(['min', 'max'])
        ax.axvspan(x1, x2, ymin=0, ymax=1, fc="none", ec='lightgray', alpha=1, lw=0, zorder=0, hatch="////")
        i += 1


def annotate_maxima(ax, df, prep_df, annotate, x, hue, palette, t_cutoff, ticks_format, hatch):
    max_vals = find_extrema(df, np.greater).sort_values(ascending=False).groupby(level=hue, sort=False).head(annotate)
    max_vals.loc[max_vals.index] = prep_df.loc[max_vals.index]
    ax.vlines(data=max_vals.reset_index(), x=x, ymin=0, ymax="activity", colors=max_vals.index.get_level_values(hue).map(palette), linestyle="dashed", label="")
    sns.scatterplot(max_vals.reset_index(), x=x, y="activity", color="black", ax=ax, zorder=5)
    annotate_points(ax, max_vals.droplevel(hue).sort_index(), *t_cutoff, ticks_format)
    hatch and hatch_groups(ax, max_vals, annotate, x, hue)


def plot_signal_and_recombined(line_df, scatter_df, x, hue, palette, annotate=2, hatch=False, t_cutoff=(1,.25), 
        ampm="ampm", prepare=lambda df: df, ax = None, scatter_kwargs = {}, line_kwargs = {}, **fig_kwargs):
    fig, ax = plt.subplots(**fig_kwargs) if ax is None else (ax.get_figure(), ax)
    ticks_format = lambda val, *args, **kwargs: format_h_min(val, ampm)
    
    prep_line = prepare(line_df).stack()
    prep_scatter = prepare(scatter_df).stack()
    sns.lineplot(prep_line.reset_index(), x=x, y=0, hue=hue, palette=palette, legend=False, ax=ax, **(dict(lw=2) | line_kwargs ))
    sns.scatterplot(prep_scatter.reset_index(), x=x, y=0, hue=hue, palette=palette, legend=False, ax=ax, **(dict(s=12) | scatter_kwargs ))
    annotate > 0 and annotate_maxima(ax, line_df, prep_line, annotate, x, hue, palette, t_cutoff, ticks_format, hatch)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(6))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(ticks_format))
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xlim(line_df.columns[0], line_df.columns[-1])
    ax.set_ylim(0, prep_scatter.max())
    return fig, ax


def plot_with_cumsum(line_df, scatter_df, axes, prepare, cumsum, **kwargs):
    plot_signal_and_recombined(line_df, scatter_df, ax=axes[0], prepare=prepare, **kwargs)
    cumsum is not None and plot_signal_and_recombined(line_df, scatter_df, ax=axes[1], prepare=cumsum, **kwargs)


def plot_multiple(line_df, scatter_df, righthand=None, hatch=[], cumsum=None, normalize=False, figsize=None, **kwargs):
    dims = sorted((1+(cumsum is not None), 1 + (righthand is not None)))
    if figsize is None:
        figsize=[a*b for a,b in zip([2.5, 3.5], dims)][::-1]
    fig, axes = plt.subplots(*dims, figsize=figsize, sharey='row', sharex='col', tight_layout=True)
    prepare=lambda df: df / df.iloc[0].mean() if normalize else df
    plot_with_cumsum(line_df, scatter_df, axes.flat[[0, (cumsum is not None)+(righthand is not None)]], prepare, cumsum, ampm="ampm", hatch='left' in hatch, **kwargs)
    if righthand is not None:
        plot_with_cumsum(righthand(line_df), righthand(scatter_df), axes.flat[[1,-1]], prepare, cumsum, ampm="durationvarsize", hatch='right' in hatch, **kwargs)
    return fig, axes


def plot_recomposition(recomposition_result, increased_activity, index_slice=pd.IndexSlice[:], title="", cumsum=None, normalize=False, **kwargs):
    g, _ = plot_multiple(
        recomposition_result.recomposed_signal.loc[index_slice], recomposition_result.signal.loc[index_slice],
        righthand=lambda df: shift_rows_by(df, increased_activity), 
        cumsum=cumsum, normalize=normalize,
        **kwargs
    )
    g.suptitle(title)
    return g


def plots_to_store(signal, recomposed_signal, increased_activity, index_slice=pd.IndexSlice[:], normalize=False, hatch=[], cumsum=None, **kwargs):
    prepare=lambda df: df / df.iloc[0].mean() if normalize else df
    fig1, _ = plot_signal_and_recombined(
        recomposed_signal.loc[index_slice], 
        signal.loc[index_slice], 
        ampm="ampm",  
        prepare=prepare, 
        hatch="left" in hatch, 
        **kwargs)

    fig2, _ = plot_signal_and_recombined(
        shift_rows_by(recomposed_signal.loc[index_slice], increased_activity), 
        shift_rows_by(signal.loc[index_slice], increased_activity), 
        ampm="durationvarsize", 
        prepare=prepare, 
        hatch="right" in hatch, 
        **kwargs)
    return fig1, fig2



def disinf_ratio(signal, std=3, padding=3, frequency_nrs=np.arange(2,5), **kwargs):
    if std is not None and padding is not None:
        signal_smoothed = smooth_looped(signal, std=std, padding=padding)
        recomposition_result = recompose(signal_smoothed, frequency_nrs)
        recomposition_result.signal_smoothed = recomposition_result.signal.copy()
        recomposition_result.signal = signal
    return recomposition_result


def activity(signal, frequency_nrs=np.arange(2,5), n=64, **kwargs):
    recomposition_result = recompose(signal, frequency_nrs)
    recomposition_result.increased_activity = get_increased_activity(recomposition_result.recomposed_signal, n=n)
    return recomposition_result


class FourierRoutine():
    
    def __init__(self, config, processor = None, **kwargs):
        if processor is None:
            processor = DataProcessor(config, columns=[Columns.POSTS.value])
        self.processor = processor
        self.config=config
        self.shared_plot_kwargs = dict(
            x=Columns.MIN_BINS15.value, 
            hue=Columns.CLUSTER.value, 
            palette=Clusters.palette(), 
        ) | kwargs

    def routine_by(self, label, clustertype, activity, disinf_ratio, disinf_activity, 
        activity_kwargs = dict(cumsum = lambda df: df.cumsum(axis=1), annotate=2, normalize=True),
        save_plots=False,
        **kwargs):

        for fourier_recomposition, kw, plottype, title in zip(
            (activity, disinf_ratio, disinf_activity),
            (activity_kwargs | dict(hatch=[]), dict(annotate=1), activity_kwargs),
            ('activity', 'disinf_ratio', 'disinf_activity'),
            (f'Activity by {label}', f'Ratio of potentially disinformative content by {label}', f'Potentially disinformative activity by {label}')
        ):
            fourier_recomposition.figure = plot_recomposition(fourier_recomposition, activity.increased_activity, title=title, **(self.shared_plot_kwargs | kwargs | kw))
            if save_plots:
                g1, g2 = plots_to_store(fourier_recomposition.signal, fourier_recomposition.recomposed_signal, activity.increased_activity, **(self.shared_plot_kwargs | kwargs | kw))
                save_plot(g1, f'{plottype}_by_{label}_fourier_clocktime', self.config, clustertype)
                save_plot(g2, f'{plottype}_by_{label}_fourier_waking', self.config, clustertype)
        plt.close('all')
        return activity, disinf_ratio, disinf_activity


    def by_user(self, clustertype, **kwargs):
        return self.routine_by(
            'user', 
            clustertype, 
            self.activity_by_user(clustertype, **kwargs), 
            self.disinf_ratio_by_user(clustertype, **kwargs), 
            self.disinf_activity_by_user(clustertype, **kwargs), 
            **kwargs)


    def by_tweet(self, clustertype, **kwargs):
        return self.routine_by(
            'tweet', 
            clustertype, 
            self.activity_by_tweet(clustertype, **kwargs), 
            self.disinf_ratio_by_tweet(clustertype, **kwargs), 
            self.disinf_activity_by_tweet(clustertype, **kwargs),
            **kwargs)


    def disinf_ratio_by_user(self, clustertype, **kwargs):
        return disinf_ratio(
            self.processor.get_per_cluster_x(clustertype, self.processor.user_disinf_ratio_t, 'mean', columns=[Columns.MIN_BINS15.value]).unstack(level=Columns.MIN_BINS15.value), 
            **kwargs)
    
    
    def disinf_ratio_by_tweet(self, clustertype, **kwargs):
        return disinf_ratio(
            self.processor.get_ratio_by_tweet_x(clustertype, self.processor.disinf_posts_per_user_t, self.processor.known_posts_per_user_t, Columns.MIN_BINS15.value).unstack(level=Columns.MIN_BINS15.value),
            **kwargs)
    

    def activity_by_user(self, clustertype, **kwargs):
        return activity(self.processor.get_per_cluster_x(clustertype, self.processor.user_activity, 'mean'), **kwargs)


    def activity_by_tweet(self, clustertype, **kwargs):
        posts_per_cluster_t = self.processor.get_per_cluster_x(clustertype, self.processor.posts_per_user_t, 'sum', columns=[Columns.MIN_BINS15.value]).unstack(level=Columns.MIN_BINS15.value, fill_value=0)
        return activity(posts_per_cluster_t.div(posts_per_cluster_t.sum(axis=1), axis=0), **kwargs)


    def disinf_activity_by_user(self, clustertype, **kwargs):
        return activity(self.processor.get_per_cluster_x(clustertype, self.processor.disinf_user_activity, 'mean'), **kwargs)


    def disinf_activity_by_tweet(self, clustertype, **kwargs):
        disinf_posts_per_cluster_t = self.processor.get_per_cluster_x(clustertype,  self.processor.disinf_posts_per_user_t, 'sum', columns=[Columns.MIN_BINS15.value]).unstack(level=Columns.MIN_BINS15.value, fill_value=0)
        return activity(disinf_posts_per_cluster_t.div(disinf_posts_per_cluster_t.sum(axis=1), axis=0), **kwargs)