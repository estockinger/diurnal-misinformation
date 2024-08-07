import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from dataclasses import dataclass


from .enums import Clusters, Columns
from .utils import shift_rows_by
from .fourier_utils import similarity_measures, FourierRoutine



class SimilarityRoutine(FourierRoutine):

    def __init__(self, config, **kwargs):
        super(SimilarityRoutine, self).__init__(config, **kwargs)

    def by_user(self, clustertype, **kwargs):
        activity_res=self.activity_by_user(clustertype, **kwargs)
        return self.routine_by(
            activity=activity_res.recomposed_signal, 
            disinf_ratio=self.disinf_ratio_by_user(clustertype, **kwargs).recomposed_signal,
            increased_activity= activity_res.increased_activity,
            **kwargs
        )

    def by_tweet(self, clustertype, **kwargs):
        activity_res=self.activity_by_tweet(clustertype, **kwargs)
        return self.routine_by(
            activity=activity_res.recomposed_signal, 
            disinf_ratio=self.disinf_ratio_by_tweet(clustertype, **kwargs).recomposed_signal, 
            increased_activity= activity_res.increased_activity,
            **kwargs
        )

    def routine_by(self, activity, disinf_ratio, increased_activity, index_slice=pd.IndexSlice[:], compare_to=None):
        curve_stats = activity.apply(find_curve_features, axis=1)
        curve_stats['onset increased activity'] = increased_activity
        if compare_to is None:
            compare_to = activity.loc[index_slice].index[0]
        return SignalDistances(
            activity_curve_stats = curve_stats.loc[index_slice],
            distance_activity_by_curve_features = mean_distance_for_shifts(activity.loc[index_slice], curve_stats.loc[index_slice], similarity_measures, compare_to=compare_to),
            distance_ratio_by_activity_curve_features = mean_distance_for_shifts(disinf_ratio.loc[index_slice], curve_stats.loc[index_slice], similarity_measures, compare_to=compare_to),
            g_activity_by_curve_features = plot_daily_activities(activity, shift_by=curve_stats.loc[index_slice], title="User activity aligned by curve features"),
            g_ratio_by_activity_curve_features = plot_daily_activities(disinf_ratio, shift_by=curve_stats.loc[index_slice], title="Ratios of potentially disinformative content aligned by activity curve features")
        )

    
def find_min_distance_shifts(activity, name, distancefunc, compare_to = None):
    if compare_to is None:
        compare_to = activity.index[0]
        
    shift_by = pd.Series(0, index=activity.index, name=name, dtype=float)
    r1 = np.column_stack((activity.loc[compare_to], activity.columns))
    r2 = np.column_stack((np.zeros_like(activity.columns), activity.columns))
    
    for c2 in activity.index.difference([compare_to]):
        min_distance = -1
        for i, b in enumerate(activity.columns):
            r2[:, 0] = np.roll(activity.loc[c2], i)
            dist = distancefunc(r2, r1)
            if min_distance < 0 or min_distance > dist:
                min_distance = dist
                shift_by.loc[c2] = -b%24
    return shift_by


def find_first_inflection(d):
    slopes = d / .25
    inflection_points = np.sign(slopes.diff()).diff()
    inflection_times = inflection_points.loc[inflection_points < 0].index
    return inflection_times.min()- 0.5  # 2 diffs


def find_first_peak(r):
    peaks, _ = find_peaks(r.to_numpy().squeeze(), height=0)
    peak_times = r.iloc[peaks].index
    return peak_times.min()


def find_curve_features(r):
    d = r.diff()
    d.iloc[0] = d.iloc[0] - d.iloc[-1]
    return pd.Series({
        'clocktime': 0,
        'min': r.idxmin(),
        'max': r.idxmax(), 
        'first inflection': find_first_inflection(d[r.idxmin():]), 
        'first peak': find_first_peak(r), 
        'steepest ascent': d.idxmax() - .25 # one diff
    })


def mean_distance(activity, shift_by, distancefunc, compare_to):
    shifted = shift_rows_by(activity, shift_by)
    r1 = np.column_stack((shifted.loc[compare_to], activity.columns))
    return shifted.loc[activity.index!=compare_to, :].apply(lambda r2: distancefunc(r1, np.column_stack((r2, activity.columns))), axis=1).mean()


def plot_daily_activities(
        activity, 
        shift_by,
        palette=Clusters.palette(),
        title="Daily Activities aligned by Curve Features"
):
    def shift_plot(data, x, *args, df, label, color, **kwargs):
        return sns.lineplot(np.roll(df.loc[label].to_numpy(), -df.columns.get_loc(data[x].iloc[0])), color=color)
    g = sns.FacetGrid(
        data=shift_by.melt(ignore_index=False, var_name='type', value_name='t').reset_index(), 
        hue=Columns.CLUSTER.value, 
        col="type", sharey="row", palette=palette)
    
    g.map_dataframe(shift_plot, x='t', df=activity)
    g.set_axis_labels("time")
    g.set_titles(col_template="{col_name}")
    plt.suptitle(title, y=1.02)
    return g

def mean_distance_for_shifts(signal, shift_by, metrics, compare_to):
    return shift_by.apply(
        lambda x: pd.Series([
            mean_distance(signal, x, v, compare_to=compare_to)
            for v in metrics.values()], index=metrics.keys())).T.style.highlight_min(props='font-weight: bold;', axis=0).format('{:.1e}')


@dataclass
class SignalDistances:
    activity_curve_stats: pd.DataFrame
    distance_activity_by_curve_features: pd.DataFrame
    distance_ratio_by_activity_curve_features: pd.DataFrame
    g_activity_by_curve_features:pd.DataFrame
    g_ratio_by_activity_curve_features: pd.DataFrame
