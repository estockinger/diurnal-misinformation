import inspect, math, datetime
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, shapiro
from suntime import Sun
from dataclasses import dataclass

from .data_processor import DataProcessor, disinformative_mapping
from .enums import Clusters, Columns, FactTypes, ContentType
from .utils import within_n_hours_before_or_after_t, t_between, groupby_with_total, weighted_mean
from .fourier_utils import FourierRoutine, get_increased_activity_with_end
from .path_utils import save_plot, save_to_latex
from .heatmap_plots import plot_heatmap_facets

def time_to_bin(time: datetime.date):
    timeobj = time.time()
    return (((timeobj.hour*3600)+(timeobj.minute*60))//900)*0.25
    

def to_sunset_sunrise(row):
    date = datetime.date(year=row.name[1], month=row.name[2], day=1)
    sun = Sun(*row)
    return pd.Series([sun.get_local_sunrise_time(date), sun.get_local_sunset_time(date)], index=['sunrise', 'sunset'])


def disinf_ratio_to_shapio(df):
    return (df.stack().to_frame()
        .groupby(level=Columns.CLUSTER.value)
        .apply(lambda x: pd.Series(shapiro(x), index=['statistic', '\pvalue']))
        .loc[Clusters.total_order()]
        .style.format('{:.3f}', subset='statistic')
        .format('{:.1e}', subset='\pvalue')
        .map(lambda v: 'font-weight: bold;' if (v <0.05)  else None, subset='\pvalue'))


def pair_column(df_day, df_night, test = mannwhitneyu, significant=.05):
    stat, p = test(df_day, df_night)
    if 'alternative' in inspect.signature(test).parameters.keys():
        _, pgreater = test(df_day, df_night, alternative='greater')
        _, pless = test(df_day, df_night, alternative='less')
        if pless < pgreater and pless < significant:
            return stat, pless, 'day'
        elif pgreater < pless and pgreater < significant:
            return stat, pgreater, 'night'
        return stat, p, '-'
    return stat, p


def for_columns(r, time_categories):
    def for_time_categories(t):
        return pd.Series(pair_column(r.loc[t == 'day'], r.loc[t=='night']), index=['statistic', '\pvalue', 'less'])
    return time_categories.apply(for_time_categories, result_type='expand').unstack()


def day_night_safety(t, daystart, dayend, safety_margin=(1, 1), day='day', night='night', safety='safety'):
    if (safety_margin[0] > 0 and within_n_hours_before_or_after_t(t, daystart, safety_margin[0], include_edges=True)) or \
            (safety_margin[1] > 0 and within_n_hours_before_or_after_t(t, dayend, safety_margin[1], include_edges=True)):
        return safety
    if t_between(t, daystart, dayend, include_edges=False):
        return day
    return night


def to_time_scales(cluster, year, month, min_bins15, sr, **kwargs):
    return pd.Series([
        day_night_safety(min_bins15, *times, **kwargs) for times in 
        [sr.clocktimes.loc[cluster], sr.suntimes.loc[(cluster, year, month)], sr.increased_activity.loc[cluster]]
        ], index=['clock', 'sun', 'waking'])
 

class DayNightRoutine():
    
    def __init__(
        self, 
        config, 
        processor = None,
        timecolumns=[Columns.YEAR.value, Columns.MONTH.value, Columns.MIN_BINS15.value],
        columns=[Columns.POSTS.value, Columns.LAT.value, Columns.LONG.value],
        **kwargs
    ):
        palette=kwargs.pop('palette', ["lightblue", "coral", "darkred"])
        self.shared_plot_kwargs = dict(
            facecolor='lightslategrey', 
            edgecolor="black", 
            cmap=sns.color_palette("blend:"+",".join(palette), as_cmap=True), 
            vmin=0, vmax=1) | kwargs
        self.timecolumns = timecolumns
        self._disinf_ratio_time_user = None
        if processor is None:
            processor = DataProcessor(config, columns=columns)
        self.config = config
        self.processor = processor
        self.fourier_routine = FourierRoutine(config, processor)
    

    @property
    def disinf_ratio_time_user(self):
        if self._disinf_ratio_time_user is None:
            posts_time_user = groupby_with_total(
                self.processor.known_data[Columns.POSTS.value],
                [*self.timecolumns, Columns.USERHASH.value, disinformative_mapping(self.processor.known_data)],
                'sum', total_for_idx=-1
            )
            self._disinf_ratio_time_user = posts_time_user.xs(True, level=-1).div(posts_time_user.xs('total', level=-1), axis=0, fill_value=0)
        return self._disinf_ratio_time_user


    def by_user(self, clustertype, maxnacount=96):
        latlong_user = weighted_mean(
            self.processor.known_data, 
            [Columns.LAT.value, Columns.LONG.value], 
            Columns.POSTS.value, 
            [Columns.USERHASH.value, Columns.YEAR.value, Columns.MONTH.value])
        latlong = self.processor.get_per_cluster_x(clustertype, latlong_user, 'mean', columns=[Columns.YEAR.value, Columns.MONTH.value])
        
        disinf_ratio = self.processor.get_per_cluster_x(
            clustertype, 
            self.disinf_ratio_time_user, 
            'mean', 
            columns=self.timecolumns
        ).unstack(level=Columns.MIN_BINS15.value)
        
        posts_per_user_time_ft = self.processor.known_data[Columns.POSTS.value].groupby([Columns.USERHASH.value, *self.timecolumns, Columns.FACTTYPE.value]).sum().unstack(level=Columns.FACTTYPE.value, fill_value=0)
        ratio_per_user_time_ft = posts_per_user_time_ft.div(posts_per_user_time_ft.sum(axis=1), axis=0)
        
        return self.routine(
            clustertype, 
            latlong, 
            disinf_ratio, 
            self.fourier_routine.activity_by_user(clustertype), 
            self.processor.get_per_cluster_x(clustertype, ratio_per_user_time_ft, 'mean', columns=self.timecolumns),
            maxnacount=maxnacount
        )
    

    def by_tweet(self, clustertype, maxnacount=96):
        latlong = pd.concat([
            weighted_mean(
                self.processor.known_data, 
                [Columns.LAT.value, Columns.LONG.value], 
                Columns.POSTS.value, 
                [self.processor.cluster_mapping(self.processor.known_data, clustertype), Columns.YEAR.value, Columns.MONTH.value]
            ), pd.concat({'total': weighted_mean(
                self.processor.known_data, 
                [Columns.LAT.value, Columns.LONG.value], 
                Columns.POSTS.value, 
                [Columns.YEAR.value, Columns.MONTH.value])}, names=[Columns.CLUSTER.value]
            )])

        disinf_ratio = self.processor.get_ratio_by_tweet_x(
            clustertype, 
            self.processor.known_data.loc[disinformative_mapping(self.processor.known_data), Columns.POSTS.value].unstack(level=Columns.MIN_BINS15.value), 
            self.processor.known_data[Columns.POSTS.value].unstack(level=Columns.MIN_BINS15.value), 
            *self.timecolumns[:-1], fill_value=0, axis=1)        

        posts_per_cluster = self.processor.get_per_cluster_x(clustertype, self.processor.known_data[Columns.POSTS.value],  'sum', columns=[*self.timecolumns, Columns.FACTTYPE.value])
        ft_ratio = posts_per_cluster.div(posts_per_cluster.groupby(level=[Columns.CLUSTER.value, *self.timecolumns]).sum(), fill_value=0)

        
        return self.routine(
            clustertype, 
            latlong, 
            disinf_ratio, 
            self.fourier_routine.activity_by_tweet(clustertype), 
            ft_ratio.unstack(level=Columns.FACTTYPE.value, fill_value=0), 
            maxnacount=maxnacount
        )


    def get_na_mask(self, df, filter_indices):
        return df.loc[~df.index.droplevel([i for i in df.index.names if i not in self.timecolumns[:-1]]).isin(filter_indices)]
    

    def routine(self, clustertype, latlong, disinf_ratio, activity_recomposition_result, ft_ratio, maxnacount):
        filter_indices = disinf_ratio.loc['total'].loc[disinf_ratio.loc['total'].isna().sum(axis=1) > maxnacount].index
        
        suntimes = self.get_na_mask(latlong, filter_indices).apply(to_sunset_sunrise, axis=1, result_type="expand").map(lambda x: x.tz_localize('utc').astimezone(self.config.TIME_ZONE)).map(time_to_bin)
        return SunResult(
            label = clustertype,
            suntimes = suntimes,
            clocktimes=(suntimes.groupby(level=Columns.CLUSTER.value).mean() * 4).astype(int) / 4,
            increased_activity=get_increased_activity_with_end(activity_recomposition_result.increased_activity),
            disinf_ratio = self.get_na_mask(disinf_ratio, filter_indices),
            ft_ratio=self.get_na_mask(ft_ratio, filter_indices)
        )
    

    def plot_ft_heatmap(self, sun_result, for_cluster='total', **kwargs):
        return self.plot_heatmap(
            sun_result.ft_ratio.xs(for_cluster, level=Columns.CLUSTER.value).unstack(level=Columns.MIN_BINS15.value).stack(Columns.FACTTYPE.value), 
            sun_result.suntimes.loc[for_cluster], 
            sun_result.increased_activity.loc[for_cluster], 
            sun_result.clocktimes.loc[for_cluster], 
            **(dict(
                facets=FactTypes.known_order('name'), 
                facet_level=Columns.FACTTYPE.value,
                label=sun_result.label,
                subtitles = FactTypes.known_order('name')
            ) | self.shared_plot_kwargs | kwargs))


    def plot_disinf_heatmap(self, sun_result, **kwargs):
        return self.plot_heatmap(
            sun_result.disinf_ratio, 
            sun_result.suntimes, 
            sun_result.increased_activity, 
            sun_result.clocktimes,  
            **(dict(facets=Clusters.order(), facet_level=Columns.CLUSTER.value, label=sun_result.label) | self.shared_plot_kwargs | kwargs))


    def plot_heatmap(self, df, suntimes, increased_activity, clocktimes, facets, facet_level, save=False, label="", nrows=1, facet_height=5, facet_width=6, **kwargs):
        if facets is not None:
            df = df.loc[df.index.get_level_values(facet_level).isin(facets)]
        ncols = math.ceil(len(facets)/nrows)

        fig, _ = plot_heatmap_facets(
            df,
            facets,
            facet_level=facet_level,
            sun_times=suntimes,
            waking_times=increased_activity, 
            clock_times=clocktimes,
            nrows=nrows, ncols=ncols,
            figsize=(ncols*facet_width, nrows*facet_height), 
            **kwargs)
        
        if save:
            for i, c in enumerate(facets):
                wi = i%ncols
                hi = nrows - (i // (ncols))
                b = (wi*facet_width, (hi-1)*facet_height, (wi+1)*facet_width, hi*facet_height)
                save_plot(fig, c, self.config, label, subdir='heatmap/',  dpi=300, bbox_inches=mpl.transforms.Bbox.from_extents(b))
        return fig


    def compare_day_night(self, sun_result, save=False):
        df = pd.concat([sun_result.ft_ratio, sun_result.disinf_ratio.stack().rename(ContentType.DISINFORMATIVE.value)], axis=1)
        time_categories = df.apply(lambda r: to_time_scales(*r.name, sun_result), axis=1, result_type='expand')

        mwu_style = (
            df
            .groupby(level=Columns.CLUSTER.value)
            .apply(lambda d: d.apply(for_columns, time_categories=time_categories))
            .T.stack(level=0)
            .loc[pd.IndexSlice[[ContentType.DISINFORMATIVE.value] + FactTypes.disinformative_order('name'), Clusters.order(), :]]
            .rename(columns={
                'clock': r'6:30am - 6:45pm\tnote{1}',
                'sun': r'sunrise - sunset\tnote{2}',
                'waking': r'waking - bedtime\tnote{3}',
                'statistic': '$U$',
            }, index={
                i: i.split(' ')[0] for i in Clusters.order()
            })
            .style
            .format('{:.1e}', subset=pd.IndexSlice[:, (slice(None), '\pvalue')])
            .format('{:,.0f}', subset=pd.IndexSlice[:, (slice(None), '$U$')])
            .map(lambda v: 'font-weight: bold;' if (v <0.05)  else None, subset=pd.IndexSlice[:, (slice(None), '\pvalue')])
        )
        if save:
            save_to_latex(
                self.config,
                mwu_style,
                f'mannwhitneyu_daynight_timetypes_{sun_result.label}',
                caption = 'Mann-Whitney U test comparing the distributions of content type ratios during different time periods: daytime and nighttime, times between and outside of sunrise and sunset, as well as regular and prolonged wakefulness times.',
                is_multi_index=True
            )
        return mwu_style


@dataclass
class SunResult:
    label: str
    suntimes: pd.DataFrame
    clocktimes: pd.DataFrame
    increased_activity: pd.DataFrame
    disinf_ratio: pd.DataFrame
    ft_ratio: pd.DataFrame