import pandas as pd
from datetime import datetime
from scipy.stats import mannwhitneyu

from .utils import groupby_with_total
from .path_utils import save_to_latex
from .enums import Columns, ContentType, Clusters
from .data_processor import DataProcessor, disinformative_mapping

def str_to_date(datestr, format='%Y-%m-%d'):
    return datetime.strptime(datestr, format).date()

class LockdownRoutine():
    
    def __init__(
        self, 
        config, 
        processor = None,
        lockdown_start = '2020-03-09',
        lockdown_end = '2020-05-18',
        columns=[Columns.POSTS.value],
    ):
        days_in_lockdown = (str_to_date(lockdown_end) - str_to_date(lockdown_start)).days
        self.lockdown_timespan = pd.Series([
            days_in_lockdown, (str_to_date(config.date_to) - str_to_date(config.date_from)).days - days_in_lockdown
        ], index=[True, False], name=Columns.LOCKDOWN.value)
        self._posts_per_user_lockdown_t = None
        if processor is None:
            processor = DataProcessor(config, columns=columns)
        self.processor = processor
        self.config = config


    @property
    def posts_per_user_disinf_lockdown_t(self):
        if self._posts_per_user_lockdown_t is None:
            self._posts_per_user_lockdown_t = groupby_with_total(self.processor.known_data[Columns.POSTS.value], 
                [Columns.MIN_BINS15.value, Columns.USERHASH.value, Columns.LOCKDOWN.value, disinformative_mapping(self.processor.known_data)],
                'sum', total_for_idx=-1
            ).unstack(level=-1, fill_value=0)
        return self._posts_per_user_lockdown_t
    

    def users_per_cluster(self, clustertype='all', post_type=ContentType.KNOWN.value):
        return self.processor.get_per_cluster_x(clustertype, self.processor.posts_per_user_ft.loc[self.processor.posts_per_user_ft[post_type] > 0, post_type], 
                                      lambda x: x.index.get_level_values(Columns.USERHASH.value).nunique())


    def disinf_by_user_cluster_lockdown_t(self, clustertype='all'):
        ratios_per_user_col_t=self.posts_per_user_disinf_lockdown_t[True].div(self.posts_per_user_disinf_lockdown_t.sum(axis=1), axis=0)
        return self.processor.get_per_cluster_x(clustertype, ratios_per_user_col_t, 'mean', columns=[Columns.MIN_BINS15.value, Columns.LOCKDOWN.value])
        

    def disinf_by_tweet_cluster_lockdown_t(self, clustertype='all'):
        posts_per_cluster_col = self.processor.get_per_cluster_x(clustertype, self.posts_per_user_disinf_lockdown_t, 'sum', columns=[Columns.MIN_BINS15.value, Columns.LOCKDOWN.value])
        return posts_per_cluster_col[True].div(posts_per_cluster_col.sum(axis=1), axis=0)


    def mwu_table(self, clustertype):
        to_mwu = lambda r: pd.Series(mannwhitneyu(r.xs(True, level=Columns.LOCKDOWN.value), r.xs(False, level=Columns.LOCKDOWN.value), alternative='less'), index=['statistic', '\pvalue'])
        return (pd.concat([
            self.disinf_by_user_cluster_lockdown_t(clustertype).groupby(level=0).apply(to_mwu), 
            self.disinf_by_tweet_cluster_lockdown_t(clustertype).groupby(level=0).apply(to_mwu)
        ], axis=1, keys=['by user', 'by tweet']).unstack(level=-1).loc[Clusters.total_order()]
        .style
        .format('{:.1e}', subset=pd.IndexSlice[:, (slice(None), '\pvalue')])
        .format('{:,.0f}', subset=pd.IndexSlice[:, (slice(None), 'statistic')])
        .map(lambda v: 'font-weight: bold;' if (v <0.05)  else None, subset=pd.IndexSlice[:, (slice(None), '\pvalue')])
        )
    
    def change_during_lockdown(self, clustertype='all', save=False):
        posts_per_cluster_lockdown_disinf = self.processor.get_per_cluster_x(clustertype, self.posts_per_user_disinf_lockdown_t[[True, 'total']], 'sum', columns=[Columns.LOCKDOWN.value])
        cluster_posts_per_day_user = posts_per_cluster_lockdown_disinf.div(self.users_per_cluster(clustertype), axis=0).div(self.lockdown_timespan, level=Columns.LOCKDOWN.value, axis=0)

        posts_per_user_disinf_lockdown = self.posts_per_user_disinf_lockdown_t.groupby(level=[Columns.USERHASH.value, Columns.LOCKDOWN.value]).sum()
        ratio_per_user_disinf_lockdown = posts_per_user_disinf_lockdown[True].div(posts_per_user_disinf_lockdown.sum(axis=1), axis=0)
        ratio_by_user_per_cluster_disinf_lockdown = self.processor.get_per_cluster_x(clustertype, ratio_per_user_disinf_lockdown, 'mean', columns=[Columns.LOCKDOWN.value])

        lockdown_stat_df_style = pd.concat([
            cluster_posts_per_day_user['total'], 
            cluster_posts_per_day_user[True], 
            posts_per_cluster_lockdown_disinf[True].div(posts_per_cluster_lockdown_disinf['total'], axis=0),
            ratio_by_user_per_cluster_disinf_lockdown
        ], keys=[
            'posts per day and user', 
            f'{ContentType.DISINFORMATIVE.value} posts per day and user', 
            f'{ContentType.DISINFORMATIVE.value} ratio by tweet', 
            f'{ContentType.DISINFORMATIVE.value} ratio by user'
        ], axis=1).apply(lambda x: (x.loc[:, True]- x.loc[:, False]).div(x.loc[:, False]), axis=0).loc[Clusters.order()].T.style.format('{:.1%}', na_rep="-")

        if save:
            save_to_latex(
                self.config,
                lockdown_stat_df_style,
                f'stats_lockdown_{clustertype}',
                caption=r"Stats during and outside of the lockdown period.",
                is_multi_index=True
            )
        return lockdown_stat_df_style


