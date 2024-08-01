import pandas as pd
import matplotlib.pyplot as plt

from .path_utils import save_plot
from .fourier_utils import FourierRoutine
from .diurnal_plot import diurnal_plot
from .data_processor import DataProcessor
from .enums import Columns, FactTypes, Clusters
from .utils import time_past_t


class DiurnalPlotRoutine():
    
    def __init__(self, config, processor = None, **kwargs):
        if processor is None:
            processor = DataProcessor(config)
        self.processor = processor
        self.fourier_routine = FourierRoutine(config, self.processor)
        self.config=config
        self.shared_plot_kwargs = dict(
            x = Columns.MIN_BINS15.value,
            y_stack = Columns.POSTS.value,
            y_line = Columns.RATIO.value,
            facet = Columns.CLUSTER.value,
            hue = Columns.FACTTYPE.value,
            palette = FactTypes.palette(),
            facet_order = Clusters.order('name'),
            line_order = FactTypes.disinformative_order('name')
         ) | kwargs
        
        self._disinf_ratio_cluster_t_by_user = {}
        self._disinf_ratio_cluster_t_by_tweet = {}
        self._posts_per_cluster_ft_t = {}


    def disinf_ratio_cluster_t_by_user(self, clustertype):
        if clustertype not in self._disinf_ratio_cluster_t_by_user:
            self._disinf_ratio_cluster_t_by_user[clustertype] = self.processor.get_per_cluster_x(clustertype, self.processor.user_disinf_ratio_t, 'mean', columns=[Columns.MIN_BINS15.value])
        return self._disinf_ratio_cluster_t_by_user[clustertype]


    def disinf_ratio_cluster_t_by_tweet(self, clustertype):
        if clustertype not in self._disinf_ratio_cluster_t_by_tweet:
            self._disinf_ratio_cluster_t_by_tweet[clustertype] = self.processor.get_ratio_by_tweet_x(clustertype, self.processor.disinf_posts_per_user_t, self.processor.known_posts_per_user_t, Columns.MIN_BINS15.value)
        return self._disinf_ratio_cluster_t_by_tweet[clustertype]


    def posts_per_cluster_ft_t(self, clustertype):
        if clustertype not in self._posts_per_cluster_ft_t:
            self._posts_per_cluster_ft_t[clustertype] = self.processor.get_per_cluster_x(clustertype, self.processor.known_data[Columns.POSTS.value], 'sum', columns=[Columns.MIN_BINS15.value, Columns.FACTTYPE.value])
        return self._posts_per_cluster_ft_t[clustertype]
    

    async def by_user(self, clustertype, **kwargs):
        return await self.routine(
            clustertype, 
            self.fourier_routine.activity_by_user(clustertype).increased_activity,
            self.processor.get_per_cluster_x(
                clustertype, self.processor.ratio_of_known_per_user_ft_t.unstack(level=Columns.FACTTYPE.value, fill_value=0), 'mean', columns=[Columns.MIN_BINS15.value]
            ).stack(level=Columns.FACTTYPE.value), 
            self.fourier_routine.disinf_ratio_by_user(clustertype), 
            'by user', **kwargs)


    async def by_tweet(self, clustertype, **kwargs):
        return await self.routine(
            clustertype, 
            self.fourier_routine.activity_by_tweet(clustertype).increased_activity,
            self.posts_per_cluster_ft_t(clustertype).div(
                self.posts_per_cluster_ft_t(clustertype).groupby(level=[Columns.CLUSTER.value, Columns.MIN_BINS15.value]).transform('sum')),
            self.fourier_routine.disinf_ratio_by_tweet(clustertype),
            'by tweet', **kwargs)


    async def routine(self, clustertype, onset, ratio_by_cluster_ft_t, disinf_ratio, label, smoothed=False, save = False, **kwargs):
        radial_df = pd.concat([self.posts_per_cluster_ft_t(clustertype), ratio_by_cluster_ft_t], axis=1, keys=['posts', label]).dropna().reset_index()
        increased_activity = pd.DataFrame.from_dict({'from': onset, 'to': time_past_t(onset, 8)})
        merged_kwargs = self.shared_plot_kwargs | kwargs | dict(data=radial_df, y_stack='posts', y_line=label, inner_arc=increased_activity)

        if smoothed:
            merged_kwargs['outer_arc'] = disinf_ratio.recomposed_signal.T
            merged_kwargs['outer_arc_kwargs']=dict(smooth=False)
        else:
            merged_kwargs['outer_arc'] = disinf_ratio.signal.T

        if save:
            await self.save_facets(clustertype, file_postfix=f'{"_smooth" if smoothed else ""}', **merged_kwargs)

        return await diurnal_plot(**merged_kwargs)
    
    async def save_facets(self, clustertype, data, x, y_stack, y_line, facet, facet_order, hue, line_order, file_postfix, **kwargs):
        # the clockface subplots are not compatible with tight layout, hence saving sparately.
        for f in facet_order:
            max_c = data.loc[data[facet].isin(facet_order)].groupby(by=[x, facet])[y_stack].sum().max()
            max_h = data.loc[data[hue].isin(line_order)].groupby(by=[x, facet])[y_line].sum().max()
            subfig, _ = await diurnal_plot(
                title=False, data=data, x=x, y_stack=y_stack, y_line=y_line, facet=facet, facet_order=[f], hue=hue, line_order=line_order,
                **(kwargs | dict(ncols=1, max_c=max_c, max_h = max_h)))
            save_plot(subfig, f'{f}_{y_line}{file_postfix}', self.config, clustertype, subdir="clockface/", dpi=300, transparent=False, bbox_inches='tight')
        plt.close('all')
