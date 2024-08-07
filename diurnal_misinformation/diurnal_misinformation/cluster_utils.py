import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import Callable

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from yellowbrick.cluster.elbow import distortion_score
import validclust.indices as vclustindices

from .enums import Clusters, Columns
from .utils import highlight_diag, smooth_looped
from .data_processor import DataProcessor
from .path_utils import save_to_latex



@dataclass
class ClusterIndicator:
    func: Callable
    maxmin: Callable


def performance_indicator_functions(metric="euclidean", indices=None):
    indicators = {
        'silhouette': ClusterIndicator(vclustindices._silhouette_score2, pd.Series.idxmax),
        'calinski': ClusterIndicator(vclustindices._calinski_harabaz_score2, pd.Series.idxmax),
        'davies': ClusterIndicator(vclustindices._davies_bouldin_score2, pd.Series.idxmin),
        'dunn': ClusterIndicator(vclustindices._dunn, pd.Series.idxmax),
        'cop': ClusterIndicator(lambda data, *args, **kwargs: vclustindices.cop(data.values, *args, **kwargs), pd.Series.idxmin),
        'distortion': ClusterIndicator(lambda data, dist, labels, *args, **kwargs: distortion_score(X=data, labels=labels, metric=metric), pd.Series.idxmax)
    }
    if indices is not None:
        return {k:v for k,v in indicators.items() if k in indices}
    return indicators



def get_clustering_scores(data, methods, metric="euclidean", ks=list(range(2,4)), indices=None):
    # small exerpt from validclust: https://validclust.readthedocs.io/en/latest/_modules/validclust/indices.html,
    # not used directly as it is outdated.
    indicators = performance_indicator_functions(metric, indices)
    
    dist = pairwise_distances(data)
    np.fill_diagonal(dist, 0)

    scores_df = pd.DataFrame(
        index=pd.MultiIndex.from_product([methods, indicators], names=['method', 'indicator']), 
        columns=ks, dtype=np.float64
    )

    for k in ks:
        for c_name, c_alg in methods.items():
            c_alg.set_params(n_clusters=k)
            labels = c_alg.fit_predict(data)
            for i, ci in indicators.items():
                scores_df.loc[(c_name, i), k] = ci.func(data, dist, labels)

    return scores_df


def best_result_per_indicator(scores_df):
    aggmap = {k: f.maxmin for k, f in performance_indicator_functions(scores_df.index.levels[1]).items()}
    return scores_df.unstack(level=0).agg(aggmap, axis=1)


def refit_predict(method, n_cluster, methods, data, cluster_indices=None):
    if cluster_indices is None:
        cluster_indices = data.index
    clustering_instance = methods[method]
    clustering_instance.set_params(n_clusters=n_cluster)
    labels = clustering_instance.fit_predict(data)
    clustercol = pd.Series(n_cluster, index=cluster_indices)
    clustercol.loc[data.index] = labels
    return clustering_instance, clustercol


def cluster_num_to_name(cluster_centers, order_only=False):
    by_time_of_day = np.insert(np.argsort((np.argmax(cluster_centers, axis=1) - 10) % 96), 0, len(cluster_centers))
    if order_only:
        return by_time_of_day
    return dict(zip(by_time_of_day, Clusters.order()))


def get_cluster_stats(posts_per_user, clustercol):
    stats=pd.DataFrame(data=clustercol.value_counts())
    stats.columns=['number of users']
    stats['number of posts'] = posts_per_user.groupby(clustercol).sum()
    stats['posts per user'] = posts_per_user.groupby(clustercol).mean()
    return stats


def get_distance_stats(cluster_centers, rhythms, clustercol, transform, separate_indices=[]):
    dists = euclidean_distances(cluster_centers)
    clustered_mask = clustercol.loc[(~clustercol.isna()) & (~clustercol.isin(separate_indices))].index
    c_dist = transform(rhythms.loc[clustered_mask])
    for i in range(cluster_centers.shape[0]):
        if i in separate_indices:
            dists[i, i] = np.linalg.norm(rhythms.loc[clustercol == i] - cluster_centers[i], axis=1).max() 
        else:
            dists[i,i] = c_dist[clustercol.loc[clustered_mask]==i, i].max()
    return dists


def format_cluster_stats_distance_table(df):
    clusters = Clusters.order()
    mask = np.zeros_like(df.values, dtype=bool)
    mask[np.tril_indices_from(mask, k=len(clusters)-1)] = True
    return (df
        .loc[clusters, [*df.columns[:-len(clusters)], *clusters]]
        .where(mask, None)
        .style
        .format(lambda x: f'{x:.3f}', na_rep="-")
        .format("{:,.0f}", subset=['number of users', 'number of posts'])
        .format("{:,.2f}", subset='posts per user')
        .apply(highlight_diag, axis=None, subset=clusters))

cluster_center_df = lambda cr: pd.DataFrame(cr.cluster_centers, columns=cr.user_activity.columns).rename(index=cr.num_to_name).rename_axis(Columns.CLUSTER.value)

@dataclass
class ClusterResult:
    label: str
    user_activity: pd.DataFrame
    score_df: pd.DataFrame
    chosen_method: str
    nr_clusters: int
    clustercol: pd.Series
    num_to_name: dict
    cluster_centers: np.ndarray
    transform: Callable


class ClusterRoutine():
    
    def __init__(self, config, methods, processor = None, cutoff_post_nr= 240, columns=[Columns.POSTS.value]):
        self.cutoff_post_nr = cutoff_post_nr
        self.methods = methods
        self.palette=Clusters.palette()
        self._users_over_threshold = None
        self._smoothed_user_activity = None
        if processor is None:
            processor = DataProcessor(config, columns=columns)
        self.processor = processor
        self.config=config

    @property
    def users_over_threshold(self):
        if self._users_over_threshold is None:
            self._users_over_threshold = self.processor.posts_per_user.loc[self.processor.posts_per_user > self.cutoff_post_nr].index
        return self._users_over_threshold

    @property
    def smoothed_user_activity(self):
        if self._smoothed_user_activity is None:
            self.smooth_user_activity()
        return self._smoothed_user_activity

    def smooth_user_activity(self, padding=3, std=3):
        self._smoothed_user_activity = smooth_looped(self.processor.user_activity, padding=padding, std=std)


    def routine_by(self, user_activity, label, metric="euclidean", ks=list(range(3,10)), verbose=True):
        score_df = get_clustering_scores(user_activity.loc[user_activity.index.intersection(self.users_over_threshold)], metric=metric, ks=ks, methods=self.methods)
        result_per_indicator = best_result_per_indicator(score_df)
        nr_clusters, chosen_method = result_per_indicator.mode()[0]
        clustering_instance, clustercol = refit_predict(
            chosen_method, nr_clusters, 
            methods=self.methods, 
            data=user_activity.loc[user_activity.index.intersection(self.users_over_threshold)], 
            cluster_indices=self.processor.posts_per_user.index
        )
        clustercol.loc[clustercol.index.difference(user_activity.index)] = None
        
        if verbose:
            display(
                score_df.unstack(level=0).swaplevel(axis=1).sort_index(axis=1)
                .style
                .apply(lambda x: ['font-weight: bold' if i else '' for i in x.index.isin(result_per_indicator.loc[result_per_indicator==(x.name[1], x.name[0])].index)])
            )
        print(f'Clustered into {nr_clusters} clusters using {chosen_method}')

        return ClusterResult(
            label = label,
            user_activity = user_activity,
            score_df = score_df,
            chosen_method = chosen_method,
            nr_clusters = nr_clusters,
            clustercol = clustercol,
            num_to_name = cluster_num_to_name(clustering_instance.cluster_centers_),
            cluster_centers = np.concatenate([clustering_instance.cluster_centers_, np.expand_dims(user_activity.loc[user_activity.index.difference(self.users_over_threshold)].mean(), 0)]),
            transform = lambda x: clustering_instance.transform(x) ** 2
        )

    def get_cluster_stats_distance_table_style(self, cluster_result, save=False):
        dists_df = get_distance_stats(cluster_result.cluster_centers, cluster_result.user_activity, cluster_result.clustercol, transform = cluster_result.transform, separate_indices=[cluster_result.nr_clusters])
        stats_dist_df_style = format_cluster_stats_distance_table(get_cluster_stats(self.processor.posts_per_user, cluster_result.clustercol)
                .merge(pd.DataFrame(dists_df), left_index=True, right_index=True)
                .rename(index=cluster_result.num_to_name, columns=cluster_result.num_to_name))
        if save:
            save_to_latex(
                self.config,
                stats_dist_df_style,
                label= f'cluster_stats_{cluster_result.label}',
                caption=r"Number of users, posts, and posts per user for each cluster as well as Euclidean distances in between (lower triangle) and within (main diagonal, highlighted in bold) cluster centroids. The mean of user activity curves were used for infrequent type users.")
        return stats_dist_df_style
