import warnings

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from packaging import version
from validclust import ValidClust
from yellowbrick.cluster.elbow import distortion_score
from validclust.indices import *
from validclust import *
import seaborn as sns

from sklearn.metrics import (
    davies_bouldin_score, silhouette_score, pairwise_distances
)

# They changed the name of calinski_harabaz_score in later version of sklearn:
# https://github.com/scikit-learn/scikit-learn/blob/c4733f4895c1becdf587b38970f6f7066656e3f9/doc/whats_new/v0.20.rst#id2012
sklearn_version = version.parse(sklearn.__version__)
nm_chg_ver = version.parse("0.23")
if sklearn_version >= nm_chg_ver:
    from sklearn.metrics import calinski_harabasz_score as _cal_score
else:
    from sklearn.metrics import calinski_harabaz_score as _cal_score


def _get_clust_pairs(clusters):
    return [(i, j) for i in clusters for j in clusters if i > j]


def _dunn(data=None, dist=None, labels=None):
    clusters = set(labels)
    inter_dists = [
        dist[np.ix_(labels == i, labels == j)].min()
        for i, j in _get_clust_pairs(clusters)
    ]
    intra_dists = [
        dist[np.ix_(labels == i, labels == i)].max()
        for i in clusters
    ]
    return min(inter_dists) / max(intra_dists)


def _silhouette_score2(data=None, dist=None, labels=None):
    return silhouette_score(dist, labels, metric='precomputed')


def _davies_bouldin_score2(data=None, dist=None, labels=None):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'divide by zero')
        return davies_bouldin_score(data, labels)


def _calinski_harabaz_score2(data=None, dist=None, labels=None):
    return _cal_score(data, labels)


def cop(data, dist, labels):
    r"""Calculate the COP CVI

    See Gurrutxaga et al. (2010) for details on how the index is calculated. [1]_

    Parameters
    ----------
    data : array-like, shape = [n_samples, n_features]
        The data to cluster.
    dist : array-like, shape = [n_samples, n_samples]
        A distance matrix containing the distances between each observation.
    labels : array [n_samples]
        The cluster labels for each observation.

    Returns
    -------
    float
        The COP index.

    References
    ----------
    .. [1] Gurrutxaga, I., Albisua, I., Arbelaitz, O., Martín, J., Muguerza,
       J., Pérez, J., Perona, I. (2010). SEP/COP: An efficient method to find
       the best partition in hierarchical clustering based on a new cluster
       validity index. Pattern Recognition, 43(10), 3364-3373. DOI:
       10.1016/j.patcog.2010.04.021.

    """
    clusters = set(labels)
    cpairs = _get_clust_pairs(clusters)
    prox_lst = [
        dist[np.ix_(labels == i[0], labels == i[1])].max()
        for i in cpairs
    ]

    out_l = []
    for c in clusters:
        c_data = data[labels == c]
        c_center = c_data.values.mean(axis=0, keepdims=True)
        c_intra = pairwise_distances(c_data, c_center).mean()

        c_prox = [prox for pair, prox in zip(cpairs, prox_lst) if c in pair]
        c_inter = min(c_prox)

        to_add = len(c_data) * c_intra / c_inter
        out_l.append(to_add)

    return sum(out_l) / len(labels)


class DistortionMapper:
    def __init__(self, metric):
        self.metric = metric

    def distortion_score(self, data=None, dist=None, labels=None):
        """
        Simple wrapper over the yellowbrick distortion_score implementation.
        
        Compute the mean distortion of all samples.

        The distortion is computed as the the sum of the squared distances between
        each observation and its closest centroid. Logically, this is the metric
        that K-Means attempts to minimize as it is fitting the model.
        
        Smaller distortions == higher quality models. 
        """
        return distortion_score(X=data, labels=labels, metric=self.metric)


def minimax(v):
    return (v - v.min()) / (v.max() - v.min())


class ValidClustMod(ValidClust):

    def __init__(self, k,
                 # No big deal that these are lists (i.e., mutable), given that
                 # we don't mutate them inside the class.
                 indices=['silhouette', 'calinski', 'davies', 'dunn'],
                 methods=['hierarchical', 'kmeans'],
                 linkage='ward', affinity='euclidean', norm_func=None):
        self.score_df = None
        self.distortionMapper = DistortionMapper(affinity)
        self.normFunc = norm_func if norm_func else minimax
        ValidClust.__init__(self, k, indices, methods, linkage, affinity)

    def add_index(self, index):
        self.indices.append(index)

    def _get_index_funs(self):
        index_fun_switcher = {
            'silhouette': _silhouette_score2,
            'calinski': _calinski_harabaz_score2,
            'davies': _davies_bouldin_score2,
            'dunn': _dunn,
            'cop': cop,
            'distortion': self.distortionMapper.distortion_score
        }
        return {i: index_fun_switcher[i] for i in self.indices}

    def fit(self, data):
        """Fit the clustering algorithm(s) to the data and calculate the CVI
        scores

        Parameters
        ----------
        data : array-like, shape = [n_samples, n_features]
            The data to cluster.

        Returns
        -------
        self
            A ``ValidClust`` object whose ``score_df`` attribute contains the
            calculated CVI scores.
        """
        method_objs = self._get_method_objs()
        index_funs = self._get_index_funs()
        dist_inds = ['silhouette', 'dunn']

        d_overlap = [i for i in self.indices if i in dist_inds]
        if d_overlap:
            dist = pairwise_distances(data)
            np.fill_diagonal(dist, 0)
        else:
            dist = None

        index = pd.MultiIndex.from_product(
            [self.methods, self.indices],
            names=['method', 'index']
        )
        output_df = pd.DataFrame(
            index=index, columns=self.k, dtype=np.float64
        )

        for k in self.k:
            for alg_name, alg_obj in method_objs.items():
                alg_obj.set_params(n_clusters=k)
                labels = alg_obj.fit_predict(data)
                labels_1d = alg_obj.labels_
                # have to iterate over self.indices here so that ordering of
                # validity indices is same in scores list as it is in output_df
                scores = [
                    index_funs[key](data, dist, labels_1d if key == 'distortion' else labels)
                    for key in self.indices
                ]
                output_df.loc[(alg_name, self.indices), k] = scores

        self.score_df = output_df
        return self

    def _normalize(self):
        score_df_norm = self.score_df.copy()
        for i in ['davies', 'cop', 'distortion']:
            if i in self.indices:
                score_df_norm.loc[(slice(None), i), :] *= -1  # - score_df_norm.loc[(slice(None), i), :]
        score_df_norm = score_df_norm.apply(self.normFunc, axis=1)
        return score_df_norm

    def plot(self):
        """Plot normalized CVI scores in a heatmap

        The CVI scores are normalized along each method/index pair using the
        `max norm <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html>`_.
        Note that, because the scores are normalized along each method/index
        pair, you should compare the colors of the cells in the heatmap only
        within a given row. You should not, for instance, compare the color of
        the cells in the "kmeans, dunn" row with those in the
        "kmeans, silhouette" row.

        Returns
        -------
        None
            Nothing is returned. Instead, a plot is rendered using a
            graphics backend.
        """
        norm_df = self._normalize()

        yticklabels = [',\n'.join(i) for i in norm_df.index.values]
        hmap = sns.heatmap(
            norm_df, cmap='Blues', cbar=False, yticklabels=yticklabels, annot=True
        )
        hmap.set_xlabel('\nNumber of clusters')
        hmap.set_ylabel('Method, index\n')
        plt.tight_layout()
        return norm_df
