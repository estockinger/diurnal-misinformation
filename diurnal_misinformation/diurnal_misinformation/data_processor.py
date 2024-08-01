import pandas as pd

from .enums import Columns, FactTypes, ContentType
from .utils import groupby_with_total
from .path_utils import get_data_file_path, get_cluster_col_path


def user_mapping(d, c, name=None):
    i = d.index.get_level_values(Columns.USERHASH.value).map(c)
    if name is not None:
        i.rename(name, inplace=True)
    return i
    

def ft_mapping(d, facttypes, **kwargs): 
    return pd.Index(d.index.get_level_values(Columns.FACTTYPE.value).isin(facttypes), **kwargs)

known_mapping = lambda d: ft_mapping(d, FactTypes.known_order('name'), name=ContentType.KNOWN.value)
disinformative_mapping = lambda d: ft_mapping(d, FactTypes.disinformative_order('name'), name=ContentType.DISINFORMATIVE.value)


class DataProcessor:

    def __init__(self, config, columns=[Columns.POSTS.value]):
        self.config=config
        self.columns=columns

        self._data = None
        self._known_data = None
        self._posts_per_user = None
        self._posts_per_user_ft = None
        self._posts_per_user_t = None
        self._posts_per_user_ft_t = None
        self._known_posts_per_user_t = None
        self._disinf_posts_per_user_t = None
        self._ratio_of_known_per_user_ft_t = None
        self._user_activity = None
        self._disinf_user_activity = None
        self._user_disinf_ratio_t = None
        self._cluster_cols = None


    def cluster_mapping(self, d, clustertype):
        return user_mapping(d, self.cluster_cols[clustertype], name=Columns.CLUSTER.value)


    @property
    def data(self):
        if self._data is None:
            self._data = pd.read_parquet(get_data_file_path(self.config), columns=self.columns)
        return self._data
    

    @property
    def known_data(self):
        if self._known_data is None:
            self._known_data = self.data.loc[known_mapping(self.data)]
        return self._known_data
    

    @property
    def posts_per_user(self):
        if self._posts_per_user is None:
            self._posts_per_user = self.data[Columns.POSTS.value].groupby(Columns.USERHASH.value).sum()
        return self._posts_per_user



    @property
    def posts_per_user_ft(self):
        if self._posts_per_user_ft is None:
            self._posts_per_user_ft = groupby_with_total(self.data[Columns.POSTS.value], [Columns.USERHASH.value, Columns.FACTTYPE.value], 'sum', total_for_idx=1).unstack(level=Columns.FACTTYPE.value, fill_value=0)
            self._posts_per_user_ft[ContentType.DISINFORMATIVE.value] = self._posts_per_user_ft[FactTypes.disinformative_order('name')].sum(axis=1)
            self._posts_per_user_ft[ContentType.KNOWN.value] = self._posts_per_user_ft[FactTypes.known_order('name')].sum(axis=1)
        return self._posts_per_user_ft[FactTypes.known_order('name') + [ContentType.DISINFORMATIVE.value, ContentType.KNOWN.value, 'total']]


    @property
    def posts_per_user_t(self):
        if self._posts_per_user_t is None:
            self._posts_per_user_t = self.data[Columns.POSTS.value].groupby(level=[Columns.USERHASH.value, Columns.MIN_BINS15.value]).sum()
        return self._posts_per_user_t


    @property
    def posts_per_user_ft_t(self):
        if self._posts_per_user_ft_t is None:
            self._posts_per_user_ft_t = (self.data[Columns.POSTS.value]
                .groupby([Columns.USERHASH.value, Columns.FACTTYPE.value, Columns.MIN_BINS15.value]).sum()
            )
        return self._posts_per_user_ft_t


    @property
    def ratio_of_known_per_user_ft_t(self):
        if self._ratio_of_known_per_user_ft_t is None:
            self._ratio_of_known_per_user_ft_t = self.posts_per_user_ft_t.loc[known_mapping(self.posts_per_user_ft_t)].div(self.known_posts_per_user_t)
        return self._ratio_of_known_per_user_ft_t


    @property
    def known_posts_per_user_t(self):
        if self._known_posts_per_user_t is None:
            self._known_posts_per_user_t = (self.data
                .loc[known_mapping(self.data), Columns.POSTS.value]
                .groupby([Columns.USERHASH.value, Columns.MIN_BINS15.value]).sum()
            )
        return self._known_posts_per_user_t


    @property
    def disinf_posts_per_user_t(self):
        if self._disinf_posts_per_user_t is None:
            self._disinf_posts_per_user_t = (
                self.data
                .loc[disinformative_mapping(self.data), Columns.POSTS.value]
                .groupby([Columns.USERHASH.value, Columns.MIN_BINS15.value]).sum()
            )
        return self._disinf_posts_per_user_t


    @property
    def user_activity(self):
        if self._user_activity is None:
            self._user_activity = (
                self.posts_per_user_t
                .unstack(level=Columns.MIN_BINS15.value, fill_value=0)
                .div(self.posts_per_user_t.groupby(level=Columns.USERHASH.value).sum(), axis=0)
            )
        return self._user_activity


    @property
    def disinf_user_activity(self):
        if self._disinf_user_activity is None:
            self._disinf_user_activity = (
                self.disinf_posts_per_user_t
                .unstack(level=Columns.MIN_BINS15.value, fill_value=0)
                .div(self.disinf_posts_per_user_t.groupby(level=Columns.USERHASH.value).sum(), axis=0)
            )
        return self._disinf_user_activity


    @property
    def user_disinf_ratio_t(self):
        if self._user_disinf_ratio_t is None:
            self._user_disinf_ratio_t = self.disinf_posts_per_user_t.div(self.known_posts_per_user_t, fill_value=0)
        return self._user_disinf_ratio_t
    

    @property
    def cluster_cols(self):
        if self._cluster_cols is None:
            self._cluster_cols = pd.read_parquet(get_cluster_col_path(self.config))
        return self._cluster_cols


    def ratio_of_known_per_cluster_by_user_ft_t(self, clustertype):
        return self.get_per_cluster_x(clustertype, self.ratio_of_known_per_user_ft_t, 'mean', columns=[Columns.FACTTYPE.value, Columns.MIN_BINS15.value])


    def get_per_cluster_x(self, clustertype, df, *args, columns=[], **kwargs):
        if columns is not None and len(columns) > 0:
            kwargs.setdefault('total_for_idx', 0)
        return groupby_with_total(df, [self.cluster_mapping(df, clustertype), *columns], *args, **kwargs)


    def get_ratio_by_tweet_x(self, clustertype, nominator, denominator, *args, **kwargs):
        return self.get_per_cluster_x(clustertype, nominator, 'sum', columns=args).div(self.get_per_cluster_x(clustertype, denominator, 'sum', columns=args), **kwargs)
    