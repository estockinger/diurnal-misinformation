import pandas as pd
from scripts.enums import Columns, FactTypes
from scripts.preprocessing import group_by_with_total, get_activity_by


class PathManager:
    def __init__(self, config, country_label, user_type):
        self.config = config
        self.country_label = country_label
        self.user_type = user_type

    def to_latex(self, df_style, label, caption, is_multi_index=False):
        kwargs = dict(
            buf=self.get_latex_path(label),
            position='htb',
            hrules=True,
            label=f'tab:{label}',
            convert_css=True,
            sparse_index=True,
            caption=caption
        )
        if is_multi_index:
            kwargs['multicol_align'] = "c"
            kwargs['multirow_align'] = "l"
            kwargs['clines'] = 'skip-last;index'
        df_style.to_latex(**kwargs)

    def get_latex_path(self, label):
        return f'{self.config.SAVE_STATS_DIR}{self.country_label}_{label}_{self.user_type}.tex'

    def get_data_path(self, label):
        return f'{self.config.SAVE_DATA_DIR}{self.country_label}_{label}_{self.user_type}.parquet.gzip'

    def save_plot(self, figure, name):
        figure.savefig(f"{self.config.SAVE_ROOT_DIR}{self.country_label}_{name}_{self.user_type}.eps",
                       transparent=False)


class CurvePathManager(PathManager):
    CURVE_TYPE = None

    def __init__(self, config, country_label, user_type):
        super().__init__(config, country_label, user_type)
        self.func_dict = {
            'signal': self.get_signal_path(),
            'fourier': self.get_fourier_path()
        }

    def get_signal_path(self):
        return self.get_data_path(f'cluster_{self.CURVE_TYPE}')

    def get_fourier_path(self):
        return self.get_data_path(f'cluster_{self.CURVE_TYPE}_fourier')

    def save(self, df, kind):
        assert kind in self.func_dict.keys()
        df.to_parquet(self.func_dict[kind], compression="gzip")

    def load(self, kind):
        assert kind in self.func_dict.keys()
        return pd.read_parquet(self.func_dict[kind])


class ActivityPathManager(CurvePathManager):
    CURVE_TYPE = 'activity'


class MachinatedPathManager(CurvePathManager):
    CURVE_TYPE = 'machinated'


def get_waking_time_path(config, country_config, kind):
    return f'{config.SAVE_DATA_DIR}{country_config.LABEL}_waking_times_{kind}.pickle'


def get_cluster_col_path(path_manager):
    return path_manager.get_data_path('cluster_col')


def save_activity_per_cluster(df, activity_path_manager):
    activity = get_activity_by(
        df,
        ys=[Columns.MIN_BINS15.value, Columns.CLUSTER.value],
        activity_col=Columns.NUM_POSTS.value)
    activity_path_manager.save(activity, 'signal')
    return activity


def save_machinated_per_cluster(df, ratio_path_manager):
    harmful_per_cluster = (
        group_by_with_total(
            df.loc[df[Columns.FACTTYPE.value] != FactTypes.OTHER.value.name],
            group_by_cols=[Columns.MIN_BINS15, Columns.CLUSTER, Columns.MACHINATED],
            aggregate_cols=[], total_for_idx=1)
        .xs(True, level=Columns.MACHINATED.value)
        [[Columns.RATIO_BY_TWEET.value, Columns.RATIO_BY_USER.value]]
        .unstack(level=1).fillna(0))
    ratio_path_manager.save(harmful_per_cluster.stack(), 'signal')
    return harmful_per_cluster
