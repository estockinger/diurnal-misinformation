import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hac
import scripts.constants as constants
from scripts.config import Config
from scripts.utils import correct_utc, get_sun, date_time_to_local
from scripts.enums import Columns, Clusters, FactTypes, ContentType
from statsmodels.stats import proportion


class DataProcessor:
    DATE_TIME_FUNCS = {
        Columns.YEAR: lambda x: x.dt.year,
        Columns.MONTH: lambda x: x.dt.month,
        Columns.HOUR: lambda x: x.dt.hour,
        Columns.MINUTES: lambda x: x.dt.minute,
        Columns.WEEK: lambda x: x.dt.isocalendar().week,
        Columns.DAY: lambda x: x.dt.isocalendar().day,
        Columns.MIN_BINS15: lambda x: (((x.dt.hour * 3600) + (x.dt.minute * 60)) // 900) * 0.25,
        Columns.IS_WEEKEND: lambda x: (x.isocalendar().day < 6).astype(bool)
    }

    def __init__(self, config: Config) -> None:
        self.unverified = None
        self.so_far_left = None
        self.posts_per_user = None
        self.config = config
        self.all = None

    def load_and_prepare(self, dataloader, country_config) -> None:
        self.all = dataloader.load_and_prepare(country_config)

        print("Converting dateTime")
        self.convert_date_time(country_config)
        self.all.reset_index(inplace=True)
        self.all.drop_duplicates(subset=Columns.TWEET_ID.value, inplace=True)
        # Dropping empty TweetTypes; known errors in data collection
        if Columns.TWEET_TYPE.value in self.all.columns:
            self.all.drop(self.all.loc[self.all[Columns.TWEET_TYPE.value] == ''].index, inplace=True)

        # self.inplace_fact_type_to_reliability()
        self.all[Columns.FACTTYPE.value] = self.all[Columns.FACTTYPE_ORIGINAL.value] \
            .map(lambda x: constants.FACTTYPE_MAP.get(x, FactTypes.OTHER).value.name)
        self.all[Columns.MACHINATED.value] = self.all[Columns.FACTTYPE.value] \
            .map(lambda x: constants.FACTTYPE_NAME_MAP.get(x, FactTypes.OTHER).value.content_type == ContentType.MANIPULATED)

        if Columns.USER.value in self.all.columns:
            print("Finding posts per user")
            self.find_posts_per_user()

        # self.get_circadian_info(country_config)
        if Columns.IS_BOT.value in self.all.columns:
            self.all[Columns.IS_BOT.value] = self.all[Columns.IS_BOT.value].astype(bool)
        if Columns.VERIFIED.value in self.all.columns:
            self.all[Columns.VERIFIED.value] = self.all[Columns.VERIFIED.value].astype(bool)
            self.unverified = self.all.loc[~self.all[Columns.VERIFIED.value]]
        if Columns.FOLLOWERS_COUNT.value in self.all.columns:
            self.all[Columns.FOLLOWERS_COUNT.value] = self.all[Columns.FOLLOWERS_COUNT.value].fillna(0).astype(int)


    def find_posts_per_user(self, bins=None):
        if bins is None:
            bins = [0, 2, 10, 20, 50, 100, 500, np.inf]
        self.posts_per_user = self.all.groupby([Columns.USER.value]).agg({Columns.TWEET_ID.value: lambda x: x.count()})
        self.posts_per_user.fillna(0, inplace=True)
        self.posts_per_user.columns = [Columns.NUM_POSTS.value]
        self.posts_per_user[Columns.NUM_POSTS_BIN.value] = pd.cut(self.posts_per_user[Columns.NUM_POSTS.value], bins)
        self.all[Columns.NUM_POSTS.value] = self.all[Columns.USER.value].map(
            self.posts_per_user[Columns.NUM_POSTS.value])
        self.all[Columns.NUM_POSTS_BIN.value] = self.all[Columns.USER.value].map(self.posts_per_user[Columns.NUM_POSTS_BIN.value])
        counts_per_user = self.posts_per_user[Columns.NUM_POSTS.value].value_counts()
        factor = 1.0 / sum(counts_per_user.values) if sum(counts_per_user.values) else 1
        normalised_counts = {k: v * factor for k, v in counts_per_user.items()}
        self.all[Columns.WEIGHT.value] = self.all[Columns.NUM_POSTS.value].map(normalised_counts).astype(float)

    def inplace_fact_type_to_reliability(self):
        self.all[Columns.RELIABILITY.value] = pd.Categorical(
            self.all[Columns.FACTTYPE_ORIGINAL.value].map(constants.RELIABLE))
        self.all[Columns.HARMSCORE.value] = self.all[Columns.FACTTYPE_ORIGINAL.value].map(constants.HARM_SCORES)

    def convert_date_time(self, country_config):
        self.all[Columns.LOCAL_TIME.value] = date_time_to_local(self.all[Columns.DATETIME.value], country_config)
        for k in self.config.time_columns:
            print("Converting ", k)
            self.all[k.value] = self.DATE_TIME_FUNCS[k](self.all[Columns.LOCAL_TIME.value])

    def get_circadian_info(self, country_config):
        self.so_far_left = self.all.apply(
            lambda row: get_sun(row, country_config),
            axis=1,
            result_type="expand")
        self.so_far_left.columns = [Columns.SUN_SO_FAR_M.value, Columns.SUN_LEFT_M.value, Columns.TOTAL_SUN_M.value]
        self.df = self.all.join(self.so_far_left)


def group_weekly(df):
    return group_by_with_total(df, [Columns.DAY, Columns.MIN_BINS15, Columns.RELIABILITY])


def group_weekly_to_daily(grouped_weekly):
    df_grouped = grouped_weekly.stack()
    return (df_grouped.groupby(by=[Columns.MIN_BINS15.value, Columns.RELIABILITY.value])
            .agg(**get_col_agg_info(df_grouped))
            .unstack())


def group_time(df, freq="4H"):
    df[Columns.DATE_BIN] = df.local_time.apply(
        lambda x: x.ceil(freq=freq, nonexistent="shift_forward", ambiguous=True))
    return group_by_with_total(df, [Columns.DATE_BIN, Columns.RELIABILITY])


def weighted_mean(df, group, avg_value, weight_value=None):
    """
    Helper function to calculate the mean of column "avg_value" by column "weight_value" during aggregation of grouped
    DataFrames.
    """
    if weight_value is None:
        return group.mean()
    sl = df.loc[group.index]
    d = sl[avg_value]
    w = sl[weight_value]
    if d[:-1].isna().values.all():
        return None
    try:
        if d[:-1].isna().values.any():
            # if a value is NaN, ignore it to calculate averages
            i = d.notna()
            return (d.loc[i] * w.loc[i]).sum() / w.loc[i].sum()
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


def weighted_count(df, group, weight_value=None):
    """
    Helper function to weight the counts of column "avg_value" by column "weight_value" during aggregation of grouped
    DataFrames.
    """
    if weight_value is None:
        return group.count()
    try:
        return df.loc[group.index][weight_value].sum()
    except ZeroDivisionError:
        return group.count()


def get_idx_vals(df, c, total_for=Columns.RELIABILITY):
    if c == total_for:
        return pd.Categorical(df[c], [c for c, cnt in df[c].value_counts().items() if cnt > 0] + ['total'])
    elif pd.api.types.is_categorical_dtype(df[c]):
        return df[c].cat.categories
    else:
        return df[c].unique()


def get_confidence_interval(row, totals):
    try:
        i = proportion.proportion_confint(row["num_posts"], totals.loc[row.name[1:], "num_posts"], alpha=0.05)
        r = (i[1] - i[0]) / 2
        return r
    except BaseException as err:
        # print("An error occurred: ", err)
        return 0


def group_by_with_total(df, group_by_cols, aggregate_cols=None, total_for_idx=-1, ratio_for_idx=-1):
    group_by_cols = [t.value for t in group_by_cols]
    total_for = group_by_cols[total_for_idx]

    agg_info = get_col_agg_info(df, aggregate_cols=aggregate_cols)
    grp_cols_no_total = [x for x in group_by_cols if x != total_for]
    df_grouped = df.groupby(by=[total_for] + grp_cols_no_total).agg(**agg_info)

    # totals
    totals = df.dropna(axis=0, how="all", subset=total_for).groupby(by=grp_cols_no_total).agg(**agg_info)
    df_grouped["num_posts_conf_int"] = df_grouped.apply(lambda row: get_confidence_interval(row, totals), axis=1)
    # add index level
    totals = pd.concat({'total': totals}, names=[total_for])
    df_grouped = pd.concat([df_grouped, totals]).reorder_levels(group_by_cols).sort_index()

    # ratios
    if ratio_for_idx == total_for_idx:
        ratio_sum_grp = df_grouped.groupby(level=grp_cols_no_total, sort=False)
        df_grouped[Columns.RATIO_BY_TWEET.value] = df_grouped[Columns.RATIO_BY_TWEET.value] / ratio_sum_grp[
            Columns.RATIO_BY_TWEET.value].transform(
            lambda x: x.xs("total", level=total_for).sum())
        df_grouped[Columns.RATIO_BY_USER.value] = df_grouped[Columns.RATIO_BY_USER.value] / ratio_sum_grp[
            Columns.RATIO_BY_USER.value].transform(
            lambda x: x.xs("total", level=total_for).sum())
    else:
        sum_ratio = df_grouped.groupby(level=[i for i in group_by_cols if i != group_by_cols[ratio_for_idx]],
                                       sort=False)
        df_grouped[Columns.RATIO_BY_TWEET.value] /= sum_ratio[Columns.RATIO_BY_TWEET.value].transform(sum)
        df_grouped[Columns.RATIO_BY_USER.value] /= sum_ratio[Columns.RATIO_BY_USER.value].transform(sum)

    return df_grouped


def get_activity_by(df, ys=None, activity_col=Columns.TWEET_ID.value):
    if ys is None:
        ys = []
    counts = df.groupby(ys)[[activity_col]].agg(**{
        Columns.ACTIVITY.value: (activity_col, lambda g: g.count()),
        Columns.ACTIVITY_WEIGHTED.value: (activity_col, lambda g: weighted_count(df, g, Columns.WEIGHT.value))
    })
    tmp = counts / counts.groupby(level=-1).sum()
    tmp = tmp.unstack()
    total_tmp = counts.groupby(level=0).sum() / counts.sum()
    total_tmp.columns = pd.MultiIndex.from_tuples(zip(total_tmp.columns, ["total", "total"]))
    tmp = pd.concat([tmp, total_tmp], axis=1).sort_index(axis=1)
    return tmp.fillna(0).stack()


def get_col_agg_info(df, aggregate_cols=None):
    if aggregate_cols is None:
        aggregate_cols = []
    mean_func = lambda _: "mean"
    count_tuple = ratio_tuple = (Columns.TWEET_ID.value, "count")

    mean_func_norm = lambda col: lambda g: weighted_mean(df, g, col, Columns.WEIGHT.value)
    ratio_tuple_norm = (Columns.TWEET_ID.value, lambda g: weighted_count(df, g, Columns.WEIGHT.value))

    agg_info = {
        Columns.NUM_POSTS.value: count_tuple,
        Columns.NUM_POSTS_WEIGHTED.value: ratio_tuple_norm,
        Columns.RATIO_BY_TWEET.value: ratio_tuple,
        Columns.RATIO_BY_USER.value: ratio_tuple_norm
    }

    update_dict = dict()
    for c in aggregate_cols:
        c = c.value
        update_dict[c] = (c, mean_func(c))
        update_dict[f"{c}_norm"] = (c, mean_func_norm(c))

        if c == "lat" or c == "long":
            continue

        update_dict[f"{c}_var"] = (c, "var")
        update_dict[f"{c}_sem"] = (c, "sem")

    agg_info.update(update_dict)
    return agg_info


def lowess_smooth(x, y, frac=0.1):
    import statsmodels.api as sm
    lowess = sm.nonparametric.lowess(y, x, frac=frac)
    return lowess[:, 1]


def form_clusters(df,
                  total_posts_per_user,
                  cut_off_level,
                  n=500,
                  smoothing="rm",
                  window=4,
                  criterion="maxclust",
                  linkage_method="ward"):
    posts_per_user_and_min_bins15 = df.loc[df[Columns.NUM_POSTS] >= n].groupby(by=[Columns.USER, Columns.MIN_BINS15])[
        Columns.TWEET_ID].count()
    posts_per_user_and_min_bins15 /= posts_per_user_and_min_bins15.index.get_level_values(level=Columns.USER).map(
        total_posts_per_user)
    posts_per_user_and_min_bins15 = posts_per_user_and_min_bins15.unstack(level=0)
    posts_per_user_and_min_bins15.fillna(0, inplace=True)

    if smoothing == "rm":
        posts_per_user_and_min_bins15 = posts_per_user_and_min_bins15.rolling(window, min_periods=1).mean()
    elif smoothing == "ewa":
        posts_per_user_and_min_bins15 = posts_per_user_and_min_bins15.ewm(span=window).mean()

    d = hac.linkage(posts_per_user_and_min_bins15.transpose(), method=linkage_method)
    result = pd.Series(hac.fcluster(d, cut_off_level, criterion=criterion))
    clusters = result.unique()
    users_in_cluster = dict()
    cluster_averages = pd.DataFrame(index=posts_per_user_and_min_bins15.index, columns=clusters)

    for c in clusters:
        cluster_index = result[result == c].index
        cluster = posts_per_user_and_min_bins15.iloc[:, cluster_index]
        cluster_averages[c] = cluster.mean(axis=1)
        for user in cluster.columns:
            users_in_cluster[user] = c

    return cluster_averages, users_in_cluster


def group_over_time(df, time_cols):
    return group_by_with_total(
        df.loc[df[Columns.LOCAL_TIME.value] < '2022-08-01'],
        group_by_cols=time_cols + [Columns.MIN_BINS15, isp.i_col, isp.sp.c_col],
        aggregate_cols=[],
        ratio_for_idx=-2
    )
