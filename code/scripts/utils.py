from statsmodels.tsa.stattools import adfuller
from datetime import date
from suntime import Sun
import pytz
import pandas as pd
import scipy.stats as stats
from statsmodels.stats import proportion
from scipy.signal import find_peaks
import numpy as np
import math

weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def highlight_diag(df):
    a = np.full(df.shape, '', dtype='<U24')
    np.fill_diagonal(a[1:], 'font-weight: bold')
    return pd.DataFrame(a, index=df.index, columns=df.columns)


def split_every_n_words(text, n=2):
    text = text.split()
    return '\n'.join([' '.join(text[i:i + n]) for i in range(0, len(text), n)])


def date_time_to_local(col, country_config):
    col = col.apply(correct_utc)
    col = pd.to_datetime(col).dt.tz_localize(tz='UTC')
    return pd.to_datetime(col.dt.tz_convert(tz=country_config.TIME_ZONE))


def get_chi2_table(df, ratio_col, x_col, y_col, nr_tries_col="num_posts"):
    return (df[[ratio_col]]
            .reset_index()
            .pivot(x_col, y_col, ratio_col).T
            .apply(lambda r: r * df[[nr_tries_col]].unstack().sum(axis=1), axis=0))


def hours_to_mins(time, format_ampm=True):
    hours = int(time)
    minutes = (time * 60) % 60
    if format_ampm:
        ampm = 'AM' if hours < 12 else 'PM'
        hours = 12 if time % 12 == 0 else hours % 12
        return f'{hours:.0f}:{minutes:>02.0f} {ampm}'
    else:
        return f'{hours:.0f}:{minutes:>02.0f}'


def shift(df_tmp, cluster, alignment, c_col, col="num_posts_norm"):
    return np.c_[df_tmp.xs(cluster, level=c_col).index.values, np.roll(df_tmp.xs(cluster, level=c_col)[col], alignment)]


def cons_max(df, n):
    max_loc = np.convolve(df.values, np.ones(n, dtype=float), mode='valid').argmax()
    return df.index[max_loc], df.index[max_loc + n - 1]


def groups_of_at_least_n(y, n):
    return y & (y.groupby((y.diff() != 0).cumsum()).transform(np.size) > n - 1)


def round_up_to_1(x):
    num_figs = int(math.floor(math.log10(abs(x))))
    return math.ceil(x / 10 ** num_figs) * 10 ** num_figs


def get_mk_sp_dfs(df, x, y):
    tmp = df.set_index(['min_bins15', x, y])['y_val'].unstack(x).unstack(y)
    mannkendall_df = calculate_c_r_for_index(tmp, mannkendall_slope_p, index=["correlations", "pvals"])
    mannkendall_df.columns = pd.MultiIndex.from_tuples(mannkendall_df.columns)
    spearman_df = calculate_c_r_for_index(tmp, lambda col: spearmanr(col.index, col, nan_policy="omit"),
                                          index=["correlations", "pvals"])
    spearman_df.columns = pd.MultiIndex.from_tuples(spearman_df.columns)
    return mannkendall_df, spearman_df


def within_n_hours_before_or_after_t(x, t, n, include_edges=False):
    return t_between(x, (t - n) % 24, (t + n) % 24, include_edges=include_edges)


def time_past_t(x, t):
    if x > t:
        return x - t
    else:
        return 24 - t + x


def t_between(t, start, cutoff, include_edges=False):
    res = start < t < cutoff if cutoff > start else t > start or t < cutoff
    if include_edges:
        res = res or math.isclose(t, start) or math.isclose(t, cutoff)
    return res


def prolongued_waking_condition(t, wakeup_time, num_h=16):
    multiplier = 4
    cutoff = (((wakeup_time + num_h) * multiplier) % (24 * multiplier)) / multiplier
    if cutoff > wakeup_time:
        return t > cutoff or t < wakeup_time
    else:
        return cutoff < t < wakeup_time


def prolongued_waking_condition_by_bedtime(t, wakeup_time, num_h=6):
    cutoff = (wakeup_time + num_h) % 24
    if cutoff > wakeup_time:
        return t > cutoff or t < wakeup_time
    else:
        return cutoff < t < wakeup_time


def calculate_c_r_for_index(df, func, index):
    c_r = pd.DataFrame(columns=df.columns, index=index)
    for c in df.columns:
        c_r[c] = func(df[c])
    return c_r


def mannkendall_slope_p(col):
    args = mk.original_test(col)
    return args.slope, args.p


def find_first_inflection(r):
    slopes = r.diff()[1:] / np.diff(
        r.index.get_level_values(level="min_bins15").values)  # first value becomes NaN after diff
    inflection_points = np.sign(slopes.diff()).diff()
    inflection_times = inflection_points.loc[inflection_points < 0].index.get_level_values(level="min_bins15")
    return inflection_times.min() - 0.5  # make up for the diffs


def find_first_peak(r):
    peaks, _ = find_peaks(r.values.squeeze(), height=0)
    peak_times = r.iloc[peaks].index.get_level_values(level="min_bins15")
    return peak_times.min()


def get_min_activity(r, cluster_min, level="Cluster"):
    return cluster_min["min_activity"].loc[r.index.get_level_values(level=level)[0]]


def realign_df_by_cluster_stats_column(df, cluster_stats, c="Cluster", isindex=True):
    if isindex:
        tomapc = df.index.get_level_values(level=c)
        tomapt = df.index.get_level_values(level="min_bins15")
    else:
        tomapc = df.loc[df[c].isin(cluster_stats.index), c]
        tomapt = df.loc[df[c].isin(cluster_stats.index), "min_bins15"]

    aligned_s = tomapc.map(lambda x: cluster_stats.loc[x] if x in cluster_stats.index else 0)
    aligned_s = tomapt - aligned_s
    return aligned_s % 24


def find_hours_spent_awake(df, cluster_stats, c="Cluster", isindex=True):
    df["min_activity"] = realign_df_by_cluster_stats_column(df, cluster_stats["min_activity"], c=c, isindex=isindex)
    df["max_activity"] = realign_df_by_cluster_stats_column(df, cluster_stats["max_activity"], c=c, isindex=isindex)
    df["first_inflection"] = realign_df_by_cluster_stats_column(df, cluster_stats["first_inflection"], c=c,
                                                                isindex=isindex)
    df["first_peak"] = realign_df_by_cluster_stats_column(df, cluster_stats["first_peak"], c=c, isindex=isindex)
    df["steepest_ascent"] = realign_df_by_cluster_stats_column(df, cluster_stats["steepest_ascent"], c=c,
                                                               isindex=isindex)
    df["min_to_first_inflection"] = realign_df_by_cluster_stats_column(df, (
            cluster_stats["min_activity"] + cluster_stats["first_inflection"]) / 2, c=c, isindex=isindex)
    df["waking_time"] = realign_df_by_cluster_stats_column(df, cluster_stats["waking_time"], c=c, isindex=isindex)


def period_averages(arr, periods):
    cutlen = math.floor(len(arr) / periods) * periods
    circa_st = arr[-cutlen:].reshape(int(cutlen / periods), periods).transpose()
    return np.average(circa_st, axis=1)


def confidence_mean_of_population(x, confidence=0.95):
    if len(x.dropna()) > 1:
        dist = NormalDist.from_samples(x.dropna())
        z = NormalDist().inv_cdf((1 + confidence) / 2.)
        return dist.stdev * z / ((len(x.dropna()) - 1) ** .5)
    return None


def min_max_scale(df, to_scale, scale_by):
    df_min_bins = df.groupby(level=scale_by, sort=False)
    return df[to_scale] / df_min_bins[to_scale].transform(sum)


def pairwise_ks_helper(df, condition_col, check_cols):
    conditions = df[condition_col].unique()

    index = ((c1, c2) for i, c1 in enumerate(conditions) for c2 in conditions[i + 1:])
    index = pd.MultiIndex.from_tuples(index, names=['c1', 'c2'])
    p_vals = pd.DataFrame(None, index=index, columns=check_cols)

    for i, c1 in enumerate(conditions):
        for c2 in conditions[i + 1:]:
            for check_col in check_cols:
                _, pvalue = stats.kstest(df.loc[df[condition_col] == c1, check_col],
                                         df.loc[df[condition_col] == c2, check_col])
                p_vals.loc[[(c1, c2), ], check_col] = pvalue
    return p_vals


def pairwise_proportion_helper(tab, check_cols, total_col, index_cols):
    index = ((c1, c2) for i, c1 in enumerate(index_cols) for c2 in index_cols[i + 1:])
    index = pd.MultiIndex.from_tuples(index, names=['c1', 'c2'])
    p_vals = pd.DataFrame(None, index=index, columns=check_cols)
    for label, items in tab[check_cols].items():
        for i, (c1, v1) in enumerate(items.items()):
            for c2, v2 in items[i + 1:].items():
                totals = tab.loc[[c1, c2], total_col]
                stat, pvalue = proportion.proportions_ztest(count=[v1, v2], nobs=totals, alternative="two-sided")
                p_vals.loc[[(c1, c2), ], label] = pvalue
    return p_vals


def pretty_print_diagonal_p_vals(df, caption, p_threshold=0.05, p_format='{:e}', add_default_to_caption=True):
    df = df.unstack(level=1)
    df = df.iloc[df.isnull().sum(axis=1).argsort(), df.isnull().sum(axis=0).mul(-1).argsort()]
    return pretty_print_p_vals(df, caption, p_threshold, p_format, add_default_to_caption)


def pretty_print_p_vals(df, caption, p_threshold=0.05, p_format='{:e}', add_default_to_caption=True):
    if add_default_to_caption:
        caption += f" Insignificant values (p>{p_threshold}) are greyed out."
    return (df.style
            .applymap(lambda v: 'opacity: 20%;' if v > p_threshold else None)
            .format(p_format)
            .set_caption(caption))


def str_to_date(date_str):
    return date(*[int(i) for i in date_str.split("-")])


def inverse_min_max(c):
    c = c.pow(-1)
    return (c - c.min()) / (c.max() - c.min())


def get_store_name(label, date_from, date_to):
    return f"{label}_{date_from}_{date_to}".replace("-", "")


def adfuller_helper(series):
    dftest = adfuller(series)
    dfout = pd.Series(dftest[0:4], index=[
        'ADF Test Statistic',
        'p-value',
        '# of Lags Used',
        '# of Observations'
    ])
    for k, v in dftest[4].items():
        dfout[f'critical value ({k})'] = v

    print(dfout.to_string())

    if dftest[1] <= 0.05:
        print("Reject the null hypothesis: Data has no unit root and is stationary.")
    else:
        print("Weak evidence against the null hypothesis: Data has unit root and is non-stationary.")


def granger_helper(df, lags):
    res = grangercausalitytests(df, lags, verbose=False)
    ssr_ftest_out = pd.DataFrame(index=range(1, lags), columns=("F-value", "p-value", "significant"))
    for lag_num, (test_stats, ols_estimation) in res.items():
        ssr_ftest_out.loc[lag_num, "F-value"] = test_stats["ssr_ftest"][0]
        ssr_ftest_out.loc[lag_num, "p-value"] = test_stats["ssr_ftest"][1]
        ssr_ftest_out.loc[lag_num, "significant"] = test_stats["ssr_ftest"][1] <= 0.05

    print(ssr_ftest_out.to_string())

    significant_ftest_lags = ssr_ftest_out.loc[ssr_ftest_out["significant"] is True].index
    if len(significant_ftest_lags):
        print("Significant p-values: ")
        print(significant_ftest_lags)
    else:
        print("Weak evidence against the null hypothesis: No significant p-values could be found")


def z_norm(df, column):
    return (df[column] - df[column].mean()) / df[column].std()


def xs_repeat(df, label, level, repeat_times):
    sl = df.xs(label, level=level)
    return sl.loc[sl.index.repeat(repeat_times)]


def correct_utc(x):
    return str(pytz.utc.localize(x, is_dst=None).astimezone('CET'))[:-6]


def ceil(dt):
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / 900) * 900 - nsecs
    # time + number of seconds to quarter hour mark.
    return dt + datetime.timedelta(seconds=delta)


def get_sun(row, country_config):
    sun = Sun(row.lat, row.long)
    tz = pytz.timezone(country_config.TIME_ZONE)
    abd_sr = sun.get_local_sunrise_time(row.local_time, local_time_zone=tz)
    abd_ss = sun.get_local_sunset_time(row.local_time, local_time_zone=tz)
    sun_so_far_m, _ = divmod((row.local_time - abd_sr).seconds, 60)
    sun_left_m, _ = divmod((abd_ss - row.local_time).seconds, 60)
    total_sun_m, _ = divmod((abd_ss - abd_sr).seconds, 60)
    return sun_so_far_m, sun_left_m, total_sun_m


def get_lims(df, mean_col, conf_col):
    return df[mean_col] - 1.96 * df[conf_col], df[mean_col] + 1.96 * df[conf_col]
