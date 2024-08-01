import pandas as pd
import numpy as np
import math
from collections.abc import Iterable 


def groupby_with_total(df, group_by, *agg_args, margins_name='total', **agg_kwargs):
    if isinstance(group_by, Iterable) and not isinstance(group_by, str) and len(group_by) > 1:
        return groupby_multiple_with_total(df, group_by, *agg_args, margins_name=margins_name, **agg_kwargs)
    else:
        df_aggr = df.groupby(group_by).agg(*agg_args, *agg_kwargs)
        df_aggr.loc[margins_name] = df.agg(*agg_args, *agg_kwargs)
        return df_aggr 


def groupby_multiple_with_total(df, group_by, *agg_args, total_for_idx, margins_name, **agg_kwargs):
    df_grouped = df.groupby(by=group_by).agg(*agg_args, **agg_kwargs)
    totals = pd.concat({margins_name: df
            .groupby([i for j, i in enumerate(group_by) if j != total_for_idx % len(group_by)])
            .agg(*agg_args, **agg_kwargs)
        }, names=[df_grouped.index.names[total_for_idx]]
    ).reorder_levels(df_grouped.index.names)
    return pd.concat([df_grouped, totals]).sort_index()


def weighted_mean(df, values, weights, groupby):
    df = df.copy()
    grouped = df.groupby(groupby)
    df[values] = df[values].div(grouped[weights].transform('sum'), axis=0).mul(df[weights], axis=0)
    return grouped[values].sum(min_count=1) 



def smooth_looped(df, std=3, padding=3):
    return (pd.concat([df.iloc[:, -padding:], df, df.iloc[:, :padding]], axis=1)
        .rolling(window=padding*2, win_type='gaussian', center=True, axis=1)
        .mean(std=std).iloc[:, padding:-padding]
)

def time_past_t(timepoint, t):
    return (timepoint - t) % 24


def highlight_diag(df, offset=0):
    a = np.full(df.shape, '', dtype='<U24')
    np.fill_diagonal(a[offset:], 'font-weight: bold')
    return a


def shift_rows_by(df, shift_dict):
    return df.apply(lambda v: pd.Series(data=np.roll(v.to_numpy(), -v.index.get_loc(shift_dict[v.name])), index=df.columns), axis=1)


def hours_to_mins(time, format_ampm=True):
    hours = int(time)
    minutes = (time * 60) % 60
    if format_ampm:
        ampm = 'AM' if (hours < 12 or hours == 24)  else 'PM'
        hours =  12 if hours % 12 == 0 else hours % 12
        return hours, minutes, ampm
    else:
        return hours, minutes
    
    
def format_h_min(time, type="ampm"):
    res = hours_to_mins(time, format_ampm=type=="ampm")
    if type=="ampm":
        return f'{res[0]:.0f}:{res[1]:>02.0f} {res[2]}'
    elif type=="durationvarsize":
        return fr'${res[0]:.0f}^{{\text{{h}}}} {res[1]:>02.0f}^{{\text{{min}}}}$'
    elif type=="duration":
        return f'{res[0]:.0f}h {res[1]:>02.0f}min'
    elif type=="digital":
        return f'{res[0]:.0f}:{res[1]:>02.0f}'
    else:
        raise NotImplementedError("Unknown Type")


def round_up_to_1(x):
    num_figs = int(math.floor(math.log10(abs(x))))
    return math.ceil(x / 10 ** num_figs) * 10 ** num_figs


def within_n_hours_before_or_after_t(x, t, n, include_edges=False):
    return t_between(x, (t - n) % 24, (t + n) % 24, include_edges=include_edges)


def t_between(t, start, cutoff, include_edges=False):
    res = start < t < cutoff if cutoff > start else t > start or t < cutoff
    if include_edges:
        res = res or math.isclose(t, start) or math.isclose(t, cutoff)
    return res