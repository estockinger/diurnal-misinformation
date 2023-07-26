import asyncio, math, warnings
from collections import namedtuple
from collections.abc import Iterable

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

import scripts.constants as sconstants
from scripts.plotting import *
from scripts.utils import *
from scripts.enums import Columns

default_patterns = '// \\\\ || / \\ | - -- ++ xx oo OO .. ** * o O + x .'.split(" ")
HueY = namedtuple("HueY", "hue y")


def get_theme_fig_ax(flen, perline, mul, collength):
    shared_args = {"subplot_kw": {"projection": "polar"}, "sharey": False}
    if flen > perline:
        lines_per_col = math.ceil(flen / perline)
        fig, axs = plt.subplots(
            nrows=lines_per_col * collength, ncols=perline,
            figsize=(perline * mul, lines_per_col * collength * mul),
            **shared_args)
        return fig, axs
    else:
        return plt.subplots(collength, flen, figsize=(perline * mul, mul * collength), **shared_args)


def plot_clock_nums(ax, r_offset, scale=0.85):
    """
    Write the clockface numbers 0, 6, 12, 18 in the center of the theme circle.
    """
    shared_args = {"transform": ax.transData._b, "ha": "center", "va": "center",
                   "fontsize": 12, "fontweight": "bold", "color": "black",
                   "alpha": 1}
    radius = r_offset * scale

    def get_xy(angle):
        return r_offset * scale * math.sin(angle), r_offset * scale * math.cos(angle)

    for a, t in zip(np.arange(0, 2, .25), range(0, 24, 3)):
        plt.text(*get_xy(a * math.pi), str(t), **shared_args)


def polar_twin(ax):
    # this does NOT work when locations/positions are changed afterwards, e.g. with tight_layout or with colorbars
    ax2 = ax.figure.add_axes(
        ax.get_position(),
        projection='polar',
        label='twin',
        frameon=False,
        theta_direction=ax.get_theta_direction(),
        theta_offset=ax.get_theta_offset()
    )
    ax.set_xticklabels([])
    ax2.set_rlabel_position(180 - ax.get_rlabel_position())

    # Ensure that original axes tick labels are on top of plots in twinned axes
    for label in ax.get_yticklabels():
        ax2.figure.texts.append(label)
    return ax2


def scale_ax_by_area(ax, max_value, r_offset):
    """
    y-scaling so that the projection preserves the ratios of areas instead of the ratio of lengths
    """
    r2 = max_value - r_offset
    alpha = r2 - r_offset
    v_offset = r_offset ** 2 / alpha

    def forward(value):
        return ((value + v_offset) * alpha) ** 0.5 + r_offset

    def reverse(radius):
        return (radius - r_offset) ** 2 / alpha - v_offset

    ax.set_yscale('function', functions=(
        lambda value: np.where(value > 0, forward(value), value),
        lambda radius: np.where(radius > 0, reverse(radius), radius)
    ))


def modify_ax(ax, max_value, r_offset, project_area=False, axisbelow=False, yticknr=5, xticknr=24, color='white',
              fmt='%.4g', **grid_kwargs):
    ax.set_theta_direction(-1)  # 0 on top
    ax.set_theta_zero_location("N")  # clockwise
    ax.grid(**grid_kwargs, linewidth=.5, axis='both', color=color)
    ax.set_thetagrids(np.linspace(0, 360, xticknr, endpoint=False), color=color, visible=True)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(yticknr))
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize='small', alpha=1, color='black')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
    ax.yaxis.get_major_ticks()[0].label1.set_visible(False)  # Remove the tick at 0
    bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.5, pad=0.1)
    plt.setp(ax.get_yticklabels(), bbox=bbox, fontsize='small', alpha=1, color='black')

    next_sign_fig = round_up_to_1(max_value)
    ax.set_rlim(0, next_sign_fig)
    ax.set_rgrids(np.linspace(0, next_sign_fig, yticknr, endpoint=True))
    ax.set_rorigin(r_offset)

    ax.set_axisbelow(axisbelow)
    ax.set_frame_on(False)

    if project_area:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scale_ax_by_area(ax, max_value, r_offset)
            ax.set_rgrids(np.linspace(0, next_sign_fig, yticknr, endpoint=True),
                          alpha=1)  # this is overwritten by the transform, so reset


def get_susceptible_df(df, susceptible_col="ratio_norm", smooth=True, window_size=10, pad_by=5, std=3):
    y = (df[susceptible_col]
         .unstack(level=sconstants.f_col)[sconstants.harmful_facttypes]
         .sum(axis=1)
         .unstack(level=sconstants.c_col)
         .reindex(df.index.unique(level=0), fill_value=0))
    if smooth:
        return (pd.concat([y.iloc[-pad_by:], y, y.iloc[:pad_by]])
                .rolling(window=window_size, win_type="gaussian", center=True)
                .mean(std=std)
                .iloc[pad_by:len(y) + pad_by])
    else:
        return y


def fill_susceptible_times(ax, x, y, ylims, quantile=0.25,
                           fc='mistyrose', lw=0, **kwargs):
    where = y > y.quantile(1 - quantile)
    ax.fill_between(np.append(x, [x[0]]), *ylims, where=np.append(where, [where[0]]),
                    facecolor=fc, lw=lw, zorder=0)


def annotate_clockhand(ax, theta, ls='-', lw=4, short=True):
    xy_l = 0 if short else .07
    ax.annotate('', xy=(0.5, 0.5),
                xytext=(theta, xy_l), xycoords='axes fraction', textcoords='data', color='black',
                arrowprops=dict(arrowstyle='<-', color='black', lw=lw, linestyle=ls),
                horizontalalignment='center', verticalalignment='center')


def plot_cumulative_lines(ax, hue_y, totalprevs, usecol, config, hatch=False, hatchpatterns={}):
    # remove the max(..) for a normal stacked plot
    y2 = hue_y.y[usecol].values.flatten()
    y2 = totalprevs + y2

    x2, y2 = np.append(config.polar_t, [config.polar_t[0]]), np.append(y2, [y2[0]])
    hue_c = colors.rgb2hex(config.palette.get(hue_y.hue))
    ax.plot(x2, y2, color='white', lw=3)
    ax.plot(x2, y2, color=hue_c)

    if hatch:
        kwargs = dict(edgecolor=hue_c, hatch=hatchpatterns.get(hue_y.hue, ''), lw=0, facecolor='none', alpha=0.5)
        if np.sum(totalprevs):
            ax.fill_between(x2, np.append(totalprevs, [totalprevs[0]]), y2, **kwargs)
    return y2[:-1]


def line_func(ax, ys, config, line_col, max_h, share_h, line_order=None, **kwargs):
    totalprevs = np.full_like(ys[0].y[line_col], 0)
    ax2 = polar_twin(ax)

    if line_order is not None:
        ys = sorted([y for y in ys if y.hue in line_order], key=lambda hY: list(line_order).index(hY.hue))
    for curr in ys:
        totalprevs = plot_cumulative_lines(
            ax=ax2,
            config=config,
            hue_y=curr,
            totalprevs=totalprevs,
            usecol=line_col,
            **kwargs)

    max_value = max_h if share_h else max(totalprevs)
    r_offset = -0.5 * max_value
    modify_ax(ax2, max_value, r_offset, project_area=config.project_area, color='white', linestyle=':', fmt='%.4f')
    ax2.tick_params(axis='y', direction='out', color='black', grid_alpha=1, pad=-20)
    return ax2, max_value


def hatch_bars(ax, x, hue_y, totalprevs, usecol, width, palette, hatch, hatch_order, hatch_patterns):
    kwargs = dict(
        color=colors.rgb2hex(palette.get(hue_y.hue)),
        width=width,
        linewidth=0,
        ecolor='white',
        bottom=(0 if totalprevs is None else max(totalprevs))
    )
    bars = [ax.bar(x,
                   hue_y.y.xs(h, level=hatch)[usecol].values.flatten(),
                   **kwargs) for h in hatch_order
            ]
    for bar, h in zip(bars, hatch_order):
        for patch in bar:
            patch.set_hatch(hatch_patterns[h])
    y = hue_y.y[usecol].values.flatten()
    return y if totalprevs is None else totalprevs + y


def simple_bars(ax, x, hue_y, totalprevs, usecol, width, palette, **kwargs):
    # remove the max(..) for a normal stacked plot
    y = hue_y.y[usecol].values.flatten()
    ax.bar(x, height=y,
           bottom=0 if totalprevs is None else max(totalprevs),
           color=colors.rgb2hex(palette.get(hue_y.hue)),
           width=width,
           linewidth=0)
    return y if totalprevs is None else totalprevs + y


def simple_area_stack(ax, hue_y, totalprevs, usecol, config, **kwargs):
    y2 = hue_y.y[usecol].values.flatten()
    max_totalprevs = 0
    if totalprevs is not None:
        max_totalprevs = np.append(totalprevs, [totalprevs[0]])
        y2 = y2 + totalprevs

    x2, y2 = np.append(config.polar_t, [config.polar_t[0]]), np.append(y2, [y2[0]])
    hue_c = colors.rgb2hex(config.palette.get(hue_y.hue))

    kwargs = dict(facecolor=hue_c, lw=0)
    if s := np.sum(totalprevs):
        ax.fill_between(x2, max_totalprevs, y2, **kwargs)
        if s > 100:
            ax.plot(x2, y2, color='white', lw=1)

    return y2[:-1]


def area_stack(ax, x, hue_y, totalprevs, usecol, palette, **kwargs):
    y2 = hue_y.y[usecol].values.flatten()
    max_totalprevs = 0 if totalprevs is None else max(totalprevs)
    y2 = y2 + max_totalprevs

    x2, y2 = np.append(x, [x[0]]), np.append(y2, [y2[0]])
    hue_c = colors.rgb2hex(palette.get(hue_y.hue))

    kwargs = dict(facecolor=hue_c, lw=1, edgecolor='white')
    ax.fill_between(x2, max_totalprevs, y2, **kwargs)
    return y2[:-1]


def indicate_prolonged_waking(ax, config, waking_times):
    where = config.linear_t.map(lambda i: not t_between(i, *waking_times, include_edges=False))
    modify_ax(ax, 1, 0, project_area=False, axisbelow=False, color='black',
              xticknr=0)  # must be false for plotting single facets
    ax.fill_between(np.append(config.polar_t, [config.polar_t[0]]), .0, .8, where=np.append(where, [where[0]]), lw=0,
                    fc=".9", alpha=1)
    ax.set_ylim([0, 2])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(axis='y', alpha=0)
    return polar_twin(ax)


def sort_ys(df, reindex, hue_col):
    ys = (HueY(
        y=df.xs(hue, level=hue_col).reindex(reindex, fill_value=0),
        hue=hue
    ) for hue in [i for i in df.index.get_level_values(hue_col).unique()] if hue != 'total')
    return sorted(ys, key=lambda hy: sum(hy.y.sum()))


def draw_ax_point(ax, x, textpos, text, extend=1.4):
    ax.axvline(x, .1, extend, color='black', ls="-", lw=.5, zorder=0, clip_on=False)  # longer line to label axis
    ax.text(.5, textpos, text, rotation=0, ha='left', clip_on=False, transform=ax.transAxes)


async def process_facet(
        ax,
        config,
        facet_df,
        bar_col,
        hue_col,
        bar_func,
        max_c,
        max_h=None,
        line_col=None, line_order=None, line_col_kwargs={}, bar_col_kwargs={},
        waking_times=None,
        susceptible_series=None, susceptible_kwargs={},
        share_c=True, share_h=True

):
    ys = sort_ys(facet_df, config.linear_t, hue_col=hue_col)

    if "hatch" in bar_col_kwargs:
        bar_col_kwargs.setdefault("hatch_order",
                                  [hue_y.hue for hue_y in sorted(ys, key=lambda hy: sum(hy.y[line_col]))
                                   if hue_y.hue != 'total'])
        bar_col_kwargs.setdefault("hatch_patterns", {h: p for (h, p) in zip(hatchorder, default_patterns)})

    if waking_times:
        ax = indicate_prolonged_waking(ax, config, waking_times)
    draw_ax_point(ax, 0, textpos=1.08, extend=1.4, text=bar_col)

    totalprevs = None
    for curr in ys:
        totalprevs = bar_func(
            ax=ax,
            config=config,
            hue_y=curr,
            totalprevs=totalprevs,
            usecol=bar_col,
            **bar_col_kwargs)

    max_value = max_c if share_c and max_c else max(totalprevs)
    r_offset = -0.5 * max_value
    modify_ax(ax, max_value, r_offset, project_area=config.project_area, axisbelow=True, color='black', linestyle=':')
    ax.set_rlabel_position(1)
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_verticalalignment('top')

    plot_clock_nums(ax, -r_offset * .9)

    if susceptible_series is not None:
        fill_susceptible_times(
            ax=ax,
            x=config.polar_t,
            y=susceptible_series,
            ylims=(0, ax.get_ylim()[1]),
            **susceptible_kwargs)

    if line_col is not None:
        draw_ax_point(ax, math.pi, textpos=-.08, extend=1.4, text=line_col)
        ax, max_value = line_func(
            ax=ax,
            ys=ys,
            line_col=line_col,
            config=config,
            max_h=max_h,
            share_h=share_h,
            line_order=line_order,
            **line_col_kwargs)
        ax.yaxis.set_major_formatter('{x:.1f}')

    locations = config.polar_t[::4]
    labels = [f'{i}:00' for i in range(24)]
    bbox = dict(boxstyle="round", ec="white", fc="white", alpha=.9, pad=0)
    ax.set_xticks(locations, labels, fontsize='small', color='black', bbox=bbox)
    ax.tick_params(axis='x', pad=3, direction='in')
    return ax


class ClockfaceConfig:
    def __init__(self, linear_t, palette, project_area=False):
        self.linear_t = linear_t
        self.polar_t = np.linspace(0, 2 * np.pi, len(linear_t), endpoint=False)
        self.width = 2 * np.pi / (len(self.linear_t))
        self.palette = palette
        self.project_area = project_area


async def theme_circle_helper(
        df,
        usecols,
        facet_col,
        hue_col,
        palette,
        susceptible_df=None,
        facetorder=None,
        perline=5,
        mul=4,  # the higher, the smaller font sizes
        waking_times=None,
        susceptible_kwargs=None,
        bar_func=simple_area_stack,
        line_col=None, line_order=None,
        share_h=True,
        project_area=True,
        title=True,
        **facet_kwargs):
    if susceptible_kwargs is None:
        susceptible_kwargs = {}
    if waking_times is None:
        waking_times = {}
    facetorder = df.index.unique(facet_col) if facetorder is None else facetorder
    line_order = df.index.unique(hue_col) if line_order is None else line_order
    fig, axes = get_theme_fig_ax(len(facetorder), perline, mul, collength=len(usecols))

    config = ClockfaceConfig(df.index.levels[0], palette, project_area)

    if line_col is not None:
        if share_h:
            facet_kwargs['max_h'] = (df
                                     .loc[(df.index.get_level_values(level=hue_col).isin(line_order)), line_col]
                                     .groupby(level=(Columns.MIN_BINS15.value, facet_col))
                                     .sum().max())

    tasks = []
    axflat = iter(axes.flatten()) if isinstance(axes, Iterable) else iter([axes])
    for col in usecols:
        max_c = (df.loc[(df.index.get_level_values(level=hue_col) != 'total') & (
                df.index.get_level_values(level=facet_col) != 'total'), col]
                 .groupby(level=(Columns.MIN_BINS15.value, facet_col)).sum().max())
        for f in facetorder:
            ax = next(axflat)
            if title:
                ax.set_title(f"{f}", loc="right", y=1.07, x=.48)
            kwargs = dict(
                config=config,
                facet_df=df.xs(f, level=facet_col),
                susceptible_kwargs=susceptible_kwargs,
                waking_times=waking_times.get(f, None),
                line_col=line_col,
                line_order=line_order,
                max_c=max_c,
                share_h=share_h,
                hue_col=hue_col,
                susceptible_series=susceptible_df[f] if susceptible_df is not None else None
            )
            tasks.append(process_facet(ax=ax, bar_col=col, bar_func=bar_func, **facet_kwargs, **kwargs))

    await asyncio.gather(*tasks)
    return fig, axes
