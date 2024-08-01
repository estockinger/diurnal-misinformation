import asyncio, math, warnings
from collections import Iterable
from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Optional, Union

from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

from .utils import t_between, format_h_min, round_up_to_1

@dataclass
class DiurnalPlotConfig:
    data: pd.DataFrame
    x: str
    y_stack: str 
    col: str 
    hue: str 
    
    y_line: Optional[str] = None 
    outer_arc: Optional[pd.DataFrame] = None
    inner_arc: Optional[pd.DataFrame] = None
    
    col_order: Iterable = field(default_factory=list)
    line_order: Iterable = field(default_factory=list)

    palette: dict = field(default_factory=dict)
    facet_height: Union[float, int] = 6
    facet_width: Union[float, int] = 6
    ncols: int = 0
    nrows: int = 0

    title:str = ""

    share_h: bool = True
    share_c: bool = True
    project_area: bool = True

    outer_arc_kwargs:dict = field(default_factory={lambda x: dict(quantile=0.25, window_size=3, smooth=True)})

    max_h: float = 0
    max_c: float = 0
    linear_t: np.array = field(init=False)
    polar_t: np.array = field(init=False)
    width: float = field(init=False)

    def __post_init__(self):
        if len(self.col_order) == 0:
            self.col_order = self.data[self.col].unique()
        
        if self.y_line is not None:
            if len(self.line_order) == 0:
                self.line_order = self.data[self.y_line].unique()
            if self.share_h and self.max_h == 0:
                self.max_h = self.data.loc[self.data[self.hue].isin(self.line_order)].groupby(by=[self.x, self.col])[self.y_line].sum().max()

        if self.share_c and self.max_c == 0:
            self.max_c = (self.data.loc[self.data[self.col].isin(self.col_order)].groupby(by=[self.x, self.col])[self.y_stack].sum().max())

        if len(self.palette) == 0:
            cmap = plt.cm.jet
            self.palette = {c: cmap(i) for i,c in enumerate(self.data[self.hue].unique())}

        if not self.ncols and not self.nrows:
            self.ncols, self.nrows = len(self.col_order), 1

        elif not self.ncols:
            self.ncols = math.ceil(len(self.col_order) / self.nrows)

        elif not self.nrows:
            self.nrows = math.ceil(len(self.col_order) / self.ncols)

        self.linear_t = self.data[self.x].unique()
        self.polar_t = np.linspace(0, 2 * np.pi, len(self.linear_t), endpoint=False)
        self.width = 2 * np.pi / (len(self.linear_t))


def polar_twin(ax):
    # this does not work when locations/positions are changed afterwards, e.g. with tight_layout or with colorbars
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

    def forward(v):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # silence runtime warning for values that are filtered out with np.where below
            return ((v + v_offset) * alpha) ** 0.5 + r_offset

    def reverse(radius):
        return (radius - r_offset) ** 2 / alpha - v_offset

    ax.set_yscale('function', functions=(
        lambda value: np.where(value + v_offset > 0, forward(value), value),
        lambda radius: np.where(radius > 0, reverse(radius), radius)
    ))


def modify_ax(ax, max_value, r_offset, project_area=False, axisbelow=False, barticknr=3, thetaticks=None, color='white', fmt='%.4g', **grid_kwargs):
    ax.set_theta_direction(-1)  # 0 on top
    ax.set_theta_zero_location("N")  # clockwise
    if color is not None:
        if thetaticks is None:
            thetaticks=np.linspace(0, 360, 24, endpoint=False)
        ax.grid(**grid_kwargs, linewidth=.5, axis='both', color=color)
        ax.set_thetagrids(thetaticks, color=color, visible=True)
    else:
        ax.grid(axis='both', visible=False)

    next_sign_fig = round_up_to_1(max_value)
    ax.set_rlim(0, next_sign_fig)
    ax.set_rorigin(r_offset)

    rgrid = np.linspace(0, next_sign_fig, barticknr, endpoint=True)

    if barticknr > 0:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(barticknr))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))
        ax.yaxis.get_major_ticks()[0].label1.set_visible(False)  # Remove the tick at 0
        bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.5, pad=0.1)
        plt.setp(ax.get_yticklabels(), bbox=bbox, fontsize='small', alpha=1, color='black')


    ax.set_axisbelow(axisbelow)
    ax.set_frame_on(False)

    if project_area:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scale_ax_by_area(ax, max_value, r_offset)
    
    ax.set_rgrids(rgrid, alpha=1)


def indicate_outer_arc(ax, x, y, ylims, quantile=0.25, fc='salmon', lw=0, smooth=True, window_size=10, **kwargs):
    where = y > y.quantile(1 - quantile)
    if smooth: 
        pad_by = window_size//2
        where = (pd.concat([where.iloc[-pad_by:], where, where.iloc[:pad_by]])
                 .rolling(window=window_size, center=True)
                 .apply(lambda x: x.mode()[0]).iloc[pad_by:-pad_by]
                 .astype(bool))
    
    ax.fill_between(np.append(x, [x[0]]), *ylims, where=np.append(where, [where[0]]),
                    facecolor=fc, lw=lw, zorder=0)
    where_idcs = np.argwhere(np.diff(where, append=where.iloc[0])).flatten()
    for j,i in enumerate(where_idcs):
        where_idcs[j] = i + int(where.iloc[i] == False)
    return where_idcs % len(x)


def annotate_clockhand(ax, theta, ls='-', lw=4, short=True):
    xy_l = 0 if short else .07
    ax.annotate('', xy=(0.5, 0.5),
                xytext=(theta, xy_l), xycoords='axes fraction', textcoords='data', color='black',
                arrowprops=dict(arrowstyle='<-', color='black', lw=lw, linestyle=ls),
                horizontalalignment='center', verticalalignment='center')



def line_func(ax, facet_df, config):
    hueorder = list(filter(lambda l: l in facet_df[config.hue].unique(), config.line_order))

    y = np.full_like(facet_df.loc[facet_df[config.hue]==hueorder[0], config.y_line], 0)
    x = np.append(config.polar_t, [config.polar_t[0]])
    for h in hueorder:
        y += facet_df.loc[facet_df[config.hue]==h, config.y_line].to_numpy().flatten()
        ax.plot(x, np.append(y, [y[0]]), color='white', lw=3)
        ax.plot(x, np.append(y, [y[0]]), color=config.palette.get(h))

    max_value = config.max_h if config.share_h else max(y)
    r_offset = -0.5 * max_value
    modify_ax(ax, max_value, r_offset, project_area=config.project_area, linestyle=':', fmt='%.4f')
    ax.tick_params(axis='y', direction='out', color='black', grid_alpha=1, pad=-20)
    return ax, max_value


def area_stack(ax, facet_df, config):
    hueorder = sorted(facet_df[config.hue].unique(), key=lambda h: facet_df.loc[facet_df[config.hue]==h, config.y_stack].sum())
    x = np.append(config.polar_t, [config.polar_t[0]])
    y = np.zeros(x.shape)
    y_prev = np.zeros(x.shape)
    for h in hueorder:
        hue_df = facet_df.loc[facet_df[config.hue]==h]
        y[:-1][np.isin(config.linear_t, hue_df[config.x])] += hue_df[config.y_stack].to_numpy().flatten()
        y[-1] = y[0]
        color = config.palette.get(h)
        ax.fill_between(x, y_prev, y, facecolor=color, edgecolor= color, lw=0)
        y_prev[:] = y[:]
    return max(y)



def indicate_inner_arc(ax, config, arc_times):
    wherefunc = np.vectorize(lambda i : not t_between(i, *arc_times, include_edges=False))
    where = wherefunc(config.linear_t)
    modify_ax(ax, 1, 0, project_area=False, axisbelow=False, color=None, barticknr=0, thetaticks=[])
    ax.fill_between(np.append(config.polar_t, [config.polar_t[0]]), .0, .7, where=np.append(where, [where[0]]), lw=0, fc=".9", alpha=1)
    ax.set_ylim([0, 2])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(axis='y', alpha=0)
    return polar_twin(ax)


def draw_ax_point(ax, x, extend=1):
    ax.axvline(x, .1, extend, color='black', ls="-", lw=.5, zorder=0, clip_on=True)  # longer line to label axis



async def process_facet(ax, col, config):
    facet_df = config.data.loc[config.data[config.col]==col]
    thetaticklocations = []
    thetaticklabels = []
    if config.inner_arc is not None:
        ax = indicate_inner_arc(ax, config, config.inner_arc.loc[col])
        thetaticklocations = np.append(thetaticklocations, config.polar_t[(config.inner_arc.loc[col] * 4).astype(int)])
        thetaticklabels = np.append(thetaticklabels, config.inner_arc.loc[col].to_numpy())
    draw_ax_point(ax, 0, extend=1.4)

    area_max_value = area_stack(ax, facet_df, config)


    max_value = config.max_c if config.share_c else area_max_value
    r_offset = -0.5 * max_value

    if config.outer_arc is not None:
        change_idcs = indicate_outer_arc(ax=ax, x=config.polar_t, y=config.outer_arc[col], ylims=(max_value*.9, max_value*1.2), **config.outer_arc_kwargs)
        thetaticklocations = np.append(thetaticklocations, config.polar_t[change_idcs])
        thetaticklabels = np.append(thetaticklabels, config.linear_t[change_idcs])
        
    for i in config.linear_t[::4*6]:
        window = np.linspace(i-.5, i+.5, 4,endpoint=False) % 24
        if not any(map(lambda w: w in thetaticklabels, window)):
            thetaticklabels = np.append(thetaticklabels, i)
            thetaticklocations = np.append(thetaticklocations, config.polar_t[int(i*4)])

    modify_ax(
        ax, max_value, r_offset, 
        project_area=config.project_area, axisbelow=True, 
        color='black', linestyle=':',
        barticknr=3,
        thetaticks=np.rad2deg(thetaticklocations))
    ax.set_rlabel_position(1)
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_verticalalignment('top')


    if config.y_line is not None:
        draw_ax_point(ax, math.pi, extend=1.4)
        ax, max_value = line_func(ax=polar_twin(ax), facet_df=facet_df, config=config)
        ax.yaxis.set_major_formatter('{x:.1f}')
        ax.set_ylabel('')

    ax.set_xticks(thetaticklocations, map(format_h_min, thetaticklabels), fontsize='small', color='black')
    ax.tick_params(axis='x', pad=2, direction='out')
    for label,rot in zip(ax.get_xticklabels(),thetaticklocations):
        label.set_rotation(rot*180./np.pi)
        alignment = "left"
        if rot > np.pi:
            alignment = "right"
        if rot == 0 or rot == np.pi*2:
            alignment = "center"
        label.set_horizontalalignment(alignment)
        label.set_rotation_mode("anchor")
    return ax


async def diurnal_plot(**kwargs):
    """ Creates a combination polar plot including a stack plot (y_stack), an optional line plot(y_line), and inner and outer act indicators for time.

    Keyword arguments:
    data: Tidy (long-form) dataframe where each column is a variable and each row is an observation.
    {col, hue}
    """
    config = DiurnalPlotConfig(**kwargs)
    fig, axes = plt.subplots(nrows=config.nrows, ncols=config.ncols, 
                        figsize=(config.ncols * config.facet_width, config.nrows * config.facet_height), 
                        subplot_kw = dict(projection="polar"), sharey = False)

    axflat = axes.flat if isinstance(axes, Iterable) else iter([axes])
    tasks = []
    
    for f in config.col_order:
        ax = next(axflat)
        if config.title:
            ax.set_title(f"{f}", loc="right", y=1.07, x=.48)
        tasks.append(process_facet(ax, f, config))

    await asyncio.gather(*tasks)
    return fig, axes
