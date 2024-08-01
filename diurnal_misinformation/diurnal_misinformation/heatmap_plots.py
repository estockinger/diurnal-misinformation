import numpy as np
import seaborn as sns
from matplotlib import pyplot, ticker
from scipy import interpolate
from .utils import format_h_min

months = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split(' ')


def format_months(val, idxs):
    if val >= len(idxs):
        y, m = idxs[-1]
        if m + 1 > 11:
            y, m = y + 1, (m + 1) % 12
    else:
        y, m = idxs[int(val)]
    return f'{y if m == 1 else ""} {months[int(m) - 1]:>3}'


def get_center(frompt, topt, max_x):
    if frompt > topt:
        return frompt + (max_x - frompt) / 2
    return frompt + (topt - frompt) / 2


def buffer_ticks(val, ticklocs, min_d=10, buffer=20):
    distances = [val*4 - t for t in ticklocs]
    t = f'{format_h_min(val)}'
    for d in distances:
        if 0 < d <= min_d:
            return t.rjust(buffer - int(d))
        if 0 > d >= -min_d:
            return t.ljust(buffer + int(d))
    return t


def draw_sunset_sunrise(ax, suntimes, buffer=1, **kwargs):
    x= np.arange(0, suntimes.shape[0])
    x_interp = np.linspace(x[0], x[-1], 100)
    y0s = interpolate.make_interp_spline(x, suntimes)(x_interp)
    sns.lineplot(x=x_interp, y=y0s[:,0], color="none", ax=ax)
    sns.lineplot(x=x_interp, y=y0s[:,1], color="none", ax=ax)
    for line in ax.lines[:2]:
        x1, y1 = line.get_data()
        ax.fill_betweenx(y=x1, x1=y1 - buffer, x2=y1 + buffer, **kwargs)

def add_annotation(
        text, frompt, topt, y, ax, max_x,
        drawvertical=True,
        color="white", xtext = None, 
        shifttext=-.5, 
        textcoords='data',
        fontweight="bold",
        ls=":",
        **kwargs
):
    if drawvertical:
        ax.vlines([frompt, topt], ymin=0, ymax=y, color=color, ls=ls)
    if frompt > topt:
        ax.hlines(y, 0, topt, color=color, ls=ls)
        ax.hlines(y, frompt, max_x, color=color, ls=ls)
    else:
        ax.hlines(y, frompt, topt, color=color, ls=ls)

    if xtext is None:
        xtext = get_center(frompt, topt, max_x)
    bbox = dict(color="white" if color=="black" else "none", boxstyle="square", pad=.1)
    ax.annotate(text=text, xy=(frompt, y + shifttext), xytext=(xtext, y + shifttext), 
                textcoords=textcoords, fontweight=fontweight, color=color, bbox=bbox, **kwargs)


def draw_heatmap(data,
                 sun_times, clock_times, waking_times,
                 max_x=24.,
                 waking_pos=5, clock_pos=3, sunlight_pos=7, 
                 facecolor = "gold", edgecolor="white",
                 **hmargs):
    max_x = max_x*4
    sun_times=sun_times*4
    clock_times= clock_times*4
    waking_times= waking_times*4
    g = sns.heatmap(data=data, **hmargs, mask=data.isna())
    g.set_facecolor(facecolor)
    g.set(xlim=(0., max_x), xlabel="", ylabel="")
    draw_sunset_sunrise(g, sun_times, hatch="--", facecolor='none', lw=0, alpha=.8, edgecolor=edgecolor)
    xtext = max(clock_times.iloc[0], sun_times.iloc[sunlight_pos, 0], waking_times.iloc[0]) + 2

    for (ts, pos, txt) in zip(
        (sun_times.to_numpy()[sunlight_pos] + [1, -1], waking_times.to_numpy(), clock_times.to_numpy()), 
        (sunlight_pos, waking_pos, clock_pos), 
        ("sunlight", "waking time", "day by clock")):
        add_annotation(txt, *ts, y=pos, ax=g, max_x=max_x, color=edgecolor, xtext=xtext, drawvertical=txt!="sunlight")
    

    tlocs = [*clock_times, *waking_times]
    g.axes.xaxis.set_minor_locator(ticker.FixedLocator(locs=tlocs))
    g.axes.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda v, _: buffer_ticks(v/4, tlocs)))
    g.axes.xaxis.set_major_locator(ticker.MultipleLocator(24))
    g.axes.yaxis.set_major_locator(ticker.MultipleLocator(2))
    g.axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: format_h_min(v/4, type="ampm")))
    g.axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: format_months(v, data.index)))

    g.axes.tick_params(axis='x', which='minor', labelsize='small', top=True, bottom=False, labeltop=True,labelbottom=False)
    g.axes.tick_params(axis='x', which='major', rotation=0)
    for tick in (*g.axes.xaxis.get_majorticklabels(), *g.axes.xaxis.get_minorticklabels()):
        if tick.get_position()[0] < 4:
            tick.set_horizontalalignment("left")
        elif tick.get_position()[0] > (max_x - 4):
            tick.set_horizontalalignment("right")
    return g


def plot_heatmap_facets(df, facets, facet_level, sun_times, clock_times, waking_times, subtitles=None,
                        figsize=(20, 10), nrows=1, ncols=None, vmin=None, vmax=None, **kwargs):
    ncols = len(facets) if ncols is None else ncols

    fig, axes = pyplot.subplots(
        nrows, ncols + 1,
        figsize=figsize,
        gridspec_kw={'width_ratios': [40] * ncols + [1]},
        tight_layout=True)
    
    if vmin is None:
        vmin = df.min().min()
    if vmax is None:
        vmax = df.max().max()

    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    for i, (c, ax1, ax2) in enumerate(zip(facets, axes[:, :-1].flat, np.repeat(axes[:, -1], ncols))):
        if subtitles and i < len(subtitles):
            ax1.set_title(subtitles[i])
        draw_heatmap(
            df.xs(c, level=facet_level),
            sun_times = sun_times.xs(c, level=facet_level) if facet_level in sun_times.index.names else sun_times, 
            clock_times = clock_times.loc[c] if c in clock_times.index else clock_times, 
            waking_times = waking_times.loc[c] if c in waking_times.index else waking_times, 
            ax=ax1, cbar_ax=ax2, vmin=vmin, vmax=vmax, **kwargs)

    # for ax in axes[:-1,:].flat:
    # ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
    for ax in axes[:, 1:-1].flat:
        ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False, left=False)
        ax.set_ylabel('')

    return fig, axes
