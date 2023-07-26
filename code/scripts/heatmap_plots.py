import numpy as np
import seaborn as sns
from matplotlib import pyplot, ticker, patheffects
from scipy import interpolate
from scripts.utils import hours_to_mins

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


def buffer_ticks(val, ticklocs, min_d=12, buffer=22):
    distances = [val - t for t in ticklocs]
    t = f'{hours_to_mins(val / 4)}'
    for d in distances:
        if 0 < d <= min_d:
            return t.rjust(buffer - int(d))
        if 0 > d >= -min_d:
            return t.ljust(buffer + int(d))
    return t


def add_annotation(
        text, frompt, topt, ax, y, max_x,
        color="white", width=1, shifttext=1, **kwargs
):
    foreground = "black" if color == "white" else "white"
    shared_args = dict(
        horizontalalignment='center', textcoords='data', annotation_clip=False, color=color,
        path_effects=[patheffects.withStroke(linewidth=1.5, foreground=foreground)], fontsize="larger",
        #fontweight="bold",
    ) | kwargs

    if frompt > topt:
        ax.hlines(y, 0, topt, color=color, lw=width)
        ax.hlines(y, frompt, max_x, color=color, lw=width)
    else:
        ax.hlines(y, frompt, topt, color=color, lw=width, alpha=1)

    xtext = get_center(frompt, topt, max_x)
    ax.annotate(text=text, xy=(frompt, y + shifttext), xytext=(xtext, y + shifttext), **shared_args)


def draw_sunset_sunrise(ax, x, ys, **args):
    x_interp = np.linspace(x[0], x[-1], 100)
    y0s = [interpolate.make_interp_spline(x, y)(x_interp) for y in ys]
    ax.xaxis.update_units(y0s[0])
    for y in y0s:
        sns.lineplot(x=ax.xaxis.convert_units(x_interp), y=y, ax=ax)
    xy1s = [line.get_data() for line in ax.lines[:len(ys)]]
    ax.clear()
    for x1, y1 in xy1s:
        ax.fill_betweenx(y=x1, x1=y1 - 1, x2=y1 + 1, **args)
    ax.xaxis.update_units(xy1s[0][1])
    ax.set_xlim(0, 24 * 4)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.grid(visible=False)


def indicate_sunset_sunrise(xs, ms="><", **args):
    for x, m in zip(xs, ms):
        sns.scatterplot(x=x, marker=m, **args)


def draw_vlines(ax, times, intersects_with, positions, length, max_x,
                hr_margin=2, shift_by=(1.5, -.5), **kwargs):
    fill = [[0, length], [0, length]]

    for i in range(2):
        for t2, pos in zip(intersects_with, positions):
            if times is not t2:
                if abs(times[i] - get_center(*t2, max_x)) < hr_margin:
                    fill[i] += [pos + shift_by[0], pos + shift_by[1]]
    for t, segments in zip(times, [sorted(f) for f in fill]):
        for frompt, topt in zip(segments[:-1:2], segments[1::2]):
            ax.vlines(t * 4, ymin=frompt, ymax=topt, **kwargs)


def mul_iter(iterable, f=4):
    return [i * f for i in iterable]


def draw_heatmap(column,
                 sun_times, day_times, waking_times,
                 ax, bar_ax, max_x, facecolor, positions=None, edgecolor="white",
                 **hmargs
                 ):
    if positions is None:
        positions = dict(waking=3, clock=1, daylight=6)
    hmargs.update(dict(data=column, ax=ax, cbar_ax=bar_ax))
    g = sns.heatmap(**hmargs, mask=column.isna())
    g.set_facecolor(facecolor)

    ys = [sun_times.sunrise * 4, sun_times.sunset * 4]
    # indicate_sunset_sunrise(xs=ys, y=sun_times.index-0.5, ax=ax, s=40, color=edgecolor)
    draw_sunset_sunrise(
        ax.twiny(), x=sun_times.index - 0.5, ys=ys,
        hatch="////", facecolor="none", lw=0, alpha=.8, edgecolor=edgecolor)

    add_annotation("daylight", *[y[17] for y in ys], ax, y=positions["daylight"], max_x=max_x, color=edgecolor)
    add_annotation("waking time", *mul_iter(waking_times), ax, y=positions["waking"], max_x=max_x, color=edgecolor)
    add_annotation("day by clock", *mul_iter(day_times), ax, y=positions["clock"], max_x=max_x, color=edgecolor)

    times = waking_times, day_times.values, sun_times.loc[positions["daylight"]].values
    for t in times[:-1]:
        draw_vlines(
            ax, times=t, intersects_with=times, positions=positions.values(), max_x=max_x,
            length=len(column), color=edgecolor, ls=":")

    ax.set(xlim=(0, max_x), xlabel="", ylabel="")

    tlocs = mul_iter(waking_times) + mul_iter(day_times)
    g.axes.xaxis.set_minor_locator(ticker.FixedLocator(locs=tlocs))
    g.axes.xaxis.set_major_locator(ticker.FixedLocator(list(range(0, 24 * 4, 4 * 6))))
    g.axes.yaxis.set_major_locator(ticker.MultipleLocator(2))
    g.axes.xaxis.set_minor_formatter(ticker.FuncFormatter(lambda v, i: buffer_ticks(v, tlocs)))
    g.axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, i: f'{hours_to_mins(v / 4)}'))
    g.axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, i: format_months(v, column.index)))
    g.axes.tick_params(axis='x', which='minor', labelsize='small', top=True, bottom=False, labeltop=True,
                       labelbottom=False)
    g.axes.tick_params(axis='x', which='major', rotation=0)
    for tick in g.axes.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("left")


def plot_heatmap_facets(df, columns, sun_times, day_times, waking_times, subtitles,
                        figsize=(20, 10), nrows=1, ncols=None, edgecolor="white", **kwargs):
    ncols = len(columns) if ncols is None else ncols

    fig, axes = pyplot.subplots(
        nrows, ncols + 1,
        figsize=figsize,
        gridspec_kw={'width_ratios': [40] * ncols + [1]},
        tight_layout=True)

    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    def_hmargs = {
        'cmap': 'magma',
        'vmin': df[columns].min().min(),
        'vmax': df[columns].max().max(),
        'facecolor': 'gold',
        'max_x': len(df.index.unique(level=-1)),
    }
    for i, (c, ax1, ax2) in enumerate(zip(columns, axes[:, :-1].flat, np.repeat(axes[:, -1], ncols))):
        if subtitles and i < len(subtitles):
            ax1.set_title(subtitles[i])
        draw_heatmap(
            df[c].unstack(level="min_bins15"),
            sun_times, day_times, waking_times[c], ax1, ax2, edgecolor=edgecolor, **def_hmargs | kwargs)

    # for ax in axes[:-1,:].flat:
    # ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
    for ax in axes[:, 1:-1].flat:
        ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False, left=False)
        ax.set_ylabel('')

    return fig, axes
