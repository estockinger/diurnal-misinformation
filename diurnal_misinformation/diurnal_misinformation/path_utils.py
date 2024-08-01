import os

def get_data_path(label, config):
    return os.path.join(config.ROOT_DIR, config.DATA_DIR, f'{config.LABEL}_{label}.parquet.gzip')


def get_cluster_col_path(config):
    return get_data_path(config.cluster_col_filename, config)


def get_data_file_path(config):
    return get_data_path(config.aggr_df_filename, config)


def save_plot(figure, label, config, type=None, subdir="", transparent=False, ext='eps', **kwargs):
    path = to_path(config, label, config.PLOT_DIR, ext=ext, type=type, subdir=subdir)
    return figure.savefig(path, transparent=transparent, **kwargs)


def to_path(config, label, dir, ext, type=None, subdir=""):
    label = f'{config.LABEL}_{label}' + (f'_{type}' if type else '') + f'.{ext}'.replace(" ", "_")
    return os.path.join(config.ROOT_DIR, dir, subdir, label)


def save_to_latex(config, df_style, label, caption, type=None, is_multi_index=False, subdir="", **kwargs):
    kwargs.update(
        buf=to_path(config, label, config.STATS_DIR, ext='tex', type=type, subdir=subdir),
        position='htb',
        label=f'tab:{label}',
        convert_css=True,
        sparse_index=True,
        caption=caption
    )
    if is_multi_index:
        kwargs.update(
            multicol_align = "c", 
            multirow_align = "l", 
            clines = 'skip-last;index'
        )
    df_style.to_latex(**kwargs)