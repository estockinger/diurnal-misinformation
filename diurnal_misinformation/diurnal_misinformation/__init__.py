import seaborn as sns

rc={
    'figure.figsize': (7,5), 
    'axes.facecolor':'white', 
    'axes.edgecolor': 'black',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor':'white', 
    'xtick.bottom': True,
    'ytick.left': True,
    'xtick.color': 'black',
    'ytick.color': 'black',
    'legend.frameon': False,
    'hatch.linewidth': 1.5
}
sns.set_theme(rc=rc)

import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")
warnings.filterwarnings("ignore", category=FutureWarning)
