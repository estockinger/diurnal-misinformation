# diurnal_misinformation

This packages accompanies the paper "Early morning hour and evening usage habits increase misinformation-spread" by Elisabeth Stockinger, Riccardo Gallotti and Carina I. Hausladen, currently in review.
A preprint is published as [http://arxiv.org/abs/2307.11575](http://arxiv.org/abs/2307.11575).

This package contains functions used throught the data analysis and visualization of the paper.

The file 'config.py' specifies the relative data path references. 'path_utils.py' provides easy access to loading and storing data. 'enums.py' define the Column names used in the main data file as well as the types of content and clusters.

The file [data_processor.py](diurnal_misinformation/data_processor.py) defines the DataProcessor class, which is concerned with data loading and basic manipulation.

The file [cluster_utils.py](diurnal_misinformation/cluster_utils.py) provides a gateway to selecting a clustering method and a number of clusters based on a set of performance indicators, clustering the input dataset accordingly, and storing the results as a parquet file. Many other utlilities in this package depend on the availability of the clustering results provided here.

The file [fourier_utils.py](diurnal_misinformation/fourier_utils.py) provides function to decompose a signal into its constituent sinusoidal functions, and to recomposed a smoothed version of the signal from the sinosoidals with the largest amplitudes. The FourierRoutine provided in this file is used in various locations in this package to smooth signals.

The file [diurnal_plot_routine.py](diurnal_misinformation/diurnal_plot_routine.py) is a context-specific gateway to the plotting functions defined in 'diurnal_plot.py'. These functions build a compound polar plot merging stack plots, communlative line plots, and inner and outer indicator arcs around a 24-hour clock. This allows the visualization of several diurnal functions simultaneously.

The file [day_night_comparison.py](diurnal_misinformation/diurnal_plot_routine.py) distinguishes between different concepts of day and night ((a)defined by clock time, (b) defined by the presence of sunlight, and (c) defined by inferred waking time) and compares the ratio of potentially disinformative content posted during day and night per clustere according to these definitions. It is also  a gateway to 'heatmap_plots.py' which is a utility plotting function for a temporal heatmap augmented with time annotations. 

The file [similarity_utils.py](diurnal_misinformation/similarity_utils.py) provides utility functions for aligning signals by curve features and comparing distance based on realignment.

In [lockdown_utils.py](diurnal_misinformation/lockdown_utils.py), there are utilities to find the change of amount and ratios of content created daily per cluster during the lockdown period in Italy.

[utils.py](diurnal_misinformation/utils.py) hosts general utility functions used throughout the package.