# Early morning hour and evening usage habits increase misinformation-spread

This workbook contains the code for the paper "Early morning hour and evening usage habits increase misinformation-spread" by Elisabeth Stockinger, Riccardo Gallotti and Carina I. Hausladen, currently in review.

An aggregated and anonymized version of the dataset used within this analysis is provided in the folder 'data'. To preserve user anonymity, the latitudes and longitudes provided in the data files are mapped to the closest 2021 territorial units level 3 (NUTS3, provinces and metropolitan cities).


Most logic is packaged in 'diurnal_misinformation'. Please install the package to execute the notebooks.

The results data analysis and visualization are provided in the folder 'notebooks'.


The file [Italy_clustering.ipynb](notebooks/Italy_clustering.ipynb)  details and background on the procedure used to separate users into clusters based on their typical activity patterns. It contains Supplementary Table 4b.

The file [Italy_fourier.ipynb](notebooks/Italy_fourier.ipynb) smoothes the curves of average daily activity and ratio of potentially disinformative content per cluster. It contains Figure 2, Table 1, Supplementary Figures S1 and S4, and Supplementary Tables S2 and S3.

[Italy_curve_distances.ipynb] aligning activity and disinformative content curves by curve features and comparing distance based on realignment. It contains Supplementary Table S4a.

[Italy_day_night.ipynb](notebooks/Italy_day_night.ipynb) compares ratios of potentially disinformative content during different conceptions of daytime and nighttime. It contains Figure 3 and Table 3.

[Italy_diurnal_plot.ipynb](notebooks/Italy_diurnal_plot.ipynb) provides the polar plots of Figure 4, showcasing a broad overview of posting habits for each cluster.

[Italy_lockdown.ipynb](notebooks/Italy_lockdown.ipynb) provides Table 4, comparing the change of posting behavior during lockdown as opposed to outside of lockdown.

The file [Italy_analysis.ipnyb](notebooks/Italy_analysis.ipynb) contains Table 2, Supplementary Figure S2, and Supplementary Tables S1 and S6.

[Germany_clustering.ipynb](notebook/Germany_clustering.ipynb) applies the same clustering procedure to Germany as has been applied to Italy, and provides Supplementary Table S5a.

[Germany_fourier.ipynb](notebooks/Germany_fourier.ipynb) provides Supplementary Figure 3 and Supplementary Table 5b.