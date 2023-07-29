# Diurnal Patterns in the Spread of COVID-19 Misinformation on Twitter within Italy

This workbook contains the code for the paper "Diurnal Patterns in the Spread of COVID-19 Misinformation on Twitter within Italy." A preprint is published as [http://arxiv.org/abs/2307.11575](http://arxiv.org/abs/2307.11575).

The access to the data necessary for running thes analyses are restricted by Twitter ToC. Requests to perform a reproduction of this work have to be addressed to the 2nd author Riccardo Gallotti.


The file ['code/Italy_clustering.ipynb'](code/Italy_clustering.ipynb) gives details and background on the procedure used to separate users into clusters based on their typical activity patterns. 
It outlines details the identification and removal of users exhibiting suspicious, bot-like activity from subsequent analysis.
The clustering procedure has been repeated for verified and unverified users, and identifies significant differences in activity patterns in between the two groups.

The file ['code/Italy_fourier.ipynb'](code/Italy_fourier.ipynb) smoothes the curves of average daily activity and ratio of potentially machinated content per cluster extracted in ['code/Italy_clustering.ipynb'](code/Italy_clustering.ipynb).
Finally, the file ['code/Italy_analysis.ipnyb'](code/Italy_analysis.ipynb) contains the statistical tests and plots used in the paper not covered elsewhere.

The files ['code/Italy_fourier_unverified.ipynb'](code/Italy_fourier_unverified.ipynb) and ['code/Italy_analysis.ipnyb'](code/Italy_analysis.ipynb) repeat the same analyses when considering only unverified clusters.
Similarly, the files ['code/Germany_clustering.ipynb'](code/Germany_clustering.ipynb), ['code/Germany_fourier.ipynb'](code/Germany_fourier.ipynb) and ['Germany_analysis.ipynb'](code/Germany_analysis.ipynb) repeat the analysis for data from Germany.

## Figures
The figures contained in the paper are plotted in the following locations:
- Figure 2 a-b: [code/Italy_fourier.ipynb: Activity: Multiple-frequency decomposition](code/Italy_fourier.ipynb#Figure2a)
- Figure 2 c-d: [code/Italy_fourier.ipynb: Ratio of potentially machinated content: Multiple-frequency decomposition](code/Italy_fourier.ipynb#Figure2c)
- Figure 3: [code/Italy_analysis.ipynb: Sunlight](code/Italy_analysis.ipynb#Figure3)
- Figure 4: [code/Italy_analysis.ipynb: Intercluster Variation: Diurnal Variation](code/Italy_analysis.ipynb#Figure4)
- Supplementary Figure 1: [code/Germany_fourier.ipynb: Germany VS Italy](code/Germany_fourier.ipynb#SupplFig1a)
- Supplementary Figure 2: [code/Italy_fourier_unverified.ipynb: Unverified VS all](code/Italy_fourier_unverified.ipynb#SupplFig2a)

## Tables
- Table 1: [code/Italy_fourier.ipynb: Statistics: Distribution size of ratios of potentially machinated content across clusters](code/Italy_fourier.ipynb#Table1)
- Table 2: [code/Italy_analysis.ipynb: Inter-cluster variation: Correlation of user activity with ratio of potentially machinated content](code/Italy_analysis.ipynb#Table2)
- Table 3: [code/Italy_analysis.ipynb: Sunlight: Statistics](code/Italy_analysis.ipynb#Table3)
- Table 4: [code/Italy_analysis.ipynb: Lockdown and harmful content](code/Italy_analysis.ipynb#Table4)
- Supplementary Table 2: [code/Italy_analysis.ipynb: Corpus, general](code/Italy_analysis.ipynb#SupplTab2)
- Supplementary Table 3: [code/Italy_fourier.ipynb: Activity: Multiple-frequency decomposition](code/Italy_fourier.ipynb#SupplTab3)
- Supplementary Table 4: [code/Italy_fourier.ipynb: Bimodality](code/Italy_fourier.ipynb#SupplTab4)
- Supplementary Table 5: [code/Italy_fourier.ipynb: Activity: Multiple-frequency decomposition](code/Italy_fourier.ipynb#SupplTab5)
- Supplementary Table 6: [code/Italy_analysis.ipynb: Inter-cluster variation: Content types: Smoothed](code/Italy_analysis.ipynb#SupplTab6)
- Supplementary Table 7: [code/Italy_analysis.ipynb: Inter-cluster variation: Content types: Smoothed](code/Italy_analysis.ipynb#SupplTab7)
- Supplementary Table 8:  [code/Italy_fourier.ipynb: Ratio of potentially machinated content: Multiple-frequency decomposition](code/Italy_fourier.ipynb#SupplTab8)
- Supplementary Table 9: [code/Italy_clustering.ipynb: Pseudo-Chronotypes: Cluster statistics](code/Italy_clustering.ipynb#SupplTab9)
- Supplementary Table 10: [code/Italy_clustering.ipynb: Pseudo-Chronotypes: Cluster statistics](code/Italy_clustering.ipynb#SupplTab10)
- Supplementary Table 11: [code/Italy_analysis.ipynb: All and unverified](code/Italy_analysis.ipynb#SupplTab11)
- Supplementary Table 12: [code/Germany_clustering.ipynb: Pseudo-Chronotypes: Cluster statistics](code/Germany_clustering.ipynb#SupplTab12)
- Supplementary Table 13: [code/Germany_fourier.ipynb: Activity](code/Germany_fourier.ipynb#SupplTab13)
- Supplementary Table 14: [code/Germany_fourier.ipynb: Activity](code/Germany_fourier.ipynb#SupplTab14)
- Supplementary Table 15: [code/Germany_fourier.ipynb: Ratio of potentially machinated content](code/Germany_fourier.ipynb#SupplTab15)
- Supplementary Table 16: [code/Italy_analysis.ipynb: Inter-cluster variation: Correlation of user activity with ratio of potentially machinated content](code/Italy_analysis.ipynb#SupplTab16)
  