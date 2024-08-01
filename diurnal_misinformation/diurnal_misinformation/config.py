from dataclasses import dataclass
from typing import ClassVar

@dataclass
class Config:
    ROOT_DIR: ClassVar[str] = ".."
    PLOT_DIR: ClassVar[str] = "plots"
    DATA_DIR: ClassVar[str] = "data"
    STATS_DIR: ClassVar[str] = "stats"
    
    aggr_df_filename = 'reduced_df_nut3'
    cluster_col_filename = 'cluster_col_all'
 
    date_from: str = '2020-01-22'
    date_to: str = '2022-08-01'


@dataclass
class CountryConfig:
    LABEL: str
    TIME_ZONE: str
    FILE_PATH: str = None
    NUTS_COUNTS_CODE:str = None


@dataclass
class ItalyConfig(CountryConfig, Config):
    LABEL:str = "ITA"
    TIME_ZONE:str = "Europe/Rome"
    NUTS_COUNTS_CODE:str = "IT"



@dataclass
class GermanyConfig(CountryConfig, Config):
    LABEL:str = "DEU"
    TIME_ZONE:str = "Europe/BERLIN"
    NUTS_COUNTS_CODE:str = "DE"

