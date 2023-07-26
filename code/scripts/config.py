from dataclasses import dataclass, field
from typing import List, ClassVar
from scripts.enums import Columns


@dataclass
class CountryConfig:
    LABEL: str
    TIME_ZONE: str
    FILE_PATH: str = None


@dataclass
class Config:
    SAVE_ROOT_DIR: ClassVar[str] = "../plots/"
    HDF5_STORE: ClassVar[str] = "../store2.h5"
    USERHASH: ClassVar[str] = "private/userhash.pickle"
    SAVE_DATA_DIR: ClassVar[str] = "../data/"
    SAVE_STATS_DIR: ClassVar[str] = "../stats/"

    date_from: str = '2020-01-22'
    date_to: str = '2022-08-01'
    where_clause: str = " "
    time_columns: List = field(default_factory=lambda: [
        Columns.YEAR,
        Columns.MONTH,
        Columns.MIN_BINS15])
    load_columns: List = field(default_factory=lambda: [
        Columns.DATETIME,
        Columns.FACTTYPE_ORIGINAL,
        Columns.TWEET_TYPE,
        Columns.USER,
        Columns.VERIFIED,
        Columns.IS_BOT,
        Columns.COUNTRY_CODE,
        Columns.LAT,
        Columns.LONG])
