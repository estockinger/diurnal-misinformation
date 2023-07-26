import pickle

import pandas as pd
import scripts.rg_infodemics as rgi
from dateutil.relativedelta import relativedelta
from scripts.utils import str_to_date
import hashlib
from scripts.enums import Columns

class APILoader():
    """
    Possible LOAD_COLS columns are:
    lat: double
    User: string
    Timestamp: int64
    Text: string
    Sentiment: float
    Language: string
    isGeocoded: bool
    GeoScore: float
    GeoConfidence: float
    VAD_V: float
    VAD_A: float
    VAD_D: float
    Big5_O: float
    Big5_C: float
    Big5_E: float
    Big5_A: float
    Big5_N: float
    Datetime: string
    URL: string
    Domain: string
    FactType: string
    reliability: string
    tweetType: string
    toUser: string
    toUser.isBot: float
    mentions: string
    hashtags: string
    media: string
    statuses_count: float
    followers_count: float
    friends_count: float
    favourites_count: float
    listed_count: float
    default_profile: float
    geo_enabled: float
    profile_use_background_image: float
    protected: float
    verified: float
    isBot: float
    fromLocation: string
    fromLocationType: string
    ADM0_A3: string
    ADMIN: string
    geohash: string
    date: string
    TweetID: int64
    """

    USE_COLS = [i.value for i in (
        Columns.DATETIME,
        Columns.FACTTYPE_ORIGINAL,
        Columns.TWEET_TYPE, Columns.USER,
        Columns.VERIFIED,
        Columns.IS_BOT,
        Columns.COUNTRY_CODE,
        Columns.LAT,
        Columns.LONG)]

    def __init__(self, config):
        self.config = config

    def load_from_api(self, store, country_config, curr_from, curr_to, append=True):
        print(f"fetching {curr_from} to {curr_to}, from api")
        args = {
            "date_range": pd.date_range(curr_from, curr_to),
            "usecols": self.USE_COLS
        }
        if country_config.LABEL != "NONE":
            args["iso3"] = country_config.LABEL

        df = rgi.load_split_infodemics_marx(**args)

        usr_hash = {u: hashlib.sha256(u.encode('utf-8')).hexdigest() for u in df['User'].unique()}

        with open(self.config.USERHASH, 'r+b') as f:
            try:
                new_hash = pickle.load(f)
                usr_hash = new_hash | usr_hash
            except EOFError:
                print('First write to', self.config.USERHASH)
                pass
            pickle.dump(usr_hash, f, pickle.HIGHEST_PROTOCOL)

        df['User'] = df['User'].map(usr_hash)
        print(df.head())

        for k, v in df.dtypes.items():
            # HDF can't handle custom data types
            if v == 'string':
                df[k] = df[k].astype(object)

        try:
            store.put(
                country_config.LABEL,
                df.loc[~df['tweetType'].isna()],
                append=append,
                data_columns=True,
                format="t",
                # min_itemsize = {'User': 50}
            )
        except ValueError as e:
            print(f"Something went wrong storing the chunck from {curr_from} to {curr_to}. Skip for now.")
            print(e)
        except BaseException as e:
            print(f"Unexpected {e}, {type(e)=}")

    def load_after(self, store, country_config, load_from, load_end, str_meta, append=True):
        curr_from = load_from
        curr_to = curr_from
        while curr_from <= load_end:
            curr_to += relativedelta(months=+1)
            curr_to = curr_to if curr_to <= load_end else load_end

            self.load_from_api(store, country_config, curr_from, curr_to, append=append)
            curr_from = curr_to
            print("Store contains ", store.keys())

            # store intermediately
            str_meta["to"] = str(curr_to)
            try:
                store.get_node(country_config.LABEL)._v_attrs["daterange"] = str_meta
            except AttributeError as e:
                print(f"Something went wrong storing the node: ", e)

            print("Stored until ", str(curr_to))
            if curr_from == load_end:
                break
        print("COMPLETED LOADING FROM API")

    def load_before(self, store, country_config, load_from, load_end, str_meta):
        curr_from = load_end
        curr_to = curr_from
        while curr_from >= load_from:
            curr_from -= relativedelta(months=+1)
            curr_from = curr_from if curr_from >= load_from else load_from

            self.load_from_api(store, country_config, curr_from, curr_to)

            curr_to = curr_from

            # store intermediately
            str_meta["from"] = str(curr_from)
            store.get_node(country_config.LABEL)._v_attrs["daterange"] = str_meta
            print("Stored from ", str(curr_from))
            if curr_from == load_from:
                break
        print("COMPLETED LOADING FROM API")

    def load_and_prepare(self, country_config):

        start_date = str_to_date(self.config.date_from)
        end_date = str_to_date(self.config.date_to)

        path = self.config.HDF5_STORE
        with pd.HDFStore(path, mode="a") as store:
            if any([country_config.LABEL in x for x in store.keys()]):
                str_meta = store.get_node(country_config.LABEL)._v_attrs["daterange"]
                stored_from = str_to_date(str_meta["from"])
                stored_to = str_to_date(str_meta["to"])
                print(f'{country_config.LABEL} in store from {str_to_date(str_meta["from"])} to '
                      f'{str_to_date(str_meta["to"])}. Store contains {store.keys()}.')
                if stored_from <= start_date and stored_to >= end_date:
                    # fully loaded from API
                    pass
                elif stored_to < start_date:
                    # no overlap, load intercept
                    self.load_after(store, country_config, stored_to, end_date, str_meta)
                elif stored_from > end_date:
                    # no overlap, load intercept
                    self.load_before(store, country_config, start_date, stored_from, str_meta)
                elif start_date < stored_from or end_date >= stored_to:
                    if start_date < stored_from:
                        self.load_before(store, country_config, start_date, stored_from, str_meta)
                    if end_date >= stored_to:
                        self.load_after(store, country_config, stored_to, end_date, str_meta)
            else:
                print(country_config.LABEL, "not in store. Store contains ", store.keys())
                self.load_after(store, country_config, start_date, end_date,
                                {"from": str(start_date), "to": str(end_date)}, append=False)

        with pd.HDFStore(path, mode="r") as store:
            print(f"Getting DF from Store for the dates {start_date} to {end_date}")

            end_date += relativedelta(days=+1)
            where = f"Datetime >= start_date & Datetime < end_date{self.config.where_clause}"
            print("where_clause: ", where)

            c = store.select_as_coordinates(country_config.LABEL, where)
            df = store.select(country_config.LABEL, where=c, columns=[i.value for i in self.config.load_columns])

            print("Completed loading DF of size ", df.shape)
        return df