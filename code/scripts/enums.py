from collections.abc import Iterable
from enum import Enum, EnumMeta
from dataclasses import dataclass

import seaborn as sns


class Columns(Enum):
    TWEET_ID = 'TweetID'
    COUNTRY_CODE = 'ADM0_A3'
    LAT = "lat"
    LONG = "long"
    DATETIME = 'Datetime'
    LOCAL_TIME = 'local_time'
    YEAR = 'year'
    MONTH = 'month'
    WEEK = 'week'
    DAY = 'day'
    HOUR = 'hour'
    MINUTES = 'minutes'
    IS_WEEKEND = 'is_weekend'

    IS_BOT = 'isBot'
    VERIFIED = 'verified'
    FOLLOWERS_COUNT = 'followers_count'
    TWEET_TYPE = 'tweetType'
    RELIABILITY = 'Reliability'
    FACTTYPE_ORIGINAL = 'FactType'
    FACTTYPE = 'FactType2'
    HARMSCORE = 'HarmScore'
    MACHINATED = 'harmful'
    USER = 'User'
    RATIO_BY_TWEET = 'ratio'
    RATIO_BY_USER = 'ratio_norm'

    NUM_POSTS = 'num_posts'
    NUM_POSTS_BIN = 'num_posts_bin'
    NUM_POSTS_WEIGHTED = 'num_posts_weighted'
    WEIGHT = 'weight'

    MIN_BINS15 = 'min_bins15'
    DATE_BIN = 'date_bin'
    SUN_SO_FAR_M = 'sun_so_far_m'
    SUN_LEFT_M = 'sun_left_m'
    TOTAL_SUN_M = 'total_sun_m'

    CLUSTER = 'cluster'
    ACTIVITY = 'activity'
    ACTIVITY_WEIGHTED = 'activity_weighted'

    LOCKDOWN = "in_lockdown"
    EMERGENCY = "emergency"


class ClusterEnumMeta(EnumMeta):
    def __iter__(self):
        for x in super().__iter__():
            if x.value != 'bot':
                yield x.value


class Clusters(Enum, metaclass=ClusterEnumMeta):
    INFREQUENT = 'infrequent type'
    MORNING = 'morning type'
    INTERMEDIATE = 'intermediate type'
    EVENING = 'evening type'
    BOT = 'bot'

    @classmethod
    def palette(cls, include_total=True):
        palette = {cluster: color for cluster, color in zip(list(cls), sns.color_palette("pastel", len(list(cls))))}
        if include_total:
            palette['total'] = 'red'
        return palette


class ContentType(Enum):
    CLEAR = "non-controversial"
    MANIPULATED = "potentially machinated"
    UNKNOWN = "unknown"

    @classmethod
    def palette(cls):
        return {c: color for c, color
                in zip(list(cls), sns.color_palette("pastel", len(list(cls))))}


@dataclass
class FactType:
    name: str
    harm_score: int
    content_type: ContentType
    type_reduced: str
    color: Iterable


class FactTypes(Enum):
    SCIENCE = FactType("Science", 1, ContentType.CLEAR, 'SCIENCE',
                       (0.21568627450980393, 0.5294117647058824, 0.7542483660130719))
    MSM = FactType("Mainstream Media", 2, ContentType.CLEAR, 'MSM',
                   (0.6718954248366014, 0.8143790849673203, 0.9006535947712418))
    SATIRE = FactType("Satire", 3, ContentType.CLEAR, 'SATIRE',
                      (0.9935870818915802, 0.8323414071510957, 0.7624913494809689))
    CLICKBAIT = FactType("Clickbait", 4, ContentType.CLEAR, 'CLICKBAIT',
                         (0.9882352941176471, 0.6261437908496732, 0.5084967320261438))
    OTHER = FactType("Other", 5, ContentType.UNKNOWN, 'OTHER', (.8, .8, .8))
    POLITICAL = FactType("Political", 7, ContentType.MANIPULATED, 'POLITICAL',
                         (0.8901960784313725, 0.18562091503267975, 0.15294117647058825))
    FAKE = FactType("Fake or hoax", 8, ContentType.MANIPULATED, 'FAKE/HOAX',
                    (0.6943944636678201, 0.07003460207612457, 0.09231833910034601))
    CONSPIRACY = FactType("Conspiracy & junk science", 9, ContentType.MANIPULATED, 'CONSPIRACY/JUNKSCI',
                          (0.9835755478662053, 0.4127950788158401, 0.28835063437139563))

    @classmethod
    def order(cls, attribute=None):
        return [getattr(c, attribute) for c in FactTypes]

    @classmethod
    def known_order(cls, attribute=None):
        return [getattr(c, attribute) for c in FactTypes if c.value.content_type != ContentType.UNKNOWN]

    @classmethod
    def reliable_order(cls, attribute=None):
        return [getattr(cls.SCIENCE, attribute), getattr(cls.MSM, attribute)]

    @classmethod
    def harmful_order(cls, attribute=None):
        return [getattr(c, attribute) for c in (cls.POLITICAL, cls.FAKE, cls.CONSPIRACY)]

    @classmethod
    def palette(cls, attribute='name'):
        return {getattr(ft, attribute): ft.value.color for ft in FactTypes}


def getattr(x, attribute=None):
    if attribute is None:
        return x
    return x.value.__getattribute__(attribute)
