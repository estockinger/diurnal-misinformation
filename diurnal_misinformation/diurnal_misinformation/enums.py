from collections.abc import Iterable
from enum import Enum
from dataclasses import dataclass

import seaborn as sns


class Columns(Enum):
    POSTS = 'posts'
    RATIO = 'ratio'
    YEAR = 'year'
    MONTH = 'month'
    MIN_BINS15 = 'min_bins15'
    LOCKDOWN = "in_lockdown"
    USERHASH = 'userhash'
    CLUSTER = 'cluster'
    VERIFIED = 'verified'
    FACTTYPE = 'FactType'
    LAT = "lat"
    LONG = "long"


@dataclass
class Cluster:
    name: str
    color: str

@dataclass(order=True, init=False)
class Clusters(Enum):
    INFREQUENT = Cluster('infrequent type', "#75dcff")
    MORNING = Cluster('morning type', "#aa75e5")
    INTERMEDIATE = Cluster('intermediate type',  "#99cc33")
    EVENING = Cluster('evening type', "#ff9f9b")
    TOTAL = Cluster('total', 'red')


    @classmethod
    def order(cls, attribute='name'):
        return [getattr(c, attribute) for c in Clusters if c.value != cls.TOTAL.value]
    

    @classmethod
    def total_order(cls, attribute='name'):
        return [getattr(c, attribute) for c in Clusters]


    @classmethod
    def palette(cls, attribute = 'name'):
        return {getattr(ft, attribute): ft.value.color for ft in Clusters}


class ContentType(Enum):
    CLEAR = "likely not disinformative"
    DISINFORMATIVE = "potentially disinformative"
    UNKNOWN = "unknown"
    KNOWN = "known"

    @classmethod
    def palette(cls):
        return {c: color for c, color
                in zip(list(cls), sns.color_palette("pastel", len(list(cls))))}


@dataclass
class FactType:
    name: str
    harm_score: int
    content_type: ContentType
    color: Iterable


@dataclass(order=True, init=False)
class FactTypes(Enum):
    SCIENCE = FactType("Science", 1, ContentType.CLEAR, (0.21568627450980393, 0.5294117647058824, 0.7542483660130719))
    MSM = FactType("Mainstream Media", 2, ContentType.CLEAR, (0.6718954248366014, 0.8143790849673203, 0.9006535947712418))
    SATIRE = FactType("Satire", 3, ContentType.CLEAR, (0.9935870818915802, 0.8323414071510957, 0.7624913494809689))
    CLICKBAIT = FactType("Clickbait", 4, ContentType.CLEAR, (0.9882352941176471, 0.6261437908496732, 0.5084967320261438))
    OTHER = FactType("Other", 5, ContentType.UNKNOWN, (.8, .8, .8))
    POLITICAL = FactType("Political", 7, ContentType.DISINFORMATIVE, (0.9835755478662053, 0.4127950788158401, 0.28835063437139563))
    FAKE = FactType("Fake or hoax", 8, ContentType.DISINFORMATIVE, (0.6943944636678201, 0.07003460207612457, 0.09231833910034601))
    CONSPIRACY = FactType("Conspiracy & junk science", 9, ContentType.DISINFORMATIVE, (0.8901960784313725, 0.18562091503267975, 0.15294117647058825))

    @classmethod
    def order(cls, attribute='name'):
        return [getattr(c, attribute) for c in FactTypes]

    @classmethod
    def known_order(cls, attribute='name'):
        return [getattr(c, attribute) for c in FactTypes if c.value.content_type != ContentType.UNKNOWN]

    @classmethod
    def reliable_order(cls, attribute='name'):
        return [getattr(cls.SCIENCE, attribute), getattr(cls.MSM, attribute)]

    @classmethod
    def disinformative_order(cls, attribute='name'):
        return [getattr(c, attribute) for c in (cls.POLITICAL, cls.FAKE, cls.CONSPIRACY)]
    
    @classmethod
    def palette(cls, attribute='name'):
        return {getattr(ft, attribute): ft.value.color for ft in FactTypes}


def getattr(x, attribute=None):
    if attribute is None:
        return x
    return x.value.__getattribute__(attribute)
