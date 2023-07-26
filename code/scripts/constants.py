import seaborn as sns
from scripts.enums import FactTypes, ContentType

FACTTYPE_NAME_MAP = {i.value.name: i for i in FactTypes}
FACTTYPE_MAP = {i.value.type_reduced: i for i in FactTypes}


CATEGORIES = ['SCIENCE', 'MSM', 'SATIRE', 'CLICKBAIT', 'OTHER', 'SHADOW', 'POLITICAL', 'FAKE/HOAX',
              'CONSPIRACY/JUNKSCI']
HARM_SCORES = {c: i + 1 for i, c in enumerate(CATEGORIES)}
RELIABLE = {c: "reliable" if i < 2 else "unknown" if 4 <= i <= 5 else "unreliable" for i, c in enumerate(CATEGORIES)}

sortby_harmscore = lambda t: HARM_SCORES.get(t[0], -1)

facttype_order_full = ['SCIENCE', 'MSM', 'SATIRE', 'CLICKBAIT', 'OTHER', 'SHADOW', 'POLITICAL', 'FAKE/HOAX',
                       'CONSPIRACY/JUNKSCI']
facttype_order_other = ['SCIENCE', 'MSM', 'SATIRE', 'CLICKBAIT', 'OTHER', 'POLITICAL', 'FAKE/HOAX',
                        'CONSPIRACY/JUNKSCI']
facttype_order = ['SCIENCE', 'MSM', 'SATIRE', 'CLICKBAIT', 'POLITICAL', 'FAKE/HOAX', 'CONSPIRACY/JUNKSCI']
reliable_facttypes = ['SCIENCE', 'MSM']
unreliable_facttypes = ['SATIRE', 'CLICKBAIT', 'POLITICAL', 'FAKE/HOAX', 'CONSPIRACY/JUNKSCI']
harmful_facttypes = ['POLITICAL', 'FAKE/HOAX', 'CONSPIRACY/JUNKSCI']
reliable_palette = {cluster: color for cluster, color in
                    zip(["reliable", "unreliable", "unknown"], sns.color_palette("pastel", 3))}
f_col = "FactType2"
h_col = "harmful"

facttype_to_proper_map = {i: i.title() for i in ['SCIENCE', 'SATIRE', 'CLICKBAIT', 'OTHER', 'POLITICAL']}
facttype_to_proper_map['MSM'] = 'Mainstream media'
facttype_to_proper_map['FAKE/HOAX'] = 'Fake or hoax'
facttype_to_proper_map['CONSPIRACY/JUNKSCI'] = 'Conspiracy or junk science'

facttype_palette = dict()
facttype_palette['SCIENCE'], facttype_palette['MSM'] = list(sns.color_palette("Blues_r", 2))
facttype_palette['SATIRE'], facttype_palette['CLICKBAIT'], facttype_palette['POLITICAL'], facttype_palette['FAKE/HOAX'], \
facttype_palette['CONSPIRACY/JUNKSCI'] = tuple(sns.color_palette("Reds", 5))

facttype_palette_full = facttype_palette.copy()
facttype_palette_full['OTHER'] = sns.color_palette("Greys", 1)[0]

facttype_palette_grey = facttype_palette.copy()
facttype_palette_grey['OTHER'], facttype_palette_grey['SHADOW'], facttype_palette_grey['MISSING'] = tuple(
    sns.color_palette("Greys", 3))

tweettype_palette = {tweetType: color for tweetType, color in zip("T RT RE".split(), sns.color_palette("pastel", 3))}
tweettype_palette["total"] = "red"
