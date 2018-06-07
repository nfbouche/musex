"""Class and functions to handle cross-matching of catalogs."""

import logging

from astropy import units as u
from astropy.coordinates import search_around_sky
from astropy.table import Table, vstack
import numpy as np

from .catalog import BaseCatalog

NULL_VALUE = {int: -9999, np.int64: -9999, float: np.nan, str: ''}


def _null_value(value):
    return NULL_VALUE[type(value)]


def cross_match(name, db, cat1, cat2, radius=1.):
    """Cross-match two catalogs.

    This function cross-match two catalogs and creates a CrossMatch catalog
    in MuseX database.

    FIXME: Using a name that already exist in the database will lead to trying
    to update the table content and will fail with integrity error. Try to find
    a way to check for existing tables.

    Parameters
    ----------
    name: str
        Name of the CrossMatch catalog in the database.
    db: `dataset.Database`
        The database object to use.
    cat1: `musex.catalog.SpatialCatalog`
        The first catalog to cross-match.
    cat2. `musex.catalog.SpatialCatalog`
        The second catalog.
    radius: float
        The cross-match radius in  arc-seconds.

    """
    logger = logging.getLogger(__name__)

    radius *= u.arcsec

    match_idx1, match_idx2, d2d, _ = search_around_sky(
        cat1.skycoord(), cat2.skycoord(), radius
    )
    nb_match = len(match_idx1)
    logger.info('%d matches.', nb_match)

    # Number of occurrences of each unique index in their respective
    # columns as a dictionary.
    count_idx1 = dict(zip(*np.unique(match_idx1, return_counts=True)))
    count_idx2 = dict(zip(*np.unique(match_idx2, return_counts=True)))
    # Number of matches.
    nb_idx1 = [count_idx1[item] for item in match_idx1]
    nb_idx2 = [count_idx2[item] for item in match_idx2]

    # Sources without counterparts
    all_idx1 = np.array([row[cat1.idname] for row in cat1.select()])
    all_idx2 = np.array([row[cat2.idname] for row in cat2.select()])
    nomatch_idx1 = all_idx1[np.isin(all_idx1, match_idx1, invert=True)]
    nomatch_idx2 = all_idx2[np.isin(all_idx2, match_idx2, invert=True)]
    nb_only1 = len(nomatch_idx1)
    logger.info('%d sources only in %s.', nb_only1, cat1.name)
    nb_only2 = len(nomatch_idx2)
    logger.info('%d sources only in %s.', nb_only2, cat2.name)

    crossmatch_table = vstack(
        [
            Table(  # Source matching
                data=[
                    np.full(nb_match, cat1.name),
                    match_idx1,
                    nb_idx1,
                    np.full(nb_match, cat2.name),
                    match_idx2,
                    nb_idx2,
                    d2d.arcsec,
                ],
                names=[
                    'cat1_name',
                    'idx1',
                    'nb_match_for_idx1',
                    'cat2_name',
                    'idx2',
                    'nb_match_for_idx2',
                    'distance',
                ],
            ),
            Table(  # Only in catalog 1
                data=[
                    np.full(nb_only1, cat1.name),
                    nomatch_idx1,
                    np.full(nb_only1, 0),
                    np.full(nb_only1, ''),
                    np.full(nb_only1, _null_value(all_idx2[0])),
                    np.full(nb_only1, -9999),
                    np.full(nb_only1, np.nan),
                ],
                names=[
                    'cat1_name',
                    'idx1',
                    'nb_match_for_idx1',
                    'cat2_name',
                    'idx2',
                    'nb_match_for_idx2',
                    'distance',
                ],
            ),
            Table(  # Only in catalog 2
                data=[
                    np.full(nb_only2, ''),
                    np.full(nb_only2, _null_value(all_idx1[0])),
                    np.full(nb_only2, -9999),
                    np.full(nb_only2, cat2.name),
                    nomatch_idx2,
                    np.full(nb_only2, 0),
                    np.full(nb_only2, np.nan),
                ],
                names=[
                    'cat1_name',
                    'idx1',
                    'nb_match_for_idx1',
                    'cat2_name',
                    'idx2',
                    'nb_match_for_idx2',
                    'distance',
                ],
            ),
        ]
    )

    # Add identifier to the cross-match table
    crossmatch_table['ID'] = np.arange(len(crossmatch_table)) + 1

    result = CrossMatch(name, db)
    result.insert(crossmatch_table)

    return result


class CrossMatch(BaseCatalog):
    """Cross-match of two catalogs.

    This class handles the cross-match of two catalogs. The match is done
    keeping all the possible associations given the provided radius.

    The table is made of these columns:

    - cat1_name: the name of the first catalog
    - idx1: the identifier of the source from the first catalog
    - nb_match_for_idx1: the number of matches associated to the catalog 1
      source.
    - cat2_name: the name of the second catalog
    - idx2: the identifier of the source from the second catalog
    - nb_match_for_idx2: the number of matches associated to the catalog 2
      source.
    - d2d: the separation between the two sources in arc-second

    The table gathers both the matching sources and the sources that are only
    in one of the catalogs. In the case of sources that are only in one
    catalog, the identifier in the second catalog is set to a null value
    depending of the type of the identifier and the distance is set to NaN.
    """

    def __init__(self, name, db):
        """Create a new cross match catalog.

        Parameters
        ----------
        name: str
            Name of this cross-match catalog.
        db: `dataset.Database`
            The database object to use.

        """
        super().__init__(name=name, db=db, idname='ID')
        self.update_meta(type='cross-match')
