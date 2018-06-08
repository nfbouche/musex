"""Class and functions to handle cross-matching of catalogs."""

import logging

from astropy import units as u
from astropy.coordinates import search_around_sky
from astropy.table import Table, vstack
from astropy.visualization import hist
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
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

    # Identifiers of matching sources
    match_ids_1 = cat1.select().as_table()[cat1.idname][match_idx1]
    match_ids_2 = cat2.select().as_table()[cat2.idname][match_idx2]

    # Number of occurrences of each unique index in their respective
    # columns as a dictionary.
    count_idx1 = dict(zip(*np.unique(match_idx1, return_counts=True)))
    count_idx2 = dict(zip(*np.unique(match_idx2, return_counts=True)))
    # Number of matches.
    nb_idx1 = [count_idx1[item] for item in match_idx1]
    nb_idx2 = [count_idx2[item] for item in match_idx2]

    # Sources without counterparts
    all_ids_1 = np.array([row[cat1.idname] for row in cat1.select()])
    all_ids_2 = np.array([row[cat2.idname] for row in cat2.select()])
    nomatch_ids_1 = all_ids_1[np.isin(all_ids_1, match_ids_1, invert=True)]
    nomatch_ids_2 = all_ids_2[np.isin(all_ids_2, match_ids_2, invert=True)]
    nb_only1 = len(nomatch_ids_1)
    logger.info('%d sources only in %s.', nb_only1, cat1.name)
    nb_only2 = len(nomatch_ids_2)
    logger.info('%d sources only in %s.', nb_only2, cat2.name)

    crossmatch_table = vstack(
        [
            Table(  # Source matching
                data=[
                    match_ids_1,
                    nb_idx1,
                    match_ids_2,
                    nb_idx2,
                    d2d.arcsec,
                ],
                names=[
                    f'{cat1.name}_id',
                    f'{cat1.name}_nbmatch',
                    f'{cat2.name}_id',
                    f'{cat2.name}_nbmatch',
                    'distance',
                ],
            ),
            Table(  # Only in catalog 1
                data=[
                    nomatch_ids_1,
                    np.full(nb_only1, 0),
                    np.full(nb_only1, _null_value(all_ids_2[0])),
                    np.full(nb_only1, -9999),
                    np.full(nb_only1, np.nan),
                ],
                names=[
                    f'{cat1.name}_id',
                    f'{cat1.name}_nbmatch',
                    f'{cat2.name}_id',
                    f'{cat2.name}_nbmatch',
                    'distance',
                ],
            ),
            Table(  # Only in catalog 2
                data=[
                    np.full(nb_only2, _null_value(all_ids_1[0])),
                    np.full(nb_only2, -9999),
                    nomatch_ids_2,
                    np.full(nb_only2, 0),
                    np.full(nb_only2, np.nan),
                ],
                names=[
                    f'{cat1.name}_id',
                    f'{cat1.name}_nbmatch',
                    f'{cat2.name}_id',
                    f'{cat2.name}_nbmatch',
                    'distance',
                ],
            ),
        ]
    )

    # Add identifier to the cross-match table
    crossmatch_table['ID'] = np.arange(len(crossmatch_table)) + 1

    result = CrossMatch(name, db)
    result.insert(crossmatch_table)
    result.cat1, result.cat2 = cat1, cat2
    result.update_meta(cat1_name=cat1.name, cat2_name=cat2.name)

    return result


class CrossMatch(BaseCatalog):
    """Cross-match of two catalogs.

    This class handles the cross-match of two catalogs. The match is done
    keeping all the possible associations given the provided radius.

    The table is made of these columns:

    - <cat1.name>_id: the identifier of the source from the first catalog
    - <cat1.name>_nbmatch: the number of matches associated to the catalog 1
      source.
    - <cat2.name>_id: the identifier of the source from the second catalog
    - <cat2.name>_nbmatch: the number of matches associated to the catalog 2
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

    def nb_diagnostic(self):
        """Output some diagnostics in a notebook."""
        from IPython.core.display import HTML, display

        # Temporarily disable interactive plotting not to display plots twice.
        plt.ioff()

        table = self.select().as_table()
        with_matches = table[~np.isnan(table['distance'])]

        # Histogram of distances
        fig, ax = plt.subplots()
        hist(with_matches['distance'], ax=ax)
        ax.set_xlabel("Distance in arc-seconds")
        display(HTML("<h3>Histogram of separation for matching sources</h3>"),
                fig)

        def _catalog_diag(cat):
            """Diagnostic of a catalog"""
            rows_in_cat = table[table[f'{cat.name}_nbmatch'] >= 0]
            nb_sources_in_cat = len(np.unique(rows_in_cat[f'{cat.name}_id']))

            rows_with_match = rows_in_cat[
                rows_in_cat[f'{cat.name}_nbmatch'] > 0]
            sources_with_match, nb_match = np.unique(
                rows_with_match[f'{cat.name}_id'], return_counts=True)
            nb_sources_with_match = len(sources_with_match)
            percentage_with_match = (100 * nb_sources_with_match /
                                     nb_sources_in_cat)

            # Bar plot of the number of sources with a given number of matches.
            fig, ax = plt.subplots()
            ax.bar(*np.unique(nb_match, return_counts=True))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Number of matches")
            ax.set_ylabel("Number of sources")

            display(
                HTML(f"""<h3>Catalog {cat.name}</h3>
                    <ul>
                        <li>{nb_sources_in_cat} sources</li>
                        <li>{nb_sources_with_match} sources with at least
                        one match ({percentage_with_match:.2f}%)</li>
                    </ul>"""),
                fig)

        _catalog_diag(self.cat1)
        _catalog_diag(self.cat2)

        # Back to interactive.
        plt.ion()
