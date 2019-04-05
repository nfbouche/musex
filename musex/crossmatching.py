"""Class and functions to handle cross-matching of catalogs."""

import logging

from astropy import units as u
from astropy.coordinates import search_around_sky
from astropy.table import Table, join, vstack
from astropy.visualization import hist
from matplotlib import pyplot as plt
import numpy as np

from .catalog import BaseCatalog

NULL_VALUE = {int: -9999, np.int64: -9999, float: np.nan, str: ''}

__all__ = ('CrossMatch', 'gen_crossmatch')


def _null_value(value):
    return NULL_VALUE[type(value)]


def gen_crossmatch(name, db, cat1, cat2, radius=1.):
    """Cross-match two catalogs.

    This function cross-match two catalogs and creates a CrossMatch catalog
    in MuseX database. Use the MuseX.cross_match function.

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

    Returns
    -------
    `musex.crossmatching.CrossMatch`

    """
    logger = logging.getLogger(__name__)

    radius *= u.arcsec

    match_idx1, match_idx2, d2d, _ = search_around_sky(
        cat1.skycoord(), cat2.skycoord(), radius
    )
    nb_match = len(match_idx1)
    logger.info('%d matches.', nb_match)

    # Catalogues identifiers
    all_ids_1 = np.array([row[cat1.idname] for row in cat1.select()])
    all_ids_2 = np.array([row[cat2.idname] for row in cat2.select()])

    # Identifiers of matching sources
    match_ids_1 = all_ids_1[match_idx1]
    match_ids_2 = all_ids_2[match_idx2]

    # Number of occurrences of each unique index in their respective
    # columns as a dictionary.
    count_idx1 = dict(zip(*np.unique(match_idx1, return_counts=True)))
    count_idx2 = dict(zip(*np.unique(match_idx2, return_counts=True)))
    # Number of matches.
    nb_idx1 = [count_idx1[item] for item in match_idx1]
    nb_idx2 = [count_idx2[item] for item in match_idx2]

    # Sources without counterparts
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

    def matched_table_with_more_than(self, min_matches, *, min_in_cat=None):
        """Return the full table of matches.

        This function returns a table with columns from both catalogs with all
        the matches and limited to sources having more than a minimum number of
        matches. If `min_in_cat` (1 or 2) is provided, the minimum match
        selection will be done only in this catalog.

        Parameters
        ----------
        min_matches: int
            Only sources having this number of matches or more will be
            returned.
        min_in_cat: int
            If not None, should be 1 or 2 and the minimum number of matches
            will be looked for only in the given catalog.
        """
        if min_in_cat == 1:
            selection = self.select(
                self.c[f'{self.cat1.name}_nbmatch'] > min_matches)
        elif min_in_cat == 2:
            selection = self.select(
                self.c[f'{self.cat2.name}_nbmatch'] > min_matches)
        else:
            selection = self.select(
                (self.c[f'{self.cat1.name}_nbmatch'] > min_matches) |
                (self.c[f'{self.cat2.name}_nbmatch'] > min_matches))

        if len(selection) == 0:
            self.logger.info("There are no sources with this number of "
                             "matches.")
            return None

        selection = selection.as_table()

        cat1 = self.cat1.select_ids(
            np.unique(selection[f'{self.cat1.name}_id'])).as_table()
        cat1.rename_column(self.cat1.idname, f'{self.cat1.name}_id')
        cat2 = self.cat2.select_ids(
            np.unique(selection[f'{self.cat2.name}_id'])).as_table()
        cat2.rename_column(self.cat2.idname, f'{self.cat2.name}_id')

        # clear meta (idname, raname, etc.) to avoid conflicts warnings and
        # weird state on the result catalog
        cat1.meta.clear()
        cat2.meta.clear()
        selection.meta.clear()

        result = join(cat1, selection, keys=f'{self.cat1.name}_id',
                      join_type='outer')
        result = join(result, cat2, keys=f'{self.cat2.name}_id',
                      join_type='outer')

        return result

    def fig_diagnostic(self):
        """Diagnostic figure for the cross-match."""
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))

        table = self.select().as_table()
        with_matches = table[~np.isnan(table['distance'])]

        # Histogram of distances
        hist(with_matches['distance'], ax=ax1)
        ax1.set_xlabel("Distance in arc-seconds")
        ax1.set_ylabel("Counts")

        def _catalog_diag(cat):
            """Return information on a catalog.

            - total number of sources in the catalog
            - number of sources with at least one match
            - two arrays: number of matches, number of sources with this number
              of matches
            """
            rows_in_cat = table[table[f'{cat.name}_nbmatch'] >= 0]
            tot_sources = len(np.unique(rows_in_cat[f'{cat.name}_id']))

            rows_with_match = rows_in_cat[
                rows_in_cat[f'{cat.name}_nbmatch'] > 0]
            sources_with_match, nb_match = np.unique(
                rows_with_match[f'{cat.name}_id'], return_counts=True)
            with_match = len(sources_with_match)

            matches, nb_sources = np.unique(nb_match, return_counts=True)

            return tot_sources, with_match, matches, nb_sources

        tot_sources_1, with_match_1, matches_1, nb_sources_1 = _catalog_diag(
            self.cat1)
        tot_sources_2, with_match_2, matches_2, nb_sources_2 = _catalog_diag(
            self.cat2)

        self.logger.info(
            "Catalog %s has %d sources, %d with at least one match (%.1f%%).",
            self.cat1.name, tot_sources_1, with_match_1,
            100 * with_match_1 / tot_sources_1)

        self.logger.info(
            "Catalog %s has %d sources, %d with at least one match (%.1f%%).",
            self.cat2.name, tot_sources_2, with_match_2,
            200 * with_match_2 / tot_sources_2)

        # Bar charts of number of matches
        width = 0.3  # Width of the bars
        ax2.bar(matches_1-width/2, nb_sources_1, width=width, color='r',
                label=self.cat1.name)
        ax2.bar(matches_2+width/2, nb_sources_2, width=width, color='b',
                label=self.cat2.name)
        ax2.set_xlabel("Number of matches")
        ax2.set_ylabel("Number of sources")
        ax2.legend(loc=0)

        return fig
