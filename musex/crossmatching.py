"""Class and functions to handle cross-matching of catalogs."""

import logging

import numpy as np
from numpy import ma
import musex

from astropy import units as u
from astropy.coordinates import search_around_sky
from astropy.table import Table, join, vstack
from astropy.visualization import hist

from .catalog import BaseCatalog

NULL_VALUE = {int: -9999, np.int64: -9999, float: np.nan, str: ""}

__all__ = ("CrossMatch", "gen_crossmatch")


def _null_value(value):
    return NULL_VALUE[type(value)]


def gen_crossmatch(name, db, cat1, cat2, radius=1.0, no_dbcat=False, col1_names=None, col2_names=None, show_progress=True):
    """Cross-match two catalogs.

    This function cross-match two catalogs and creates a CrossMatch catalog
    in MuseX database. Use the MuseX.cross_match function.

    Parameters
    ----------
    name : str
        Name of the CrossMatch catalog in the database.
    db : `dataset.Database`
        The database object.
    cat1 : `musex.Catalog`
        The first catalog to cross-match.
    cat2 : `musex.Catalog`
        The second catalog.
    radius : float
        The cross-match radius in  arc-seconds.

    Returns
    -------
    `musex.crossmatching.CrossMatch`

    """
    logger = logging.getLogger(__name__)

    radius *= u.arcsec
    
    if type(cat1) is musex.catalog.ResultSet:
        res1 = cat1
        idname1 = cat1.catalog.idname if col1_names is None else col1_names[0]
        cname1 = cat1.catalog.name
        sky1 = cat1.skycoord(skycols=None if col1_names is None else (col1_names[1],col1_names[2]))
    else:
        res1 = cat1.select()
        idname1 = cat1.idname if col1_names is None else col1_names[0]
        cname1 = cat1.name
        sky1 = cat1.skycoord(idname=idname1, skycols=None if col1_names is None else (col1_names[1],col1_names[2]))
    if type(cat2) is musex.catalog.ResultSet:
        res2 = cat2
        idname2 = cat2.catalog.idname if col2_names is None else col2_names[0]
        cname2 = cat2.catalog.name
        sky2 = cat2.skycoord(skycols=None if col2_names is None else (col2_names[1],col2_names[2]))
    else:
        res2 = cat2.select()
        idname2 = cat2.idname if col2_names is None else col2_names[0]
        cname2 = cat2.name
        sky2 = cat2.skycoord(idname=idname2, skycols=None if col2_names is None else (col2_names[1],col2_names[2]))

    match_idx1, match_idx2, d2d, _ = search_around_sky(
        sky1, sky2, radius
    )
    nb_match = len(match_idx1)
    logger.info("%d matches.", nb_match)

    # Catalogues identifiers
 
    all_ids_1 = np.array([row[idname1] for row in res1])
    all_ids_2 = np.array([row[idname2] for row in res2])

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
    logger.info("%d sources only in %s.", nb_only1, cname1)
    nb_only2 = len(nomatch_ids_2)
    logger.info("%d sources only in %s.", nb_only2, cname2)

    crossmatch_table = vstack(
        [
            Table(  # Source matching
                data=[match_ids_1, nb_idx1, match_ids_2, nb_idx2, d2d.arcsec],
                names=[
                    f"{cname1}_id",
                    f"{cname1}_nbmatch",
                    f"{cname2}_id",
                    f"{cname2}_nbmatch",
                    "distance",
                ],
            ),
            Table(  # Only in catalog 1
                data=[
                    nomatch_ids_1,
                    np.full(nb_only1, 0),
                    ma.masked_equal(
                        np.full(nb_only1, _null_value(all_ids_2[0])),
                        _null_value(all_ids_2[0]),
                    ),
                    ma.masked_equal(np.full(nb_only1, -9999), -9999),
                    np.full(nb_only1, np.nan),
                ],
                names=[
                    f"{cname1}_id",
                    f"{cname1}_nbmatch",
                    f"{cname2}_id",
                    f"{cname2}_nbmatch",
                    "distance",
                ],
            ),
            Table(  # Only in catalog 2
                data=[
                    ma.masked_equal(
                        np.full(nb_only2, _null_value(all_ids_1[0])),
                        _null_value(all_ids_1[0]),
                    ),
                    ma.masked_equal(np.full(nb_only2, -9999), -9999),
                    nomatch_ids_2,
                    np.full(nb_only2, 0),
                    np.full(nb_only2, np.nan),
                ],
                names=[
                    f"{cname1}_id",
                    f"{cname1}_nbmatch",
                    f"{cname2}_id",
                    f"{cname2}_nbmatch",
                    "distance",
                ],
            ),
        ]
    )

    # Add identifier to the cross-match table
    crossmatch_table["id_"] = np.arange(len(crossmatch_table)) + 1
    if no_dbcat:
        logger.info('Returning only astropy table, no db catalog creation')       
        return crossmatch_table

    result = CrossMatch(name, db, idname='id_', primary_id='id_')
    result.insert(crossmatch_table, show_progress=show_progress)
    result.cat1, result.cat2 = cat1, cat2
    result.update_meta(cat1_name=cname1, cat2_name=cname2)

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

    catalog_type = "cross-match"

    def matched_table_with_more_than(self, min_matches, *, min_in_cat=None):
        """Return the full table of matches.

        This function returns a table with columns from both catalogs with all
        the matches and limited to sources having more than a minimum number of
        matches. If `min_in_cat` (1 or 2) is provided, the minimum match
        selection will be done only in this catalog.

        Parameters
        ----------
        min_matches : int
            Only sources having this number of matches or more will be
            returned.
        min_in_cat : int
            If not None, should be 1 or 2 and the minimum number of matches
            will be looked for only in the given catalog.
        """
        if type(self.cat1) is musex.catalog.ResultSet:
            res1 = self.cat1.as_table()
            idname1 = self.cat1.catalog.idname 
            cname1 = self.cat1.catalog.name
        else:
            res1 = self.cat1.select().as_table()
            idname1 = self.cat1.idname 
            cname1 = self.cat1.name
        nbcol1 = f"{cname1}_nbmatch"
        idcol1 = f"{cname1}_id"
        
        if type(self.cat2) is musex.catalog.ResultSet:
            res2 = self.cat2.as_table()
            idname2 = self.cat2.catalog.idname 
            cname2 = self.cat2.catalog.name
        else:
            res2 = self.cat2.select().as_table()
            idname2 = self.cat2.idname 
            cname2 = self.cat2.name
        nbcol2 = f"{cname2}_nbmatch"
        idcol2 = f"{cname2}_id"
            
        if min_in_cat == 1:
            selection = self.select(self.c[nbcol1] > min_matches)
        elif min_in_cat == 2:
            selection = self.select(self.c[nbcol2] > min_matches)
        else:
            selection = self.select(
                (self.c[nbcol1] > min_matches)
                | (self.c[nbcol2] > min_matches)
            )

        if len(selection) == 0:
            self.logger.info("There are no sources with this number of matches.")
            return None

        selection = selection.as_table()

        cat1 = res1[np.in1d(res1[idname1], selection[idcol1])]
        cat1.rename_column(idname1, idcol1)
        
        cat2 = res2[np.in1d(res2[idname2], selection[idcol2])]
        cat2.rename_column(idname2, idcol2)

        # clear meta (idname, raname, etc.) to avoid conflicts warnings and
        # weird state on the result catalog
        cat1.meta.clear()
        cat2.meta.clear()
        selection.meta.clear()

        result = join(cat1, selection, keys=idcol1, join_type="outer")
        result = join(result, cat2, keys=idcol2, join_type="outer")

        return result

    def info(self):
        """ print catalog info """
        pass

    def fig_diagnostic(self):  # pragma: no cover
        """Diagnostic figure for the cross-match."""
        from matplotlib import pyplot as plt

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))

        table = self.select().as_table()
        with_matches = table[~np.isnan(table["distance"])]

        # Histogram of distances
        hist(with_matches["distance"], ax=ax1)
        ax1.set_xlabel("Distance in arc-seconds")
        ax1.set_ylabel("Counts")

        def _catalog_diag(cat):
            """Return information on a catalog.

            - total number of sources in the catalog
            - number of sources with at least one match
            - two arrays: number of matches, number of sources with this number
              of matches
            """
            rows_in_cat = table[table[f"{cat.name}_nbmatch"] >= 0]
            tot_sources = len(np.unique(rows_in_cat[f"{cat.name}_id"]))

            rows_with_match = rows_in_cat[rows_in_cat[f"{cat.name}_nbmatch"] > 0]
            sources_with_match, nb_match = np.unique(
                rows_with_match[f"{cat.name}_id"], return_counts=True
            )
            with_match = len(sources_with_match)

            matches, nb_sources = np.unique(nb_match, return_counts=True)

            return tot_sources, with_match, matches, nb_sources

        tot_sources_1, with_match_1, matches_1, nb_sources_1 = _catalog_diag(self.cat1)
        tot_sources_2, with_match_2, matches_2, nb_sources_2 = _catalog_diag(self.cat2)

        self.logger.info(
            "Catalog %s has %d sources, %d with at least one match (%.1f%%).",
            self.cat1.name,
            tot_sources_1,
            with_match_1,
            100 * with_match_1 / tot_sources_1,
        )

        self.logger.info(
            "Catalog %s has %d sources, %d with at least one match (%.1f%%).",
            self.cat2.name,
            tot_sources_2,
            with_match_2,
            200 * with_match_2 / tot_sources_2,
        )

        # Bar charts of number of matches
        width = 0.3  # Width of the bars
        ax2.bar(
            matches_1 - width / 2,
            nb_sources_1,
            width=width,
            color="r",
            label=self.cat1.name,
        )
        ax2.bar(
            matches_2 + width / 2,
            nb_sources_2,
            width=width,
            color="b",
            label=self.cat2.name,
        )
        ax2.set_xlabel("Number of matches")
        ax2.set_ylabel("Number of sources")
        ax2.legend(loc=0)

        return fig
