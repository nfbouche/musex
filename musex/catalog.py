import json
import logging
import os
import re
import warnings
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from numpy import ma
from sqlalchemy import sql

from astropy.table import Table, vstack
from astropy.utils.decorators import lazyproperty
from mpdaf.sdetect import Catalog as _Catalog
from mpdaf.tools import isiter, progressbar

from .utils import table_to_odict

DIRNAME = os.path.abspath(os.path.dirname(__file__))

__all__ = (
    'ResultSet',
    'Catalog',
    'BaseCatalog',
    'InputCatalog',
    'MarzCatalog',
    'IdMapping',
)

FILL_VALUES = {int: -9999, float: np.nan, str: ''}

RESERVED_COLNAMES = ['active', 'merged_in', 'merged']


def get_cat_name(res_or_cat):
    """Helper function to get the catalog name from different objects."""
    if isinstance(res_or_cat, str):
        return res_or_cat
    elif isinstance(res_or_cat, BaseCatalog):
        return res_or_cat.name
    elif isinstance(res_or_cat, (Table, ResultSet)):
        return res_or_cat.catalog.name
    else:
        raise ValueError('cat must be a Catalog instance or name')


def get_result_table(res_or_cat, filter_active=False):
    """Helper function to get an Astropy Table from different objects."""
    if isinstance(res_or_cat, ResultSet):
        tbl = res_or_cat.as_table()
    elif isinstance(res_or_cat, Catalog):
        tbl = res_or_cat.select().as_table()
    elif isinstance(res_or_cat, Table):
        tbl = res_or_cat
    else:
        raise ValueError('invalid input for res_or_cat')

    if filter_active and 'active' in tbl.colnames:
        tbl = tbl[tbl['active']]

    return tbl


class ResultSet(Sequence):
    """Contains the result of a query on the database."""

    def __init__(self, results, whereclause=None, catalog=None, columns=None):
        self.results = list(results)
        # TODO: use weakref here ?
        self.catalog = catalog
        self.columns = columns

        self.whereclause = whereclause
        if whereclause is not None and not isinstance(whereclause, str):
            self.whereclause = str(
                whereclause.compile(compile_kwargs={"literal_binds": True})
            )

    def __repr__(self):
        wherecl = self.whereclause or ''
        return f'<{self.__class__.__name__}({wherecl})>, {len(self)} results'

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]

    def as_table(self, mpdaf_catalog=True):
        """Return a table with the query results.

        By default, it returns a `mpdaf.sdetect.Catalog` object (if
        ``mpdaf_catalog`` is True), otherwise a `astropy.table.Table` object.

        """
        cls = _Catalog if mpdaf_catalog else Table
        tbl = cls(masked=True)
        tbl.whereclause = self.whereclause
        tbl.catalog = self.catalog
        for col, val in zip(self.columns, zip(*[r.values() for r in self.results])):
            dtype = col.type.python_type
            fill_value = FILL_VALUES.get(dtype)
            mask = [v is None for v in val]
            val = [fill_value if v is None else v for v in val]
            tbl[col.name] = ma.array(val, mask=mask, dtype=dtype, fill_value=fill_value)

        tbl.meta['name'] = self.catalog.name
        tbl.meta['idname'] = self.catalog.idname
        try:
            tbl.meta['raname'] = self.catalog.raname
            tbl.meta['decname'] = self.catalog.decname
            tbl.meta['zname'] = self.catalog.zname
            tbl.meta['zconfname'] = self.catalog.zconfname
        except AttributeError:
            pass

        return tbl


class BaseCatalog:
    """Handle Catalogs by the way of a database table, on top of
    `dataset.Table`.

    Parameters
    ----------
    name : str
        Name of the catalog and associated SQL table.
    db : `dataset.Database`
        The database object.
    idname : str
        Name of the 'id' column.
    primary_id : str
        The primary id for the SQL table, must be a column name. Defaults to
        ``idname``.
    author : str
        Name of the person making changes. This is stored in the history table,
        and is taken from the settings file.

    """

    catalog_type = ''

    def __init__(self, name, db, idname='ID', primary_id=None, author=None):
        self.name = name
        self.db = db
        self.idname = idname
        self.author = author
        self.logger = logging.getLogger(__name__)

        if not re.match(r'[0-9a-zA-Z_]+$', self.name):
            warnings.warn(
                'catalog name should contain only ascii letters '
                '(a-zA-Z), digits (0-9) and underscore, otherwise '
                'using it in a column name will fail',
                UserWarning,
            )

        if self.name not in self.db:
            self.logger.debug(
                'create table %s (primary key: %s)', self.name, self.idname
            )

        # Get the reference to the db table, which is created if needed
        primary_id = primary_id or self.idname
        self.table = self.db.create_table(self.name, primary_id=primary_id)
        # Force the creation of the SQLATable
        assert self.table.table is not None

        # Insert default meta about the table if it doesn't exist yet
        if self.meta is None:
            self.update_meta(
                creation_date=datetime.utcnow().isoformat(),
                type=self.catalog_type,
                parent_cat=None,
                idname=self.idname,
                primary_id=primary_id,
            )

    def __len__(self):
        return len(self.table)

    def __repr__(self):
        return f"<{self.__class__.__name__}('{self.name}', {len(self)} rows)>"

    @lazyproperty
    def history(self):
        # Create the history table and its columns
        tbl = self.db.create_table('history', primary_id='_id')
        assert tbl.table is not None
        date = datetime.utcnow()
        tbl._sync_columns(
            dict(catalog='', id=0, date=date, msg='', data='', author=''), True
        )
        return tbl

    def log(self, id_, msg, row=None, **kwargs):
        if row is not None:
            try:
                row = json.dumps(row)
            except TypeError as e:
                self.logger.debug('log error: %s', e)
                row = None

        # Force the columns creation if additional columns are passed.
        # Otherwise the automatic creation of columns is disabled to save some
        # processing time.
        ensure = bool(kwargs)

        date = datetime.utcnow()
        self.history.insert(
            dict(
                catalog=self.name,
                id=id_,
                author=self.author,
                date=date,
                msg=msg,
                data=row,
                **kwargs,
            ),
            ensure=ensure,
        )

    def get_log(self, id_):
        return self.history.find(catalog=self.name, id=int(id_))

    def max(self, colname):
        """Return the maximum value of a column."""
        return self.db.executable.execute(sql.func.max(self.c[colname])).scalar()

    @property
    def c(self):
        """The list of columns from the SQLAlchemy table object."""
        return self.table.table.c

    @lazyproperty
    def meta(self):
        """Return metadata associated with the catalog."""
        return self.db['catalogs'].find_one(name=self.name)

    def update_meta(self, **kwargs):
        """Update metadata associated with the catalog."""
        self.db['catalogs'].upsert({'name': self.name, **kwargs}, ['name'])
        del self.meta

    def info(self):
        """Print information about the catalog (columns etc.)."""
        print(f"{self.__class__.__name__} '{self.name}' - {len(self)} rows.")
        if self.meta:
            maxlen = max(len(k) for k in self.meta.keys()) + 1
            meta = '\n'.join(
                f'- {k:{maxlen}s}: {v}'
                for k, v in self.meta.items()
                if k not in ('id', 'name')
            )
            print(f"\nMetadata:\n{meta}")

        maxlen = max(len(k) for k in self.table.table.columns.keys()) + 1
        columns = '\n'.join(
            f"- {k:{maxlen}s}: {v.type} {v.default or ''}"
            for k, v in self.table.table.columns.items()
        )
        print(f"\nColumns:\n{columns}")

    def drop(self):
        """Drop the SQL table and its metadata."""
        self.table.drop()
        self.db['catalogs'].delete(name=self.name)
        del self.table

    def _prepare_rows_for_insert(self, rows, version=None, show_progress=True):
        # Convert Astropy Table to a list of dict
        if isinstance(rows, Table):
            # Change minus to underscore in column names because we can't have
            # a minus in a database column name. We do it only on the table
            # because passing a dictionary to the function will generally
            # happen when the dictionary comes from Musex.
            for colname in rows.colnames:
                if '-' in colname:
                    new_colname = colname.replace('-', '_')
                    rows.rename_column(colname, new_colname)
                    self.logger.warning(
                        "The column %s was renamed to %s.", colname, new_colname
                    )
            rows = table_to_odict(rows)

        if version is not None:
            rows = [row.copy() for row in rows]
            for row in rows:
                row.setdefault('version', version)

        # Create missing columns
        self.table._sync_columns(rows[0], True)

        if show_progress:
            rows = progressbar(rows)

        return rows

    def insert(self, rows, version=None, show_progress=True):
        """Insert rows in the catalog.

        Parameters
        ----------
        rows : list of dict or `astropy.table.Table`
            List of rows or Astropy Table to insert. Each row must be a dict
            with column names as keys.
        version : str
            Version added to each row (if not available in the row values).
        show_progress : bool
            Show a progress bar.

        """
        rows = self._prepare_rows_for_insert(
            rows, version=version, show_progress=show_progress
        )
        ids = []
        assert self.history is not None  # create table to avoid schema warning
        with self.db as tx:
            tbl = tx[self.name]
            for row in rows:
                ids.append(tbl.insert(row, ensure=False))
                self.log(ids[-1], f'inserted from input catalog', row=row)

        if ids and self.idname in self.table.columns:
            self.update_meta(maxid=self.max(self.idname))

        if not tbl.has_index(self.idname):
            tbl.create_index(self.idname)
        self.logger.info('%d rows inserted', len(ids))
        return ids

    def upsert(self, rows, version=None, show_progress=True, keys=None):
        """Insert or update rows in the catalog.

        Parameters
        ----------
        rows : list of dict or `astropy.table.Table`
            List of rows or Astropy Table to insert. Each row must be a dict
            with column names as keys.
        version : str
            Version added to each row (if not available in the row values).
        show_progress : bool
            Show a progress bar.
        keys : list of str
            If rows with matching keys exist they will be updated, otherwise
            a new row is inserted in the table. Defaults to
            ``[idname, 'version']``.

        """
        count = defaultdict(int)
        rows = self._prepare_rows_for_insert(
            rows, version=version, show_progress=show_progress
        )

        if keys is None:
            keys = [self.idname]
            if version is not None:
                keys.append('version')

        # Check that all the rows contain the keys.
        try:
            if not set(keys).issubset(set(rows.colnames)):
                raise KeyError(
                    "The table does not contain all the keys: %s" % ", ".join(keys)
                )
        except AttributeError:
            for row in rows:
                if not set(keys).issubset(set(row)):
                    raise KeyError(
                        "At least one dictionary in the list does "
                        "not contain all the keys: %s" % ", ".join(keys)
                    )

        assert self.history is not None  # create table to avoid schema warning
        with self.db as tx:
            tbl = tx[self.name]
            for row in rows:
                res = tbl.upsert(row, keys)
                op = 'updated' if res is True else 'inserted'
                count[op] += 1
                if res is True:
                    # if row was updated we need to get its ID
                    res = row.get(self.idname)
                if res is not None:
                    # log operation if we have an ID
                    self.log(res, f'{op} from input catalog', row=row)

        if count['inserted']:
            self.update_meta(maxid=self.max(self.idname))

        if not tbl.has_index(self.idname):
            tbl.create_index(self.idname)
        self.logger.info(
            '%d rows inserted, %s updated', count['inserted'], count['updated']
        )

    def select(self, whereclause=None, columns=None, **params):
        """Select rows in the catalog.

        Parameters
        ----------
        whereclause :
            The SQLAlchemy selection clause.
        columns : list of str
            List of columns to retrieve (all columns if None).
        **params
            Additional parameters are passed to `sqlalchemy.sql.select`.

        """
        if columns is not None:
            columns = [self.c[col] for col in columns]
        else:
            columns = [self.table.table]

        wc = sql.text(whereclause) if isinstance(whereclause, str) else whereclause
        query = sql.select(columns=columns, whereclause=wc, **params)
        res = self.db.query(query)
        return ResultSet(
            res, whereclause=whereclause, catalog=self, columns=query.columns
        )

    def select_id(self, id_):
        """Return a dict with all keys for a given ID."""
        return self.table.find_one(**{self.idname: id_})

    def select_ids(self, idlist, columns=None, idcolumn=None, **params):
        """Select rows with a list of IDs.

        If a column name is provided in `idcolumn`, this column will be used
        to select the IDs in; else the catalog main ID column (`idname`) will
        be used.

        Parameters
        ----------
        idlist : int or list of int
            List of IDs.
        columns : list of str
            List of columns to retrieve (all columns if None).
        idcolumn : str
            Name of the column containing the IDs when not using the default
            column.
        params : dict
            Additional parameters are passed to `dataset.Database.query`.

        """
        if not isiter(idlist):
            idlist = [idlist]
        elif isinstance(idlist, np.ndarray):
            idlist = idlist.tolist()

        if idcolumn is None:
            idcolumn = self.idname

        if len(idlist) > 999 and self.db.engine.driver == 'pysqlite':
            warnings.warn('Selecting too many ids will fail with SQLite', UserWarning)

        whereclause = self.c[idcolumn].in_(idlist)
        return self.select(whereclause=whereclause, columns=columns, **params)

    def join(
        self,
        othercats,
        whereclause=None,
        columns=None,
        keys=None,
        use_labels=True,
        isouter=False,
        debug=False,
        **params,
    ):
        """Join catalog with other catalogs.

        Parameters
        ----------
        whereclause :
            The SQLAlchemy selection clause.
        columns : list of str
            List of columns to retrieve (all columns if None).
        keys : list of tuple
            List of keys to do the join for each catalog. If None, the IDs of
            each catalog are used (from the ``idname`` attribute). Otherwise it
            must be a list of tuples, where each tuple contains the key for
            self and the key for the other catalog.
        use_labels : bool
            By default, all columns are selected which may gives name
            conflicts. So ``use_labels`` allows to rename the columns by
            prefixing the name with the catalog name.
        isouter : bool
            If True, render a LEFT OUTER JOIN, instead of JOIN.
        params : dict
            Additional parameters are passed to `sqlalchemy.sql.select`.

        """
        tbl = self.table.table
        if isinstance(othercats, BaseCatalog):
            othercats = [othercats]
        tables = [cat.table.table for cat in othercats]
        if columns is None:
            columns = [tbl] + tables

        if keys is None:
            keys = [(self.idname, cat.idname) for cat in othercats]

        query = sql.select(
            columns, use_labels=use_labels, whereclause=whereclause, **params
        )
        joincl = tbl
        for (key1, key2), other in zip(keys, tables):
            joincl = joincl.join(other, tbl.c[key1] == other.c[key2], isouter=isouter)
        query = query.select_from(joincl)

        # FIXME: .reduce_columns() should allow to filter duplicate columns
        # (removing the need to use use_labels), but it does not work for float
        # values ?

        if debug:
            print(query)
        res = self.db.query(query)
        return ResultSet(
            res, whereclause=whereclause, catalog=self, columns=query.columns
        )

    def update_id(self, id_, **kwargs):
        """Update values for a given ID."""
        return self.table.update({self.idname: id_, **kwargs}, [self.idname])

    def update_column(self, name, values):
        """Update (or create) a column ``name`` with the given values."""
        if np.isscalar(values):
            values = [values] * len(self)
        if name not in self.table.columns:
            self.logger.info("creating column '%s.%s'", self.name, name)
            self.table.create_column_by_example(name, values[0])
        with self.db as tx:
            tbl = tx[self.name]
            for i, row in enumerate(tbl.find()):
                row[name] = values[i]
                tbl.upsert(row, [self.idname], ensure=False)


class Catalog(BaseCatalog):
    """Handle user catalogs, with spatial information.

    Parameters
    ----------
    name : str
        Name of the catalog and associated SQL table.
    db : `dataset.Database`
        The database object.
    idname : str
        Name of the 'id' column.
    raname : str, optional
        Name of the 'ra' column.
    decname : str, optional
        Name of the 'dec' column.
    zname : str, optional
        Name of the 'z' column.
    zconfname : str, optional
        Name of the 'confid' column.
    primary_id : str
        The primary id for the SQL table, must be a column name.
    author : str
        Name of the person making changes. This is stored in the history table,
        and is taken from the settings file.
    prefix : str
        Prefix for the extension name in sources export.

    """

    catalog_type = 'user'

    def __init__(
        self,
        name,
        db,
        idname='ID',
        raname=None,
        decname=None,
        zname=None,
        zconfname=None,
        primary_id=None,
        author=None,
        prefix=None,
    ):
        super().__init__(name, db, idname=idname, primary_id=primary_id, author=author)
        self.raname = raname
        self.decname = decname
        self.zname = zname
        self.zconfname = zconfname
        self.prefix = prefix
        self.update_meta(
            raname=self.raname,
            decname=self.decname,
            prefix=self.prefix,
            zconfname=self.zconfname,
            zname=self.zname,
        )

        # FIXME: sadly this doesn't work well currently, it is not taken into
        # account until an insert is done
        # from sqlalchemy.schema import ColumnDefault
        # self.table._sync_columns({idname: 1, raname: 1., decname: 1.,
        #                           'active': True, 'merged': False}, True)
        # self.c.active.default = ColumnDefault(True)
        # self.c.merged.default = ColumnDefault(False)
        # self.table.create_column('active', self.db.types.boolean)
        # self.table.create_column('merged', self.db.types.boolean)

    @classmethod
    def from_parent_cat(cls, parent_cat, name, whereclause, primary_id=None):
        """Create a new `Catalog` from another one."""
        cat = cls(
            name,
            parent_cat.db,
            primary_id=primary_id,
            idname=parent_cat.idname,
            raname=getattr(parent_cat, 'raname', None),
            decname=getattr(parent_cat, 'decname', None),
            zname=getattr(parent_cat, 'zname', None),
            zconfname=getattr(parent_cat, 'zconfname', None),
            author=parent_cat.author,
            prefix=parent_cat.prefix,
        )
        if parent_cat.meta.get('query'):
            # Combine query string with the parent cat's one
            whereclause = f"{parent_cat.meta['query']} AND {whereclause}"
        cat.update_meta(
            parent_cat=parent_cat.name,
            maxid=parent_cat.meta['maxid'],
            query=whereclause,
        )
        return cat

    def select(
        self, whereclause=None, columns=None, wcs=None, margin=0, mask=None, **params
    ):
        """Select rows in the catalog.

        For spatial catalogs, this method allows to pass as WCS to select only
        the sources that fall inside.

        Parameters
        ----------
        whereclause :
            The SQLAlchemy selection clause.
        columns : list of str
            List of columns to retrieve (all columns if None).
        wcs : `mpdaf.obj.WCS`
            If present sources are selected inside the given WCS.
        margin : float
            Margin from the edges (pixels) for the WCS selection.
        mask : array-like
            If addition to the WCS, corresponding mask used to select sources
            (1 to mask).
        params : dict
            Additional parameters are passed to `sqlalchemy.sql.select`.

        """
        # We need the position to select inside a WCS.
        if wcs is not None and columns is not None:
            if self.raname not in columns:
                columns.append(self.raname)
            if self.decname not in columns:
                columns.append(self.decname)

        res = super().select(whereclause=whereclause, columns=columns, **params)

        if wcs is not None:
            t = res.as_table()
            t = t.select(
                wcs, ra=self.raname, dec=self.decname, margin=margin, mask=mask
            )
            res = ResultSet(
                table_to_odict(t),
                whereclause=whereclause,
                catalog=self,
                columns=res.columns,
            )

        return res

    def skycoord(self):
        """Return an `astropy.coordinates.SkyCoord` object."""
        columns = [self.idname, self.raname, self.decname]
        tbl = self.select(columns=columns).as_table()
        return tbl.to_skycoord(ra=self.raname, dec=self.decname)

    def merge_sources(self, idlist, id_=None, dataset=None, weights_colname=None):
        """Merge sources into one.

        A new source is created, with the "merged" column set to True. The new
        id is stored in the "merged_in" column for the input sources.

        # TODO: implement merging sources with masks

        Parameters
        ----------
        idlist : list
            List of ids to merge.
        id_ : int
            The new ID for the merged source. If not given, it is automatically
            determined from the maxid and autoincremented.
        dataset : `musex.Dataset`
            The associated dataset. To compute the masks this dataset must be
            given, and must be the same as the one used for `attach_dataset`.
        weights_colname : str
            Name of a column to be used as weights.

        """
        # compute minimum id for "custom" sources
        # TODO: add a setting for this ("customid_start") ?
        maxid = self.max(self.idname)
        cat_maxid = self.meta['maxid']

        sources = self.select_ids(idlist)
        if len(sources) == 0:
            raise ValueError('no sources found')

        tbl = sources.as_table()
        coords = np.stack([tbl[self.decname].data, tbl[self.raname].data])
        weights = tbl[weights_colname] if weights_colname else None
        dec, ra = np.ma.average(coords, weights=weights, axis=1)

        # version
        versions = set(s.get('version') for s in sources)
        if len(versions) == 1:
            version = versions.pop()
        else:
            self.logger.warning('sources have different version')
            version = None

        row = {'merged': True, self.raname: ra, self.decname: dec, 'version': version}

        if id_ is not None:
            row[self.idname] = id_
        elif maxid <= cat_maxid:
            row[self.idname] = 10 ** (len(str(cat_maxid)))

        # Create missing columns
        self.table._sync_columns({'active': False, 'merged_in': 0, **row}, True)

        with self.db as tx:
            tbl = tx[self.name]
            # create new (merged) source
            newid = tbl.insert(row)
            if id_ is not None:
                assert newid == id_, 'this should never happen!'
            # deactivate the other sources
            idname = self.idname
            for s in sources:
                if s.get('merged_in') is not None:
                    raise ValueError(f'source {s[idname]} is already merged')
                tbl.upsert(
                    {idname: s[idname], 'active': False, 'merged_in': newid},
                    [idname],
                    ensure=False,
                )

        self.logger.info('sources %s have been merged in %s', idlist, newid)
        return newid

    def get_ids_merged_in(self, id_):
        """Return the IDs that were merged in ``id_``."""
        return [o[self.idname] for o in self.table.find(merged_in=id_)]

    def add_column_with_merged_ids(self, name):
        """Add (or update) a str column with a comma-separated list of the
        merged ids.
        """
        if name in RESERVED_COLNAMES:
            raise ValueError(
                f"column name '{name}' cannot be used, it is "
                f"reserved for specific purpose"
            )

        merged = defaultdict(list)
        for o in self.select(
            self.c.merged_in.isnot(None), columns=[self.idname, 'merged_in']
        ):
            merged[o['merged_in']].append(o[self.idname])

        merged = {k: ",".join(str(x) for x in sorted(v)) for k, v in merged.items()}
        val = [
            merged.get(x[self.idname], str(x[self.idname]))
            for x in self.select(columns=[self.idname])
        ]
        self.update_column(name, val)


class InputCatalog(Catalog):
    """Handles catalogs imported from an existing file."""

    catalog_type = 'input'

    def __init__(self, name, db, **kwargs):
        super().__init__(name, db, **kwargs)
        self.version_meta = None

    @classmethod
    def from_settings(cls, name, db, **kwargs):
        """Create an InputCatalog from the settings file."""
        init_keys = ('idname', 'raname', 'decname', 'zname', 'zconfname', 'author')
        kw = {k: v for k, v in kwargs.items() if k in init_keys}
        cat = cls(name, db, **kw)

        for key in ('catalog', 'version'):
            if kwargs.get(key) is None:
                raise ValueError(f'an input {key} is required')
            setattr(cat, key, kwargs[key])

        cat.params = kwargs.copy()
        cat.version_meta = kwargs.get('version_meta')
        return cat

    def ingest_input_catalog(
        self,
        catalog=None,
        limit=None,
        upsert=False,
        keys=None,
        show_progress=True,
        version_meta=None,
    ):
        """Ingest an input catalog.

        The catalog to ingest can be given with the ``catalog`` parameter or
        in the setting file.  Existing records can be updated using
        ``upsert=True`` and ``keys``.

        The ``limit`` parameter is used to limit the number of rows from the
        catalog to ingest.

        Parameters
        ----------
        catalog : str or `astropy.table.Table`
            Table to insert.
        limit : int
            To limit the number of rows from the catalog.
        upsert : bool
            If True, existing rows with the same values for ``keys`` are
            updated.
        keys : list of str
            If rows with matching keys exist they will be updated, otherwise
            a new row is inserted in the table. Defaults to
            ``[idname, 'version']``.
        show_progress : bool
            Show a progress bar.
        version_meta : str, optional
            Keyword in the catalog file that is used to identify the version of
            the catalog. It is used for ORIGIN catalogs to check that the
            sources correspond to the catalog.

        """
        catalog = catalog or self.catalog
        version_meta = version_meta or self.version_meta

        # Catalog
        if isinstance(catalog, str):
            self.logger.info('ingesting catalog %s', catalog)
            catalog = Table.read(catalog)
        else:
            self.logger.info('ingesting catalog')

        if limit:
            self.logger.info('keeping only %d rows', limit)
            catalog = catalog[:limit]

        if upsert:
            status = 'updated'
            self.upsert(
                catalog, version=self.version, keys=keys, show_progress=show_progress
            )
        else:
            if self.meta.get('status') is not None:
                self.logger.warning(
                    'catalogue already ingested, use upsert=True to update'
                )
                return

            status = 'inserted'
            self.insert(catalog, version=self.version, show_progress=show_progress)

        meta = {
            'creation_date': datetime.utcnow().isoformat(),
            'type': self.catalog_type,
            'status': status,
        }
        if version_meta is not None:
            meta.update(
                {'version_meta': version_meta, version_meta: catalog.meta[version_meta]}
            )
        self.update_meta(**meta)


class MarzCatalog(InputCatalog):
    """Handles catalogs imported from MarZ."""

    catalog_type = 'marz'

    def select_flat(self, *, limit_to_cat=None, max_order=5, columns=None):
        """Creates a ResultSet with one MarZ solution per row.

        This method creates a ResultSet converting the one source per row
        MarZ format to a one solution per row format, adding a `marz_sol`
        column containing the solution order. For instance, the `AutoZ` column
        with `marz_sol` to 2 contains the value of the `AutoZ2` for the
        corresponding ID.

        By default, the columns AutoZ, AutoTID, AutoTN, and AutoXCor are
        exported (in addition to `catalog`, `ID`, `RA`, `DEC`, and `QOP`) up to
        order 5. If the Marz catalog has different columns or orders, the
        `max_order` and `columns` parameters must be adapted.

        Parameters
        ----------
        limit_to_cat : str, optional
            If provided, only the lines for the corresponding catalog are used
            in `marzcat`.
        maximum_order : int, optional
            Maximum order of the solutions to take; e.g. 2 will export only the
            first two solutions.
        columns : list of str, optional
            Name of the columns in addition to `catalog`, `ID`, `RA`, `DEC`,
            and `QOP` to export.

        """
        if columns is None:
            columns = ["AutoZ", "AutoTID", "AutoTN", "AutoXCor"]

        if limit_to_cat is None:
            marz_table = self.select().as_table()
        else:
            marz_table = self.select(
                whereclause=(self.c.catalog == limit_to_cat)
            ).as_table()

        marz_sol = []
        for sol_order in ["", "2", "3", "4", "5"][:max_order]:
            sub_table = marz_table[
                ['catalog', 'ID', 'RA', 'DEC', 'QOP']
                + [col + sol_order for col in columns]
            ]
            for col in columns:
                sub_table[col + sol_order].name = col
            if sol_order == "":
                sol_order = "1"
            sub_table["marz_sol"] = int(sol_order)
            marz_sol.append(sub_table)

        marz_sol = vstack(marz_sol)
        marz_sol.sort(['catalog', 'ID', 'marz_sol'])
        # We must add a new unique identifier column as the ID columns contains
        # duplicates by definition.
        marz_sol["_id"] = np.arange(len(marz_sol)) + 1

        return ResultSet(
            results=table_to_odict(marz_sol), catalog=self, columns=marz_sol.colnames
        )


class IdMapping(BaseCatalog):
    """Handles Id mapping."""

    catalog_type = 'id'

    def insert(self, rows, cat, allow_upsert=True):
        """Insert new ids which references other ids from `cat`."""
        catname = get_cat_name(cat)
        rows = np.atleast_2d(rows)
        if rows.shape[1] != 2:
            raise ValueError('rows must be a tuple of ids or a list of tuple')

        keys = (self.idname, f'{catname}_id')
        rows = [dict(zip(keys, (int(x) for x in row))) for row in rows]
        if allow_upsert:
            super().upsert(rows, show_progress=False, keys=[self.idname])
        else:
            super().insert(rows, show_progress=False)

    def add_ids(self, idlist, cat):
        """Insert new ids which references other ids from `cat`."""
        idlist = np.atleast_1d(idlist)
        key = '{}_id'.format(get_cat_name(cat))
        rows = [{key: int(id_)} for id_ in idlist]
        super().insert(rows, show_progress=False)
