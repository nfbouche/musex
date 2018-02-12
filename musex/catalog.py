import importlib
import logging
import numpy as np
import os
import re
import textwrap
import warnings

import astropy.units as u
from astropy.table import Table
from astropy.utils.decorators import lazyproperty

from collections import OrderedDict, defaultdict
from collections.abc import Sequence
from datetime import datetime
from joblib import delayed, Parallel
from mpdaf.sdetect import Catalog as _Catalog
from numpy import ma
from os.path import exists, relpath
from pathlib import Path
from sqlalchemy import sql

from .segmap import SegMap
from .utils import struct_from_moffat_fwhm, isiter, progressbar

DIRNAME = os.path.abspath(os.path.dirname(__file__))

__all__ = ('table_to_odict', 'ResultSet', 'Catalog', 'BaseCatalog',
           'InputCatalog', 'MarzCatalog', 'IdMapping')

FILL_VALUES = {int: -9999, float: np.nan, str: ''}


def load_input_catalogs(settings, db):
    """Load input catalogs defined in the settings."""
    catalogs = {}
    for name, conf in settings['catalogs'].items():
        if 'class' in conf:
            mod, class_ = conf['class'].rsplit('.', 1)
            mod = importlib.import_module(mod)
            cls = getattr(mod, class_)
        else:
            cls = InputCatalog
        catalogs[name] = cls.from_settings(name, db, **conf)
    return catalogs


def table_to_odict(table):
    """Convert a `astropy.table.Table` to a list of `OrderedDict`."""
    colnames = table.colnames
    return [OrderedDict(zip(colnames, row))
            for row in zip(*[c.tolist() for c in table.columns.values()])]


def get_cat_name(res_or_cat):
    """Helper function to get the catalog name from different objects."""
    if isinstance(res_or_cat, str):
        return res_or_cat
    elif isinstance(res_or_cat, BaseCatalog):
        return res_or_cat.name
    elif isinstance(res_or_cat, ResultSet):
        return res_or_cat.catalog.name
    else:
        raise ValueError('cat must be a Catalog instance or name')


def _get_psf_convolution_params(convolve_fwhm, segmap, psf_threshold):
    if convolve_fwhm:
        # compute a structuring element for the dilatation, to simulate
        # a convolution with a psf, but faster.
        dilateit = 1
        # logging.getLogger(__name__).debug(
        #     'dilate with %d iterations, psf=%.2f', dilateit, convolve_fwhm)
        struct = struct_from_moffat_fwhm(segmap.img.wcs, convolve_fwhm,
                                         psf_threshold=psf_threshold)
    else:
        dilateit = 0
        struct = None
    return dilateit, struct


class ResultSet(Sequence):
    """Contains the result of a query on the database."""

    def __init__(self, results, whereclause=None, catalog=None, columns=None):
        self.results = list(results)
        # TODO: use weakref here ?
        self.catalog = catalog
        self.columns = columns

        self.whereclause = whereclause
        if whereclause is not None and not isinstance(whereclause, str):
            self.whereclause = str(whereclause.compile(
                compile_kwargs={"literal_binds": True}))

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
        for col, val in zip(self.columns,
                            zip(*[r.values() for r in self.results])):
            dtype = col.type.python_type
            fill_value = FILL_VALUES.get(dtype)
            mask = [v is None for v in val]
            val = [fill_value if v is None else v for v in val]
            tbl[col.name] = ma.array(val, mask=mask, dtype=dtype,
                                     fill_value=fill_value)

        tbl.meta['name'] = self.catalog.name
        tbl.meta['idname'] = self.catalog.idname
        tbl.meta['raname'] = self.catalog.raname
        tbl.meta['decname'] = self.catalog.decname
        return tbl


class BaseCatalog:
    """Handle Catalogs by the way of a database table.

    Parameters
    ----------
    name: str
        Name of the catalog.
    db: `dataset.Database`
        The database object.
    idname: str
        Name of the 'id' column.
    raname: str
        Name of the 'ra' column.
    decname: str
        Name of the 'dec' column.
    segmap: str
        Path to the segmentation map file associated with the catalog.
    primary_id: str
        The primary id for the SQL table, must be a column name.

    """

    catalog_type = ''

    def __init__(self, name, db, idname='ID', raname='RA', decname='DEC',
                 segmap=None, primary_id=None):
        self.name = name
        self.db = db
        self.idmap = None
        self.segmap = segmap
        self.idname = idname
        self.raname = raname
        self.decname = decname
        self.logger = logging.getLogger(__name__)

        if not re.match(r'[0-9a-zA-Z_]+$', self.name):
            warnings.warn('catalog name should contain only ascii letters '
                          '(a-zA-Z), digits (0-9) and underscore, otherwise '
                          'using it in a column name will fail', UserWarning)

        # if self.name not in self.db:
        #     self.logger.debug('create table %s (primary key: %s)',
        #                       self.name, self.idname)

        # Get the reference to the db table, which is created if needed
        primary_id = primary_id or self.idname
        self.table = self.db.create_table(self.name, primary_id=primary_id)
        # Force the creation of the SQLATable
        assert self.table.table is not None

        # Insert default meta about the table if it doesn't exist yet
        if self.meta is None:
            self.update_meta(creation_date=datetime.utcnow().isoformat(),
                             type=self.catalog_type, parent_cat=None,
                             raname=self.raname, decname=self.decname,
                             idname=self.idname, segmap=self.segmap)

    def __len__(self):
        return self.table.count()

    def __repr__(self):
        return f"<{self.__class__.__name__}('{self.name}', {len(self)} rows)>"

    @property
    def history(self):
        # Create the history table and its columns
        tbl = self.db.create_table('history', primary_id='_id')
        assert tbl.table is not None
        tbl._sync_columns(dict(catalog='', id=0, date='', msg=''), True)
        return tbl

    def log(self, id_, msg):
        self.history.insert(dict(catalog=self.name, id=id_,
                                 date=datetime.utcnow().isoformat(), msg=msg),
                            ensure=False)

    def get_log(self, id_):
        return self.history.find(catalog=self.name, id=id_)

    def max(self, colname):
        """Return the maximum value of a column."""
        return self.db.executable.execute(
            sql.func.max(self.c[colname])).scalar()

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
        print(textwrap.dedent(f"""\
        {self.__class__.__name__} '{self.name}' - {len(self)} rows.

        Workdir: {getattr(self, 'workdir', '')}
        Segmap : {self.segmap}
        """))

        if self.meta:
            maxlen = max(len(k) for k in self.meta.keys()) + 1
            meta = '\n'.join(f'- {k:{maxlen}s}: {v}'
                             for k, v in self.meta.items()
                             if k not in ('id', 'name'))
            print(f"Metadata:\n{meta}\n")

        maxlen = max(len(k) for k in self.table.table.columns.keys()) + 1
        columns = '\n'.join(f"- {k:{maxlen}s}: {v.type} {v.default or ''}"
                            for k, v in self.table.table.columns.items())
        print(f"Columns:\n{columns}\n")

    def drop(self):
        """Drop the SQL table and its metadata."""
        self.table.drop()
        self.db['catalogs'].delete(name=self.name)

    def _prepare_rows_for_insert(self, rows, version=None, show_progress=True):
        # Convert Astropy Table to a list of dict
        if isinstance(rows, Table):
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
        rows: list of dict or `astropy.table.Table`
            List of rows or Astropy Table to insert. Each row must be a dict
            with column names as keys.
        version: str
            Version added to each row (if not available in the row values).
        show_progress: bool
            Show a progress bar.

        """
        rows = self._prepare_rows_for_insert(rows, version=version,
                                             show_progress=show_progress)
        ids = []
        assert self.history is not None  # create table to avoid schema warning
        with self.db as tx:
            tbl = tx[self.name]
            for row in rows:
                ids.append(tbl.insert(row, ensure=False))
                self.log(ids[-1], f'inserted from input catalog')

        if self.idmap:
            self.idmap.add_ids(ids, self.name)

        if not tbl.has_index(self.idname):
            tbl.create_index(self.idname)
        self.logger.info('%d rows inserted', len(ids))
        return ids

    def upsert(self, rows, version=None, show_progress=True, keys=None):
        """Insert or update rows in the catalog.

        Parameters
        ----------
        rows: list of dict or `astropy.table.Table`
            List of rows or Astropy Table to insert. Each row must be a dict
            with column names as keys.
        version: str
            Version added to each row (if not available in the row values).
        show_progress: bool
            Show a progress bar.
        keys: list of str
            If rows with matching keys exist they will be updated, otherwise
            a new row is inserted in the table. Defaults to
            ``[idname, 'version']``.

        """
        count = defaultdict(list)
        rows = self._prepare_rows_for_insert(rows, version=version,
                                             show_progress=show_progress)

        if keys is None:
            keys = [self.idname]
            if version is not None:
                keys.append('version')

        assert self.history is not None  # create table to avoid schema warning
        with self.db as tx:
            tbl = tx[self.name]
            for row in rows:
                res = tbl.upsert(row, keys, ensure=False)
                op = 'updated' if res is True else 'inserted'
                count[op].append(res)
                self.log(res, f'{op} from input catalog')

        if self.idmap:
            self.idmap.add_ids(count['inserted'], self.name)

        if not tbl.has_index(self.idname):
            tbl.create_index(self.idname)
        self.logger.info('%d rows inserted, %s updated',
                         len(count['inserted']), len(count['updated']))

    def select(self, whereclause=None, columns=None, wcs=None, margin=0,
               **params):
        """Select rows in the catalog.

        Parameters
        ----------
        whereclause:
            The SQLAlchemy selection clause.
        columns: list of str
            List of columns to retrieve (all columns if None).
        wcs: `mpdaf.obj.WCS`
            If present sources are selected inside the given WCS.
        margin: float
            Margin from the edges (pixels) for the WCS selection.
        params: dict
            Additional parameters are passed to `sqlalchemy.sql.select`.

        """
        if columns is not None:
            if wcs is not None:
                if self.raname not in columns:
                    columns.append(self.raname)
                if self.decname not in columns:
                    columns.append(self.decname)
            columns = [self.c[col] for col in columns]
        else:
            columns = [self.table.table]

        query = sql.select(columns=columns, whereclause=whereclause, **params)
        res = self.db.query(query)
        res = ResultSet(res, whereclause=whereclause, catalog=self,
                        columns=query.columns)

        if wcs is not None:
            t = res.as_table()
            t = t.select(wcs, ra=self.raname, dec=self.decname, margin=margin)
            res = ResultSet(table_to_odict(t), whereclause=whereclause,
                            catalog=self, columns=query.columns)

        return res

    def select_id(self, id_):
        """Return a dict with all keys for a given ID."""
        if self.idmap:
            res = self.idmap.table.find_one(ID=id_)
            if res and f'{self.name}_id' in res:
                id_ = res[f'{self.name}_id']
            else:
                return None
        return self.table.find_one(**{self.idname: id_})

    def select_ids(self, idlist, columns=None, **params):
        """Select rows with a list of IDs.

        Parameters
        ----------
        idlist: int or list of int
            List of IDs.
        columns: list of str
            List of columns to retrieve (all columns if None).
        params: dict
            Additional parameters are passed to `dataset.Database.query`.

        """
        if not isiter(idlist):
            idlist = [idlist]
        elif isinstance(idlist, np.ndarray):
            idlist = idlist.tolist()

        if len(idlist) > 999 and self.db.engine.driver == 'pysqlite':
            warnings.warn('Selecting too many ids will fail with SQLite',
                          UserWarning)

        whereclause = self.c[self.idname].in_(idlist)
        return self.select(whereclause=whereclause, columns=columns, **params)

    def join(self, othercats, whereclause=None, columns=None, keys=None,
             use_labels=True, isouter=False, debug=False, **params):
        """Join catalog with other catalogs.

        Parameters
        ----------
        whereclause:
            The SQLAlchemy selection clause.
        columns: list of str
            List of columns to retrieve (all columns if None).
        keys: list of tuple
            List of keys to do the join for each catalog. If None, the IDs of
            each catalog are used (from the ``idname`` attribute). Otherwise it
            must be a list of tuples, where each tuple contains the key for
            self and the key for the other catalog.
        use_labels: bool
            By default, all columns are selected which may gives name
            conflicts. So ``use_labels`` allows to rename the columns by
            prefixinf the name with the catalog name.
        isouter: bool
            If True, render a LEFT OUTER JOIN, instead of JOIN.
        params: dict
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

        query = sql.select(columns, use_labels=use_labels,
                           whereclause=whereclause, **params)
        joincl = tbl
        for (key1, key2), other in zip(keys, tables):
            joincl = joincl.join(other, tbl.c[key1] == other.c[key2],
                                 isouter=isouter)
        query = query.select_from(joincl)

        # FIXME: .reduce_columns() should allow to filter duplicate columns
        # (removing the need to use use_labels), but it does not work for float
        # values ?

        if debug:
            print(query)
        res = self.db.query(query)
        return ResultSet(res, whereclause=whereclause, catalog=self,
                         columns=query.columns)

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

    def add_to_source(self, src, row, **kwargs):
        """Add information to the Source object.

        FIXME: see how to improve conf here.

        """
        # Add catalog as a BinTableHDU
        cat = self.select(columns=kwargs.get('columns')).as_table()
        wcs = src.images[kwargs.get('select_in', 'WHITE')].wcs
        scat = cat.select(wcs, ra=self.raname, dec=self.decname, margin=0)
        scat['DIST'] = scat.edgedist(wcs, ra=self.raname, dec=self.decname)
        # FIXME: is it the same ?
        # cat = in_catalog(cat, src.images['HST_F775W_E'], quiet=True)
        catname = f"{kwargs['prefix']}_{kwargs.get('name', 'CAT')}"
        self.logger.debug('Adding catalog %s (%d rows)', catname, len(scat))
        src.add_table(scat, catname)
        src.REFCAT = catname
        cat = src.tables[catname]
        cat.meta['name'] = self.name
        cat.meta['idname'] = self.idname
        cat.meta['raname'] = self.raname
        cat.meta['decname'] = self.decname

        # Add redshifts
        for name, val in kwargs.get('redshifts', {}).items():
            try:
                if isinstance(val, str):
                    z, errz = row[val], 0
                else:
                    z, errz = row[val[0]], (row[val[1]], row[val[2]])
            except KeyError:
                pass
            else:
                if 0 <= z < 50:
                    src.add_z(name, z, errz=errz)

        # Add magnitudes
        for name, val in kwargs.get('mags', {}).items():
            try:
                if isinstance(val, str):
                    mag, magerr = row[val], 0
                else:
                    mag, magerr = row[val[0]], row[val[1]]
            except KeyError:
                pass
            else:
                if 0 <= mag < 50:
                    src.add_mag(name, mag, magerr)

    def skycoord(self):
        """Return an `astropy.coordinates.SkyCoord` object."""
        from astropy.coordinates import SkyCoord
        columns = [self.idname, self.raname, self.decname]
        tab = self.select(columns=columns).as_table()
        return SkyCoord(ra=tab[self.raname], dec=tab[self.decname],
                        unit=(u.deg, u.deg), frame='fk5')


class Catalog(BaseCatalog):
    """Handle user catalogs."""

    catalog_type = 'user'

    def __init__(self, name, db, idname='ID', raname='RA', decname='DEC',
                 segmap=None, workdir=None):
        super().__init__(name, db, idname=idname, raname=raname,
                         decname=decname, segmap=segmap)
        self._segmap_aligned = {}
        # Work dir for intermediate files
        if workdir is None:
            raise Exception('FIXME: find a better way to handle this')
        self.workdir = Path(workdir) / self.name
        self.workdir.mkdir(exist_ok=True, parents=True)

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
    def from_parent_cat(cls, parent_cat, name, workdir, whereclause):
        """Create a new `Catalog` from another one."""
        cat = cls(name, parent_cat.db, workdir=workdir,
                  idname=parent_cat.idname, raname=parent_cat.raname,
                  decname=parent_cat.decname, segmap=parent_cat.segmap)
        if parent_cat.meta.get('query'):
            # Combine query string with the parent cat's one
            whereclause = f"{parent_cat.meta['query']} AND {whereclause}"
        cat.update_meta(parent_cat=parent_cat.name, segmap=parent_cat.segmap,
                        maxid=parent_cat.meta['maxid'], query=whereclause)
        return cat

    @lazyproperty
    def segmap_img(self):
        """The segmentation map as an `musex.Segmap` object."""
        if self.segmap:
            return SegMap(self.segmap)

    def get_segmap_aligned(self, dataset):
        """Get a segmap image rotated and aligned to a dataset."""
        name = dataset.name
        if name not in self._segmap_aligned:
            self._segmap_aligned[name] = self.segmap_img.align_with_image(
                dataset.white, truncate=True)
        return self._segmap_aligned[name]

    @lazyproperty
    def maskobj_name(self):
        """Return the path of the source mask files."""
        name = 'mask-source-{:05d}.fits'  # add setting ?
        dataset = self.meta['dataset']
        assert dataset is not None
        return str(self.workdir / dataset / name)

    @lazyproperty
    def masksky_name(self):
        """Return the path of the sky mask file."""
        dataset = self.meta['dataset']
        assert dataset is not None
        return str(self.workdir / dataset / 'mask-sky.fits')

    def attach_dataset(self, dataset, skip_existing=True, convolve_fwhm=0,
                       mask_size=(20, 20), psf_threshold=0.5, n_jobs=1,
                       verbose=0):
        """Attach a dataset to the catalog and generate intermediate products.

        Create masks from the segmap, adapted to a given dataset.

        TODO: store preprocessing status?

        Parameters
        ----------
        dataset: `musex.Dataset`
            The dataset.

        """
        if self.segmap is None:
            self.logger.info('no segmap available, skipping masks creation')
            return

        # create output path if needed
        outpath = self.workdir / dataset.name
        outpath.mkdir(exist_ok=True)

        ref_dataset = self.meta.get('dataset')
        if ref_dataset is not None and dataset.name != ref_dataset:
            raise ValueError('cannot compute masks with a different '
                             'dataset as the one used previously')

        self.update_meta(dataset=dataset.name, convolve_fwhm=convolve_fwhm,
                         mask_size_x=mask_size[0], mask_size_y=mask_size[1],
                         psf_threshold=psf_threshold)

        # Get a segmap image rotated and aligned with our dataset
        segmap = self.get_segmap_aligned(dataset)

        white = dataset.white
        dilateit, struct = _get_psf_convolution_params(convolve_fwhm, segmap,
                                                       psf_threshold)

        # create sky mask
        if exists(self.masksky_name) and skip_existing:
            self.logger.debug('sky mask exists, skipping')
        else:
            self.logger.debug('creating sky mask')
            segmap.get_mask(0, inverse=True, dilate=dilateit, struct=struct,
                            regrid_to=white, outname=self.masksky_name)

        # check sources inside dataset, excluding inactive sources
        if 'active' in self.c:
            tab = self.select(self.c.active.isnot(False)).as_table()
        else:
            tab = self.select().as_table()

        idname = self.idname
        ntot = len(tab)
        tab = tab.select(white.wcs, ra=self.raname, dec=self.decname)
        self.logger.info('%d sources inside dataset (%d in catalog)',
                         len(tab), ntot)

        # extract source masks
        minsize = min(*mask_size) // 2
        to_compute = []
        stats = defaultdict(list)
        for row in tab:
            id_ = int(row[idname])  # need int, not np.int64
            source_path = self.maskobj_name.format(id_)
            if exists(source_path) and skip_existing:
                stats['skipped'].append(id_)
            else:
                center = (row[self.decname], row[self.raname])
                if 'merged' in row.colnames and row['merged']:
                    ids = [o[idname] for o in self.select(
                        self.c.merged_in == id_, columns=[idname])]
                    # debug('merged sources, using ids %s for the mask', ids)
                else:
                    ids = id_

                stats['computed'].append(ids)
                to_compute.append(delayed(segmap.get_source_mask)(
                    ids, center, mask_size, minsize=minsize, struct=struct,
                    dilate=dilateit, outname=source_path, regrid_to=white))

        # FIXME: check which value to use for max_nbytes
        if to_compute:
            Parallel(n_jobs=n_jobs, verbose=verbose)(progressbar(to_compute))

        for row in tab:
            # update in db
            id_ = int(row[idname])  # need int, not np.int64
            source_path = self.maskobj_name.format(id_)
            self.table.upsert(
                {idname: id_,
                 'mask_obj': relpath(source_path, self.workdir),
                 'mask_sky': relpath(self.masksky_name, self.workdir)},
                [idname])

        for key, val in stats.items():
            self.logger.info('%s %d sources: %r', key, len(val),
                             np.array(val, dtype=object))

    def add_segmap_to_source(self, src, conf, dataset):
        """Add the segmap extension to the source object."""
        segm = self.get_segmap_aligned(dataset)
        tag = f'{conf["prefix"]}_SEGMAP'
        src.SEGMAP = tag
        src.add_image(segm.img, tag, rotate=True, order=0)

    def merge_sources(self, idlist, id_=None, dataset=None):
        """Merge sources into one.

        A new source is created, with the "merged" column set to True. The new
        id is stored in the "merged_in" column for the input sources.

        Parameters
        ----------
        idlist: list
            List of ids to merge.
        id_: int
            The new ID for the merged source. If not given, it is automatically
            determined from the maxid and autoincremented.
        dataset: `musex.Dataset`
            The associated dataset. To compute the masks this dataset must be
            given, and must be the same as the one used for `attach_dataset`.

        """
        # compute minimum id for "custom" sources
        # TODO: add a setting for this ("customid_start") ?
        maxid = self.max(self.idname)
        cat_maxid = self.meta['maxid']

        sources = self.select_ids(idlist)
        if len(sources) == 0:
            raise ValueError('no sources found')

        coords = np.array([(s[self.decname], s[self.raname]) for s in sources])
        # TODO: use better weights (flux)
        weights = np.ones(coords.shape[0])
        dec, ra = (np.sum(coords * weights[:, np.newaxis], axis=0) /
                   weights.sum())

        # version
        versions = set(s.get('version') for s in sources)
        if len(versions) == 1:
            version = versions.pop()
        else:
            self.logger.warning('sources have different version')
            version = None

        row = {'merged': True, self.raname: ra, self.decname: dec,
               'version': version}

        if id_ is not None:
            row[self.idname] = id_
        elif maxid <= cat_maxid:
            row[self.idname] = 10**(len(str(cat_maxid)))

        # Create missing columns
        self.table._sync_columns({'active': False, 'merged_in': 0, **row},
                                 True)

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
                tbl.upsert({idname: s[idname], 'active': False,
                            'merged_in': newid}, [idname], ensure=False)

        if self.idmap:
            self.idmap.add_ids(newid, self.name)

        self.logger.info('sources %s have been merged in %s', idlist, newid)

        if dataset is None:
            # Just return in this case
            # self.logger.debug('cannot compute mask (missing dataset)')
            return newid

        if dataset.name != self.meta['dataset']:
            self.logger.warning('cannot compute masks with a different '
                                'dataset as the one used previously')
            return newid

        try:
            # maskobj
            maskobj = self.maskobj_name.format(newid)
            segmap = self.get_segmap_aligned(dataset)
            dilateit, struct = _get_psf_convolution_params(
                self.meta['convolve_fwhm'], segmap, self.meta['psf_threshold'])

            mask_size = (self.meta['mask_size_x'], self.meta['mask_size_y'])
            minsize = min(*mask_size) // 2
            segmap.get_source_mask(
                idlist, (dec, ra), mask_size, minsize=minsize, struct=struct,
                dilate=dilateit, outname=maskobj, regrid_to=dataset.white)

            # masksky
            # FIXME: currenlty we suppose that mask_sky is always the same
            masksky = relpath(self.masksky_name, self.workdir)
            if any(s['mask_sky'] != masksky for s in sources):
                self.logger.warning('cannot reuse mask_sky')
                masksky = None
        except Exception:
            self.logger.error('unexpected error while computing the masks',
                              exc_info=True)
        else:
            # update in db
            self.table.upsert({self.idname: newid, 'mask_sky': masksky,
                               'mask_obj': relpath(maskobj, self.workdir)},
                              [self.idname])

        return newid

    def get_ids_merged_in(self, id_):
        """Return the IDs that were merged in ``id_``."""
        return [o[self.idname] for o in self.table.find(merged_in=id_)]


class InputCatalog(BaseCatalog):
    """Handles catalogs imported from an exiting file."""

    catalog_type = 'input'

    @classmethod
    def from_settings(cls, name, db, **kwargs):
        """Create an InputCatalog from the settings file."""
        init_keys = ('idname', 'raname', 'decname', 'segmap')
        kw = {k: v for k, v in kwargs.items() if k in init_keys}
        cat = cls(name, db, **kw)
        for key in ('catalog', 'version', 'extract'):
            if kwargs.get(key) is None:
                raise ValueError(f'an input {key} is required')
            setattr(cat, key, kwargs[key])
        return cat

    def ingest_input_catalog(self, catalog=None, limit=None, upsert=False,
                             keys=None, show_progress=True):
        """Ingest an input catalog.

        The catalog to ingest can be given with the ``catalog`` argument or
        in the settings file.  Existing records can be updated using
        ``upsert=True`` and ``keys``.

        Parameters
        ----------
        catalog: str or `astropy.table.Table`
            Table to insert.
        limit: int
            To limit the number of rows.
        upsert: bool
            If True, existing rows with the same values for ``keys`` are
            updated.
        keys: list of str
            If rows with matching keys exist they will be updated, otherwise
            a new row is inserted in the table. Defaults to
            ``[idname, 'version']``.
        show_progress: bool
            Show a progress bar.

        """
        catalog = catalog or self.catalog
        if isinstance(catalog, str):
            self.logger.info('ingesting catalog %s', catalog)
            catalog = Table.read(catalog)
        else:
            self.logger.info('ingesting catalog')

        if limit:
            self.logger.info('keeping only %d rows', limit)
            catalog = catalog[:limit]

        if upsert:
            self.upsert(catalog, version=self.version, keys=keys,
                        show_progress=show_progress)
        else:
            self.insert(catalog, version=self.version,
                        show_progress=show_progress)

        self.update_meta(creation_date=datetime.utcnow().isoformat(),
                         type=self.catalog_type, maxid=self.max(self.idname))


class MarzCatalog(InputCatalog):
    """Handles catalogs imported from MarZ."""

    catalog_type = 'marz'


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
