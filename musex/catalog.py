import importlib
import logging
import numpy as np
import os
import textwrap
from datetime import datetime

import astropy.units as u
from astropy.table import Table, Column
from astropy.utils.console import ProgressBar
from astropy.utils.decorators import lazyproperty

from collections import OrderedDict
from collections.abc import Sequence
from mpdaf.sdetect import Catalog as _Catalog
from os.path import exists, relpath
from pathlib import Path
from sqlalchemy.sql import select, func

from .segmap import SegMap
from .settings import isnotebook
from .utils import struct_from_moffat_fwhm, isiter

DIRNAME = os.path.abspath(os.path.dirname(__file__))

__all__ = ('load_input_catalogs', 'table_to_odict', 'ResultSet', 'Catalog',
           'InputCatalog')


def load_input_catalogs(settings, db):
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
    """Convert a `astropy.table.Table` to a list of OrderedDict."""
    colnames = table.colnames
    return [OrderedDict(zip(colnames, row))
            for row in zip(*[c.tolist() for c in table.columns.values()])]


def _get_psf_convolution_params(convolve_fwhm, segmap, psf_threshold):
    if convolve_fwhm:
        # compute a structuring element for the dilatation, to simulate
        # a convolution with a psf, but faster.
        dilateit = 1
        logging.getLogger(__name__).debug(
            'dilate with %d iterations, psf=%.2f', dilateit, convolve_fwhm)
        struct = struct_from_moffat_fwhm(segmap.img.wcs, convolve_fwhm,
                                         psf_threshold=psf_threshold)
    else:
        dilateit = 0
        struct = None
    return dilateit, struct


class ResultSet(Sequence):
    """Contains the result of a query on the database."""

    def __init__(self, results, whereclause=None, catalog=None):
        self.results = list(results)
        self.whereclause = whereclause
        # TODO: use weakref here ?
        self.catalog = catalog

    def __repr__(self):
        out = f'<{self.__class__.__name__}('
        if self.whereclause is not None:
            query = self.whereclause.compile(
                compile_kwargs={"literal_binds": True})
            out += f'{query}'
        out += f')>, {len(self)} results'
        return out

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]

    def as_table(self, mpdaf_catalog=True):
        cls = _Catalog if mpdaf_catalog else Table
        tbl = cls(data=self.results, names=self.results[0].keys())
        tbl.whereclause = self.whereclause
        tbl.catalog = self.catalog
        return tbl


class BaseCatalog:
    """Handle Catalogs by the way of a database table.

    Parameters
    ----------
    name: str
        Name of the catalog.
    db: dataset.Database
        The database object.
    idname: str
        Name of the 'id' column.
    raname: str
        Name of the 'ra' column.
    decname: str
        Name of the 'dec' column.
    segmap: str
        Path to the segmentation map file associated with the catalog.

    """

    def __init__(self, name, db, idname='ID', raname='RA', decname='DEC',
                 segmap=None):
        self.name = name
        self.db = db
        self.segmap = segmap
        self.idname = idname
        self.raname = raname
        self.decname = decname
        self.logger = logging.getLogger(__name__)

        # if self.name not in self.db:
        #     self.logger.debug('create table %s (primary key: %s)',
        #                       self.name, self.idname)

        # Get the reference to the db table, which is created if needed
        self.table = self.db.create_table(self.name, primary_id=self.idname)

        # Insert default meta about the table if it doesn't exist yet
        if self.meta is None:
            self.update_meta(creation_date=datetime.utcnow().isoformat(),
                             type=None, parent_cat=None, idname=self.idname,
                             raname=self.raname, decname=self.decname,
                             segmap=self.segmap)
            self.db['catalogs'].create_column('maxid', self.db.types.integer)

    def __len__(self):
        return self.table.count()

    def __repr__(self):
        return f"<{self.__class__.__name__}('{self.name}', {len(self)} rows)>"

    def max(self, colname):
        return self.db.executable.execute(func.max(self.c[colname])).scalar()

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
        columns = '\n'.join(f'- {k:{maxlen}s}: {v.type}'
                            for k, v in self.table.table.columns.items())
        print(f"Columns:\n{columns}\n")

    def insert_rows(self, rows, version=None, show_progress=True):
        """Insert rows in the catalog.

        Parameters
        ----------
        rows: list of dict
            List of rows to insert. Each row must be a dict with column names
            as keys.
        version: str
            Version added to each row (if not available in the row values).
        show_progress: bool
            Show a progress bar.

        """
        count_inserted = 0
        count_updated = 0
        if show_progress:
            rows = ProgressBar(rows, ipython_widget=isnotebook())
        with self.db as tx:
            tbl = tx[self.name]
            for row in rows:
                row.setdefault('version', version)

                res = tbl.upsert(row, [self.idname, 'version'])
                if res is True:
                    count_updated += 1
                else:
                    count_inserted += 1

        if not tbl.has_index(self.idname):
            tbl.create_index(self.idname)
        self.logger.info('%d rows inserted, %s updated', count_inserted,
                         count_updated)

    def insert_table(self, table, version=None, show_progress=True):
        """Insert rows from an Astropy Table in the catalog.

        Parameters
        ----------
        table: `astropy.table.Table`
            Table to insert.
        version: str
            Version added to each row (if not available in the table).
        show_progress: bool
            Show a progress bar.

        """
        if not isinstance(table, Table):
            raise ValueError('table should be an Astropy Table object')
        self.insert_rows(table_to_odict(table), version=version,
                         show_progress=show_progress)

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
            Additional parameters are passed to `dataset.Database.query`.

        """
        if columns is not None:
            columns = [self.c[col] for col in columns]
        else:
            columns = [self.table.table]
        query = self.db.query(select(columns=columns, whereclause=whereclause,
                                     **params))
        res = ResultSet(query, whereclause=whereclause, catalog=self)

        if wcs is not None:
            t = res.as_table()
            t = t.select(wcs, ra=self.raname, dec=self.decname, margin=margin)
            res = ResultSet(table_to_odict(t), whereclause=whereclause,
                            catalog=self)

        return res

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

        whereclause = self.c[self.idname].in_(idlist)
        return self.select(whereclause=whereclause, columns=columns, **params)

    def add_to_source(self, src, conf):
        """Add information to the Source object.

        FIXME: see how to improve conf here.

        """
        # Add catalog as a BinTableHDU
        cat = self.select(columns=conf['columns']).as_table()
        wcs = src.images[conf.get('select_in', 'WHITE')].wcs
        scat = cat.select(wcs, ra=self.raname, dec=self.decname,
                          margin=conf['margin'])
        dist = scat.edgedist(wcs, ra=self.raname, dec=self.decname)
        scat.add_column(Column(name='DIST', data=dist))
        # FIXME: is it the same ?
        # cat = in_catalog(cat, src.images['HST_F775W_E'], quiet=True)
        name = conf.get('name', 'CAT')
        self.logger.debug('Adding catalog %s (%d rows)', name, len(scat))
        src.add_table(scat, f'{conf["prefix"]}_{name}')

    def skycoord(self):
        """Return an `astropy.coordinates.SkyCoord` object."""
        from astropy.coordinates import SkyCoord
        columns = [self.idname, self.raname, self.decname]
        tab = self.select(columns=columns).as_table()
        return SkyCoord(ra=tab[self.raname], dec=tab[self.decname],
                        unit=(u.deg, u.deg), frame='fk5')


class Catalog(BaseCatalog):

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
        name = 'mask-source-{:05d}.fits'  # add setting ?
        dataset = self.meta['dataset']
        assert dataset is not None
        return str(self.workdir / dataset / name)

    @lazyproperty
    def masksky_name(self):
        dataset = self.meta['dataset']
        assert dataset is not None
        return str(self.workdir / dataset / 'mask-sky.fits')

    def attach_dataset(self, dataset, skip_existing=True, convolve_fwhm=0,
                       mask_size=(20, 20), psf_threshold=0.5):
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
        debug = self.logger.debug
        if exists(self.masksky_name) and skip_existing:
            debug('sky mask exists, skipping')
        else:
            debug('creating sky mask')
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
        self.logger.info('%d/%d sources inside dataset', len(tab), ntot)

        # extract source masks
        minsize = min(*mask_size) // 2
        for row in ProgressBar(tab, ipython_widget=isnotebook()):
            id_ = int(row[idname])  # need int, not np.int64
            source_path = self.maskobj_name.format(id_)
            if exists(source_path) and skip_existing:
                debug('source %05d exists, skipping', id_)
            else:
                center = (row[self.decname], row[self.raname])
                if 'merged' in row.colnames and row['merged']:
                    ids = [o[idname] for o in self.select(
                        self.c.merged_in == id_, columns=[idname])]
                    debug('merged sources, using ids %s for the mask', ids)
                else:
                    ids = id_
                segmap.get_source_mask(
                    ids, center, mask_size, minsize=minsize, struct=struct,
                    dilate=dilateit, outname=source_path, regrid_to=white)

            # update in db
            self.table.upsert(
                {idname: id_,
                 'mask_obj': relpath(source_path, self.workdir),
                 'mask_sky': relpath(self.masksky_name, self.workdir)},
                [idname])

    def add_segmap_to_source(self, src, conf, dataset):
        segm = self.get_segmap_aligned(dataset)
        src.add_image(segm.img, f'{conf["prefix"]}_SEGMAP', rotate=True,
                      order=0)

    def merge_sources(self, idlist, dataset=None):
        """Merge sources into one.

        A new source is created, with the "merged" column set to True. The new
        id is stored in the "merged_in" column for the input sources.

        Parameters
        ----------
        idlist: list
            List of ids to merge.
        dataset: `musex.Dataset`
            The associated dataset. To compute the masks this dataset must be
            given, and must be the same as the one used for `attach_dataset`.

        """
        # compute minimum id for "custom" sources
        # TODO: add a setting for this ("customid_start") ?
        maxid = self.max(self.idname)
        if self.meta['parent_cat']:
            cat_maxid = self.db['catalogs'].find_one(
                name=self.meta['parent_cat'])['maxid']
        else:
            cat_maxid = self.meta['maxid']

        sources = self.select_ids(idlist)
        coords = np.array([(s[self.decname], s[self.raname]) for s in sources])
        # TODO: use better weights (flux)
        weights = np.ones(coords.shape[0])
        dec, ra = (np.sum(coords * weights[:, np.newaxis], axis=0) /
                   weights.sum())

        # version
        versions = set(s['version'] for s in sources)
        if len(versions) == 1:
            version = versions.pop()
        else:
            self.logger.warning('sources have different version')
            version = None

        row = {'merged': True, self.raname: ra, self.decname: dec,
               'version': version}
        if maxid <= cat_maxid:
            row[self.idname] = 10**(len(str(cat_maxid)))

        with self.db as tx:
            tbl = tx[self.name]
            # create new (merged) source
            newid = tbl.insert(row)
            # deactivate the other sources
            idname = self.idname
            for s in sources:
                if s.get('merged_in') is not None:
                    raise ValueError(f'source {s[idname]} is already merged')
                tbl.upsert({idname: s[idname], 'active': False,
                            'merged_in': newid}, [idname])

        self.logger.info('sources %s have been merged in %s', idlist, newid)

        if dataset is None:
            # Just return in this case
            self.logger.debug('cannot compute mask (missing dataset)')
            return

        if dataset.name != self.meta['dataset']:
            self.logger.warning('cannot compute masks with a different '
                                'dataset as the one used previously')
            return

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


class InputCatalog(BaseCatalog):
    """Handles catalogs imported from an exiting file."""

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

    def ingest_input_catalog(self, limit=None, show_progress=True):
        """Ingest the source catalog (given in the settings file). Existing
        records are updated."""
        self.logger.info('ingesting catalog %s', self.catalog)
        cat = Table.read(self.catalog)
        if limit:
            self.logger.info('keeping only %d rows', limit)
            cat = cat[:limit]
        self.insert_table(cat, version=self.version,
                          show_progress=show_progress)
        self.update_meta(creation_date=datetime.utcnow().isoformat(),
                         type='input', maxid=self.max(self.idname))
