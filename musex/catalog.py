import importlib
import logging
import numpy as np
import os
import textwrap
from datetime import datetime
from functools import partial

import astropy.units as u
from astropy.table import Table, Column
from astropy.utils.console import ProgressBar
from astropy.utils.decorators import lazyproperty

from collections import OrderedDict
from collections.abc import Sequence
from mpdaf.obj import moffat_image
from mpdaf.sdetect import Catalog as _Catalog
from os.path import exists, relpath
from pathlib import Path
from sqlalchemy.sql import select

from .segmap import SegMap
from .settings import isnotebook
from .utils import regrid_to_image

DIRNAME = os.path.abspath(os.path.dirname(__file__))

__all__ = ['load_input_catalogs', 'Catalog']


#     def merge_close_sources(self, maxdist=0.2*u.arcsec):
#         from astropy.coordinates import SkyCoord
#         columns = [self.idname, self.raname, self.decname]
#         tab = self.select(columns=columns).as_table()
#         coords = SkyCoord(ra=tab[self.raname], dec=tab[self.decname],
#                           unit=(u.deg, u.deg), frame='fk5')
#         dist = coords.separation(coords[:, None])
#         ind = np.where(np.sum(dist < maxdist, axis=0) > 1)
#         # FIXME: find how to merge close sources ...


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

        if self.name not in self.db:
            self.logger.debug('create table %s (primary key: %s)',
                              self.name, self.idname)
        self.table = self.db.create_table(self.name, primary_id=self.idname)

    def __len__(self):
        return self.table.count()

    def __repr__(self):
        return f"<{self.__class__.__name__}('{self.name}', {len(self)} rows)>"

    def max(self, colname):
        res = next(self.db.query(f'SELECT max({colname}) FROM {self.name}'))
        return res[f'max({colname})']

    @property
    def c(self):
        """The list of columns from the SQLAlchemy table object."""
        return self.table.table.c

    @property
    def meta(self):
        """Return metadata associated with the catalog."""
        return self.db['catalogs'].find_one(name=self.name)

    def update_meta(self, **kwargs):
        """Update metadata associated with the catalog."""
        self.db['catalogs'].upsert({'name': self.name, **kwargs}, ['name'])

    def info(self):
        """Print information about the catalog (columns etc.)."""
        print(textwrap.dedent(f"""\
        {self.__class__.__name__} '{self.name}' - {len(self)} rows.

        Workdir: {getattr(self, 'workdir', '')}
        Segmap : {self.segmap}
        """))

        meta = self.meta
        if meta:
            maxlen = max(len(k) for k in meta.keys()) + 1
            meta = '\n'.join(f'- {k:{maxlen}s}: {v}' for k, v in meta.items()
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
        if not isinstance(idlist, (list, tuple)):
            idlist = [idlist]
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

    def attach_dataset(self, dataset, skip_existing=True, convolve_fwhm=0,
                       mask_size=(20, 20), psf_threshold=0.5):
        """Attach a dataset to the catalog and generate intermediate products.

        Create masks from the segmap, adapted to a given dataset.

        TODO: store preprocessing status?

        """
        if self.segmap is None:
            self.logger.info('No segmap available, skipping masks creation')
            return

        debug = self.logger.debug
        mask_name = 'mask-source-{:05d}.fits'  # add setting ?

        # create output path if needed
        outpath = self.workdir / dataset.name
        outpath.mkdir(exist_ok=True)

        self.update_meta(dataset=dataset.name)

        # Get a segmap image rotated and aligned with our dataset
        segmap = self.get_segmap_aligned(dataset)

        if convolve_fwhm:
            # compute a structuring element for the dilatation, to simulate
            # a convolution with a psf.
            dilateit = 1
            size = round(convolve_fwhm / segmap.img.get_step(u.arcsec)[0]) + 1
            if size % 2 == 0:
                size += 1
            debug('dilate with %d iterations, psf = %d pixels', dilateit, size)
            psf = moffat_image(fwhm=(convolve_fwhm, convolve_fwhm), n=2.5,
                               peak=True, wcs=segmap.img.wcs[:51, :51])
            struct = psf._data > psf_threshold
        else:
            dilateit = 0
            struct = None

        # create sky mask
        sky_path = str(self.workdir / dataset.name / 'mask-sky.fits')
        if exists(sky_path) and skip_existing:
            debug('sky mask exists, skipping')
        else:
            debug('creating sky mask')
            sky = segmap.get_mask(0, inverse=True, dilate=dilateit,
                                  dtype=float, struct=struct)
            sky = regrid_to_image(sky, dataset.white, inplace=True, order=0,
                                  antialias=False)
            sky._data = (~(np.around(sky._data).astype(bool))).astype(np.uint8)
            sky.write(sky_path, savemask='none')

        # check sources inside dataset
        wcsref = dataset.white.wcs
        columns = [self.idname, self.raname, self.decname]
        tab = self.select(columns=columns).as_table()
        ntot = len(tab)
        tab = tab.select(wcsref, ra=self.raname, dec=self.decname)
        self.logger.info('%d/%d sources inside dataset', len(tab), ntot)

        # extract source masks
        usize = u.arcsec
        udeg = u.deg
        minsize = min(*mask_size) // 2
        inc = wcsref.get_axis_increments(unit=usize)
        newdim = mask_size / wcsref.get_step(unit=usize)
        get_mask = partial(segmap.get_source_mask, minsize=minsize,
                           dtype=float, dilate=dilateit, struct=struct,
                           unit_center=udeg, unit_size=usize)
        source_mask_path = str(self.workdir / dataset.name / mask_name)
        for id_, ra, dec in ProgressBar(tab, ipython_widget=isnotebook()):
            id_ = int(id_)  # need int, not np.int64
            source_path = source_mask_path.format(id_)
            if exists(source_path) and skip_existing:
                debug('source %05d exists, skipping', id_)
            else:
                center = (dec, ra)
                mask = get_mask(id_, center, mask_size)
                refpos = mask.wcs.pix2sky([0, 0])[0]
                mask.regrid(newdim, refpos, [0, 0], inc, order=0,
                            unit_inc=usize, inplace=True, antialias=False)
                mask._data = np.around(mask._data).astype(np.uint8)
                debug('source %05d (%.5f, %.5f), extract mask (%d masked '
                      'pixels)', id_, ra, dec, np.count_nonzero(mask._data))
                mask.write(source_path, savemask='none')

            # update in db
            self.table.upsert({self.idname: id_,
                               'mask_obj': relpath(source_path, self.workdir),
                               'mask_sky': relpath(sky_path, self.workdir)},
                              [self.idname])

    def add_segmap_to_source(self, src, conf, dataset):
        segm = self.get_segmap_aligned(dataset)
        src.add_image(segm.img, f'{conf["prefix"]}_SEGMAP', rotate=True,
                      order=0)


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
                         type='input', parent_cat=None, idname=self.idname,
                         raname=self.raname, decname=self.decname,
                         segmap=self.segmap, maxid=self.max(self.idname))
