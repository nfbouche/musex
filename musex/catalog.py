import importlib
import logging
import numpy as np
import os
import textwrap

import astropy.units as u
from astropy.table import Table, Column
from astropy.utils.console import ProgressBar
from astropy.utils.decorators import lazyproperty

from collections import OrderedDict
from collections.abc import Sequence
from mpdaf.sdetect import Catalog as _Catalog
from os.path import exists, relpath
from pathlib import Path
from sqlalchemy.sql import select

from .hstutils import align_with_image
from .segmap import SegMap
from .settings import isnotebook

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
        catalogs[name] = cls.from_settings(name, db,
                                           workdir=settings['workdir'], **conf)
    return catalogs


class ResultSet(Sequence):

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
        t = cls(data=self.results, names=self.results[0].keys())
        if '_id' in t.columns:
            t.remove_column('_id')
        return t


class BaseCatalog:
    """Handle Catalogs by the way of a database table.
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
            self.logger.info('create table %s (primary key: %s)',
                             self.name, self.idname)
        self.table = self.db.create_table(self.name, primary_id=self.idname)

    def __len__(self):
        return self.table.count()

    def __repr__(self):
        return f"<{self.__class__.__name__}('{self.name}', {len(self)} rows)>"

    @property
    def meta(self):
        return self.db['catalogs'].find_one(name=self.name)

    def info(self):
        print(textwrap.dedent(f"""\
        {self.__class__.__name__} '{self.name}' - {len(self)} rows.

        Workdir: {self.workdir}
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
        count_inserted = 0
        count_updated = 0
        if show_progress:
            rows = ProgressBar(rows, ipython_widget=isnotebook())
        with self.db as tx:
            tbl = tx[self.name]
            for row in rows:
                if 'version' not in row:
                    if version is not None:
                        row['version'] = version
                    else:
                        raise ValueError('version should be specified')

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
        if not isinstance(table, Table):
            raise ValueError('table should be an Astropy Table object')

        rows = [OrderedDict(zip(table.colnames, row))
                for row in zip(*[c.tolist() for c in table.columns.values()])]
        self.insert_rows(rows, version=version, show_progress=show_progress)

    @property
    def c(self):
        """The list of columns from the SQLAlchemy table object."""
        return self.table.table.c

    def select(self, whereclause=None, columns=None, **params):
        """Select rows in the catalog.

        Parameters
        ----------
        whereclause:
            The SQLAlchemy selection clause.
        columns: list of str
            List of columns to retrieve (all columns if None).

        """
        if columns is not None:
            columns = [self.c[col] for col in columns]
        else:
            columns = [self.table.table]
        query = self.db.query(select(columns=columns, whereclause=whereclause,
                                     **params))
        return ResultSet(query, whereclause=whereclause, catalog=self)

    def add_to_source(self, src, conf):
        """Add information to the Source object."""

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
        # Work dir for intermediate files
        if workdir is None:
            raise Exception('FIXME: find a better way to handle this')
        self.workdir = Path(workdir) / self.name
        self.workdir.mkdir(exist_ok=True)

    @lazyproperty
    def segmap_img(self):
        """The segmentation map."""
        if self.segmap:
            return SegMap(self.segmap)

    def attach_dataset(self, dataset, skip_existing=True,
                       mask_convolve_fsf=0, mask_size=(20, 20)):
        """Attach a dataset to the catalog and generate intermediate products.

        Create masks from the segmap, adapted to a given dataset.

        TODO: store in the database for each source: the sky and source mask
        paths, and preprocessing status

        """
        if self.segmap is None:
            self.logger.info('No segmap available, skipping masks creation')
            return

        debug = self.logger.debug
        mask_name = 'mask-source-{:05d}.fits'  # add setting ?

        # create output path if needed
        outpath = self.workdir / dataset.name
        outpath.mkdir(exist_ok=True)

        self.db['catalogs'].upsert(dict(name=self.name, dataset=dataset.name),
                                   ['name'])

        # create sky mask
        sky_path = str(self.workdir / dataset.name / 'mask-sky.fits')

        if exists(sky_path) and skip_existing:
            debug('sky mask exists, skipping')
        else:
            debug('creating sky mask')
            sky = self.segmap_img.get_mask(0)
            sky._data = np.logical_not(sky._data).astype(float)
            align_with_image(sky, dataset.white, order=0, inplace=True,
                             fsf_conv=mask_convolve_fsf)
            sky.data /= np.max(sky.data)
            sky._data = np.where(sky._data > 0.1, 0, 1)
            sky.write(sky_path, savemask='none')

        # check sources inside dataset
        columns = [self.idname, self.raname, self.decname]
        tab = self.select(columns=columns).as_table()
        ntot = len(tab)
        tab = tab.select(dataset.white.wcs, ra=self.raname, dec=self.decname)
        self.logger.info('%d sources inside dataset out of %d', len(tab), ntot)
        usize = u.arcsec
        ucent = u.deg
        minsize = min(*mask_size) // 2

        # TODO: prepare the psf image for convolution
        # ima = gauss_image(self.shape, wcs=self.wcs, fwhm=fwhm, gauss=None,
        #                   unit_fwhm=usize, cont=0, unit=self.unit)
        # ima.norm(typ='sum')

        # extract source masks
        white = dataset.white
        get_mask = self.segmap_img.get_source_mask
        source_mask_path = str(self.workdir / dataset.name / mask_name)
        for id_, ra, dec in ProgressBar(tab, ipython_widget=isnotebook()):
            id_ = int(id_)  # need int, not np.int64
            source_path = source_mask_path.format(id_)
            if exists(source_path) and skip_existing:
                debug('source %05d exists, skipping', id_)
            else:
                debug('source %05d (%.5f, %.5f), extract mask', id_, ra, dec)
                mask = get_mask(id_, (dec, ra), mask_size, minsize=minsize,
                                unit_center=ucent, unit_size=usize)
                subref = white.subimage((dec, ra), mask_size, minsize=minsize,
                                        unit_center=ucent, unit_size=usize)
                align_with_image(mask, subref, order=0, inplace=True,
                                 fsf_conv=mask_convolve_fsf)
                data = mask.data.filled(0)
                mask._data = np.where(data / data.max() > 0.1, 1, 0)
                mask.write(source_path, savemask='none')

            # update in db
            self.table.upsert({self.idname: id_,
                               'mask_obj': relpath(source_path, self.workdir),
                               'mask_sky': relpath(sky_path, self.workdir)},
                              [self.idname])


class InputCatalog(Catalog):
    """Handles catalogs imported from an exiting file."""

    @classmethod
    def from_settings(cls, name, db, **kwargs):
        init_keys = ('idname', 'raname', 'decname', 'workdir', 'segmap')
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
