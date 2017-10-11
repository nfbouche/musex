import importlib
import logging
import numpy as np
import os

import astropy.units as u
from astropy.table import Table, Column
from astropy.utils.console import ProgressBar
from astropy.utils.decorators import lazyproperty

from collections import OrderedDict
from collections.abc import Sequence
from mpdaf.obj import Image
from mpdaf.sdetect import Catalog as _Catalog
from pathlib import Path
from sqlalchemy.sql import select

from .segmap import SegMap
from .settings import isnotebook

DIRNAME = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)

__all__ = ['load_catalogs', 'Catalog', 'PriorCatalog']


def load_catalogs(settings, db):
    catalogs = {}
    for name, conf in settings['catalogs'].items():
        mod, class_ = conf['class'].rsplit('.', 1)
        mod = importlib.import_module(mod)
        catalogs[name] = getattr(mod, class_)(name, conf, db)
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
            out += f'{self.whereclause}, {self.whereclause.compile().params}'
        out += f')>, {len(self)} results'
        return out

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]

    def as_table(self):
        t = _Catalog(data=self.results, names=self.results[0].keys())
        if '_id' in t.columns:
            t.remove_column('_id')
        return t


class Catalog:

    def __init__(self, name, settings, db):
        self.name = name
        self.settings = settings
        self.db = db
        self.logger = logging.getLogger(__name__)
        for key in ('catalog', 'colnames', 'version'):
            setattr(self, key, self.settings.get(key))
        for key, val in self.settings['colnames'].items():
            setattr(self, key, val)

    @lazyproperty
    def table(self):
        return self.db.create_table(self.name, primary_id='_id')

    def ingest_catalog(self, limit=None):
        logger.info('ingesting catalog %s', self.catalog)
        cat = Table.read(self.catalog)
        if limit:
            logger.info('keeping only %d rows', limit)
            cat = cat[:limit]

        # TODO: Use ID as primary key ?
        cat.add_column(Column(name='version', data=[self.version] * len(cat)))

        table = self.table
        colnames = cat.colnames
        count_inserted = 0
        count_updated = 0
        rows = list(zip(*[c.tolist() for c in cat.columns.values()]))
        with self.db as tx:
            table = tx[self.name]
            for row in ProgressBar(rows, ipython_widget=isnotebook()):
                res = table.upsert(OrderedDict(zip(colnames, row)),
                                   [self.idname, 'version'])
                if res is True:
                    count_updated += 1
                else:
                    count_inserted += 1

        table.create_index(self.idname)
        logger.info('%d rows inserted, %s updated', count_inserted,
                    count_updated)

    @property
    def c(self):
        return self.table.table.c

    def select(self, whereclause=None, columns=None, **params):
        if columns is not None:
            columns = [self.c[col] for col in columns]
        else:
            columns = [self.table.table]
        query = self.db.query(select(columns=columns, whereclause=whereclause,
                                     **params))
        return ResultSet(query, whereclause=whereclause, catalog=self)


class PriorCatalog(Catalog):

    @lazyproperty
    def segmap(self):
        return SegMap(self.settings['segmap'])

    def preprocess_segmap(self, dataset, skip=True):
        """Create masks from the segmap, adapted to a given dataset."""

        outpath = Path(self.settings['masks']['outpath']) / dataset.name
        outpath.mkdir(exist_ok=True)

        # sky mask
        sky_path = outpath / 'sky.fits'
        if sky_path.exists() and skip:
            self.logger.debug('sky mask exists, skipping')
            sky = None
        else:
            self.logger.debug('creating sky mask')
            sky = self.segmap.get_mask(0)
            sky.align_with_image(dataset.white, inplace=True)
            sky._data = np.where(sky._data > 0.5, 1, 0)
            sky.write(str(sky_path), savemask='none')

        fsf = dataset.meanfsf
        skyconv_path = outpath / 'sky_convolved.fits'
        if skyconv_path.exists() and skip:
            self.logger.debug('convolved sky mask exists, skipping')
        else:
            self.logger.debug('creating convolve sky mask, fsf=%.1f', fsf)
            sky = sky or Image(str(sky_path))
            sky._data = np.logical_not(sky._data).astype(float)
            skyconv = sky.fftconvolve_gauss(fwhm=(fsf, fsf))
            skyconv._data = np.where(skyconv._data > 0.1, 0, 1)
            skyconv.write(str(skyconv_path), savemask='none')

        # extract source masks
        size = self.settings['masks']['size']
        outname = str(outpath / self.settings['masks']['outname'])
        columns = [self.idname, self.raname, self.decname]

        # check sources inside dataset
        tab = self.select(columns=columns).as_table()
        ntot = len(tab)
        tab = tab.select(dataset.white.wcs, ra=self.raname, dec=self.decname)
        self.logger.info('%d sources inside dataset out of %d', len(tab), ntot)

        for id_, ra, dec in ProgressBar(tab, ipython_widget=isnotebook()):
            self.logger.debug('extract mask - source %05d', id_)
            mask = self.segmap.get_source_mask(
                id_, (dec, ra), size, unit_center=u.deg, unit_size=u.arcsec)
            mask.align_with_image(dataset.white, inplace=True)
            mask.crop()
            mask.data /= mask.data.max()
            mask._data = np.where(mask._data > 0.01, 1, 0)
            mask.write(outname.format(id_) + '-test.fits', savemask='none')

            # TODO: create gaussian image only once
            maskconv = mask.fftconvolve_gauss(fwhm=(fsf, fsf))
            maskconv._data = np.where(maskconv._data > 0.1, 0, 1)
            maskconv.write(outname.format(id_), savemask='none')
            break
