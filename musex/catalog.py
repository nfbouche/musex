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
from os.path import exists, join
from pathlib import Path
from sqlalchemy.sql import select

from .hstutils import align_with_image
from .segmap import SegMap
from .settings import isnotebook

DIRNAME = os.path.abspath(os.path.dirname(__file__))

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

    def as_table(self, mpdaf_catalog=True):
        cls = _Catalog if mpdaf_catalog else Table
        t = cls(data=self.results, names=self.results[0].keys())
        if '_id' in t.columns:
            t.remove_column('_id')
        return t


class Catalog:
    """Handle Catalogs by the way of a database table.
    """

    def __init__(self, name, settings, db):
        self.name = name
        self.settings = settings
        self.db = db
        self.logger = logging.getLogger(__name__)
        for key in ('catalog', 'colnames', 'version', 'prefix'):
            setattr(self, key, self.settings.get(key))
        for key, val in self.settings['colnames'].items():
            setattr(self, key, val)

    @lazyproperty
    def table(self):
        """The Dataset Table object.

        https://dataset.readthedocs.io/en/latest/api.html#table
        """
        return self.db.create_table(self.name, primary_id='_id')

    def preprocess(self, dataset, skip=True):
        """Generate intermediate results linked to a given dataset."""
        pass

    def ingest_catalog(self, limit=None):
        """Ingest the source catalog (given in the settings file). Existing
        records are updated.
        """
        self.logger.info('ingesting catalog %s', self.catalog)
        cat = Table.read(self.catalog)
        if limit:
            self.logger.info('keeping only %d rows', limit)
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
        self.logger.info('%d rows inserted, %s updated', count_inserted,
                         count_updated)

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

    def add_to_source(self, src):
        """Add information to the Source object."""
        conf = self.settings['extract']
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
        src.add_table(scat, f'{self.prefix}_{name}')

#     def merge_close_sources(self, maxdist=0.2*u.arcsec):
#         from astropy.coordinates import SkyCoord
#         columns = [self.idname, self.raname, self.decname]
#         tab = self.select(columns=columns).as_table()
#         coords = SkyCoord(ra=tab[self.raname], dec=tab[self.decname],
#                           unit=(u.deg, u.deg), frame='fk5')
#         dist = coords.separation(coords[:, None])
#         ind = np.where(np.sum(dist < maxdist, axis=0) > 1)
#         # FIXME: find how to merge close sources ...


class PriorCatalog(Catalog):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skyim = {}

    @lazyproperty
    def segmap(self):
        """The segmentation map."""
        return SegMap(self.settings['segmap'])

    def get_sky_mask_path(self, dataset):
        return join(self.settings['masks']['outpath'], dataset.name,
                    'sky.fits')

    def get_source_mask_path(self, dataset, source_id):
        return join(self.settings['masks']['outpath'], dataset.name,
                    self.settings['masks']['outname']).format(source_id)

    def get_sky_mask(self, dataset, source_id, size, center=None):
        sky = self._skyim.get(dataset)
        if sky is None:
            sky = self._skyim[dataset] = Image(self.get_sky_mask_path(dataset))
        if center is None:
            s = self.select(self.c[self.idname] == source_id)[0]
            center = (s[self.decname], s[self.raname])
        return sky.subimage(center, size, minsize=size, unit_size=u.arcsec)

    def get_source_mask(self, dataset, source_id, size, center=None):
        src = Image(self.get_source_mask_path(dataset, source_id))
        if center is None:
            s = self.select(self.c[self.idname] == source_id)[0]
            center = (s[self.decname], s[self.raname])
        return src.subimage(center, size, minsize=size, unit_size=u.arcsec)

    def preprocess(self, dataset, skip=True):
        """Create masks from the segmap, adapted to a given dataset.

        TODO: store in the database for each source: the sky and source mask
        paths, and preprocessing status

        """
        super().preprocess(dataset, skip=skip)
        debug = self.logger.debug

        # create output path if needed
        outpath = Path(self.settings['masks']['outpath']) / dataset.name
        outpath.mkdir(exist_ok=True)

        # sky mask
        sky_path = self.get_sky_mask_path(dataset)
        fsf = self.settings['masks']['convolve_fsf']

        if exists(sky_path) and skip:
            debug('sky mask exists, skipping')
        else:
            debug('creating sky mask')
            sky = self.segmap.get_mask(0)
            sky._data = np.logical_not(sky._data).astype(float)
            align_with_image(sky, dataset.white, order=0, inplace=True,
                             fsf_conv=fsf)
            sky.data /= np.max(sky.data)
            sky._data = np.where(sky._data > 0.1, 0, 1)
            sky.write(sky_path, savemask='none')

        # extract source masks
        size = self.settings['masks']['size']
        columns = [self.idname, self.raname, self.decname]

        # check sources inside dataset
        tab = self.select(columns=columns).as_table()
        ntot = len(tab)
        tab = tab.select(dataset.white.wcs, ra=self.raname, dec=self.decname)
        self.logger.info('%d sources inside dataset out of %d', len(tab), ntot)
        usize = u.arcsec
        ucent = u.deg
        minsize = min(*size) // 2

        # TODO: prepare the psf image for convolution
        # ima = gauss_image(self.shape, wcs=self.wcs, fwhm=fwhm, gauss=None,
        #                   unit_fwhm=usize, cont=0, unit=self.unit)
        # ima.norm(typ='sum')

        for id_, ra, dec in ProgressBar(tab, ipython_widget=isnotebook()):
            source_path = self.get_source_mask_path(dataset, id_)
            if exists(source_path):
                debug('source %05d exists, skipping', id_)
                continue

            debug('source %05d, extract mask', id_)
            mask = self.segmap.get_source_mask(
                id_, (dec, ra), size, unit_center=ucent, unit_size=usize)
            subref = dataset.white.subimage((dec, ra), size, minsize=minsize,
                                            unit_center=ucent, unit_size=usize)
            align_with_image(mask, subref, order=0, inplace=True, fsf_conv=fsf)
            mask.data /= mask.data.max()
            mask._data = np.where(mask._data > 0.1, 1, 0)
            mask.write(source_path, savemask='none')

    def add_to_source(self, src, dataset, nskywarn=(50, 5)):
        super().add_to_source(src)

        size = src.default_size
        center = (src.DEC, src.RA)
        seg_obj = self.get_source_mask(dataset, src.ID, size, center=center)
        seg_sky = self.get_sky_mask(dataset, src.ID, size, center=center)

        # add segmentation map
        src.images['SEG_HST'] = seg_obj
        src.images['SEG_HST_ALL'] = seg_sky

        # create masks
        src.find_sky_mask(['SEG_HST_ALL'], sky_mask='MASK_SKY')
        src.find_union_mask(['SEG_HST'], union_mask='MASK_OBJ')

        # delete temporary segmentation masks
        del src.images['SEG_HST_ALL'], src.images['SEG_HST']

        # compute surface of each masks and compare to field of view, save
        # values in header
        nsky = np.count_nonzero(src.images['MASK_SKY']._data)
        nobj = np.count_nonzero(src.images['MASK_OBJ']._data)
        nfracsky = 100.0 * nsky / np.prod(src.images['MASK_OBJ'].shape)
        nfracobj = 100.0 * nobj / np.prod(src.images['MASK_OBJ'].shape)
        min_nsky_abs, min_nsky_rel = nskywarn
        if nsky < min_nsky_abs or nfracsky < min_nsky_rel:
            self.logger.warning('Sky Mask is too small. Size is %d spaxel or '
                                '%.1f %% of total area', nsky, nfracsky)

        fsf = self.settings['masks']['convolve_fsf']
        src.add_attr('FSFMSK', fsf, 'HST Mask Conv Gauss FWHM in arcsec')
        src.add_attr('NSKYMSK', nsky, 'Size of MASK_SKY in spaxels')
        src.add_attr('FSKYMSK', nfracsky, 'Relative Size of MASK_SKY in %')
        src.add_attr('NOBJMSK', nobj, 'Size of MASK_OBJ in spaxels')
        src.add_attr('FOBJMSK', nfracobj, 'Relative Size of MASK_OBJ in %')
        # src.add_attr('MASKT1', thres[0], 'Mask relative threshold T1')
        # src.add_attr('MASKT2', thres[1], 'Mask relative threshold T2')
        return nobj, nfracobj, nsky, nfracobj
