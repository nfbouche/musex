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

from .hstutils import align_with_image
from .segmap import SegMap
from .settings import isnotebook

DIRNAME = os.path.abspath(os.path.dirname(__file__))

__all__ = ['load_input_catalogs', 'Catalog', 'PriorCatalog']


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
        conf = conf['import']
        if 'class' in conf:
            mod, class_ = conf['class'].rsplit('.', 1)
            mod = importlib.import_module(mod)
            cls = getattr(mod, class_)
        else:
            cls = InputCatalog
        catalogs[name] = cls(name, db, workdir=settings['workdir'], **conf)
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


class Catalog:
    """Handle Catalogs by the way of a database table.
    """

    def __init__(self, name, db, workdir=None, idname='ID', raname='RA',
                 decname='DEC'):
        self.name = name
        self.db = db
        self.idname = idname
        self.raname = raname
        self.decname = decname
        self.logger = logging.getLogger(__name__)

        if self.name in self.db:
            self.table = self.db[self.name]
        else:
            self.logger.info('create table %s (primary key: %s)',
                             self.name, self.idname)
            self.table = self.db.create_table(self.name,
                                              primary_id=self.idname)

        # Work dir for intermediate files
        if workdir is None:
            raise Exception('FIXME: find a better way to handle this')
        self.workdir = Path(workdir) / self.name
        self.workdir.mkdir(exist_ok=True)

    def __len__(self):
        return self.table.count()

    def preprocess(self, dataset, skip=True):
        """Generate intermediate results linked to a given dataset."""

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
        # conf = self.settings['extract']
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


class InputCatalog(Catalog):
    """Handles catalogs imported from an exiting file."""

    def __init__(self, name, db, catalog=None, version=None, **kwargs):
        super().__init__(name, db, **kwargs)
        if catalog is None:
            raise ValueError('an input catalog is required')
        if version is None:
            raise ValueError('an input version is required')
        self.catalog = catalog
        self.version = version

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


class PriorCatalog(InputCatalog):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skyim = {}

    @lazyproperty
    def has_segmap(self):
        """True if a segmentation map is available."""
        return 'segmap' in self.settings

    @lazyproperty
    def segmap(self):
        """The segmentation map."""
        if self.has_segmap:
            return SegMap(self.settings['segmap'])

    def get_sky_mask_path(self, dataset):
        return self.workdir / dataset.name / 'sky.fits'

    def get_source_mask_path(self, dataset, source_id):
        return (self.workdir / dataset.name /
                self.settings['masks']['outname'].format(source_id))

    def get_sky_mask(self, dataset, source_id, size, center=None):
        sky = self._skyim.get(dataset)
        if sky is None:
            sky = self._skyim[dataset] = Image(
                str(self.get_sky_mask_path(dataset)))
        if center is None:
            s = self.select(self.c[self.idname] == source_id)[0]
            center = (s[self.decname], s[self.raname])
        minsize = min(*size) // 2
        return sky.subimage(center, size, minsize=minsize, unit_size=u.arcsec)

    def get_source_mask(self, dataset, source_id, size, center=None):
        src = Image(str(self.get_source_mask_path(dataset, source_id)))
        if center is None:
            s = self.select(self.c[self.idname] == source_id)[0]
            center = (s[self.decname], s[self.raname])
        minsize = min(*size) // 2
        return src.subimage(center, size, minsize=minsize, unit_size=u.arcsec)

    def preprocess(self, dataset, skip=True):
        """Create masks from the segmap, adapted to a given dataset.

        TODO: store in the database for each source: the sky and source mask
        paths, and preprocessing status

        """
        super().preprocess(dataset, skip=skip)
        debug = self.logger.debug

        if self.segmap is None:
            self.logger.info('No segmap available, skipping masks creation')
            return

        # create output path if needed
        outpath = self.workdir / dataset.name
        outpath.mkdir(exist_ok=True)

        # sky mask
        sky_path = self.get_sky_mask_path(dataset)
        fsf = self.settings['masks'].get('convolve_fsf')

        if sky_path.exists() and skip:
            debug('sky mask exists, skipping')
        else:
            debug('creating sky mask')
            sky = self.segmap.get_mask(0)
            sky._data = np.logical_not(sky._data).astype(float)
            align_with_image(sky, dataset.white, order=0, inplace=True,
                             fsf_conv=fsf)
            sky.data /= np.max(sky.data)
            sky._data = np.where(sky._data > 0.1, 0, 1)
            sky.write(str(sky_path), savemask='none')

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
            if source_path.exists() and skip:
                debug('source %05d exists, skipping', id_)
                continue

            debug('source %05d (%.5f, %.5f), extract mask', id_, ra, dec)
            mask = self.segmap.get_source_mask(
                id_, (dec, ra), size, minsize=minsize, unit_center=ucent,
                unit_size=usize)
            subref = dataset.white.subimage((dec, ra), size, minsize=minsize,
                                            unit_center=ucent, unit_size=usize)
            align_with_image(mask, subref, order=0, inplace=True, fsf_conv=fsf)
            data = mask.data.filled(0)
            mask._data = np.where(data / data.max() > 0.1, 1, 0)
            mask.write(str(source_path), savemask='none')

    def add_to_source(self, src, dataset, nskywarn=(50, 5)):
        super().add_to_source(src)

        size = (src.default_size, src.default_size)
        center = (src.DEC, src.RA)
        seg_obj = self.get_source_mask(dataset, src.ID, size, center=center)
        seg_sky = self.get_sky_mask(dataset, src.ID, size, center=center)

        # add segmentation map
        src.images['MASK_SKY'] = seg_sky

        # FIXME: check that this is enough (no need to use find_union_mask)
        src.images['MASK_OBJ'] = seg_obj
        # src.images['SEG_HST'] = seg_obj
        # src.find_union_mask(['SEG_HST'], union_mask='MASK_OBJ')
        # # delete temporary segmentation masks
        # del src.images['SEG_HST']

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

        fsf = self.settings['masks'].get('convolve_fsf', 0)
        src.add_attr('FSFMSK', fsf, 'Mask Conv Gauss FWHM in arcsec')
        src.add_attr('NSKYMSK', nsky, 'Size of MASK_SKY in spaxels')
        src.add_attr('FSKYMSK', nfracsky, 'Relative Size of MASK_SKY in %')
        src.add_attr('NOBJMSK', nobj, 'Size of MASK_OBJ in spaxels')
        src.add_attr('FOBJMSK', nfracobj, 'Relative Size of MASK_OBJ in %')
        # src.add_attr('MASKT1', thres[0], 'Mask relative threshold T1')
        # src.add_attr('MASKT2', thres[1], 'Mask relative threshold T2')
        # return nobj, nfracobj, nsky, nfracobj
