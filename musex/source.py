import astropy.units as u
import logging
import numpy as np
from astropy.table import Column
from mpdaf.sdetect import Source, SourceList
from os.path import basename

# from .hstutils import skymask_from_hst, objmask_from_hst
from .version import __version__


class SourceX(Source):

    def add_mask_from_white(self):
        self.add_seg_images(['MUSE_WHITE'])
        self.find_union_mask(['SEG_MUSE_WHITE'], 'MASK_OBJ')
        self.find_sky_mask(['SEG_MUSE_WHITE'], 'MASK_SKY')

    def add_mask_from_aperture(self, radius, hstsegmap=None, fsf=0.6,
                               thres=(0.01, 0.1), nskywarn=(50, 5)):
        if 'MUSE_WHITE' not in self.images:
            raise ValueError('MUSE_WHITE image not found')
        white = self.images['MUSE_WHITE']
        if white.wcs.sameStep(hstsegmap.wcs):
            size = white.shape
        else:
            size = white.wcs.get_step(unit=u.arcsec)[0] * white.shape[0]

        self._logger.debug('Computing aperture mask of %.1f radius', radius)
        mask = white.copy()
        center = mask.wcs.sky2pix((self.DEC, self.RA), unit=u.deg)[0]
        rad = radius / (mask.wcs.get_step(unit=u.arcsec))
        radius2 = rad[0]**2
        mask.data *= 0
        grid = np.meshgrid(np.arange(mask.shape[0]) - center[0],
                           np.arange(mask.shape[1]) - center[1], indexing='ij')
        mask.data[(grid[0] ** 2 + grid[1] ** 2) < radius2] = 1

        if hstsegmap is not None:
            seg_sky = skymask_from_hst((self.RA, self.DEC), hstsegmap, white,
                                       size, fsf=fsf, thres=thres)
            self.images['SEG_HST'] = seg_sky
            self.images['SEG_APER'] = mask
            # create masks
            self.find_sky_mask(['SEG_HST', 'SEG_APER'], 'MASK_SKY')
            self.images['MASK_OBJ'] = mask
        else:
            self.images['MASK_OBJ'] = mask
            sky = mask.copy()
            self.images['MASK_SKY'] = 1 - sky

        # compute surface of each masks and compare to field of view, save
        # values in header
        nsky = np.count_nonzero(self.images['MASK_SKY']._data)
        nobj = np.count_nonzero(self.images['MASK_OBJ']._data)
        nfracsky = 100.0 * nsky / np.prod(self.images['MASK_OBJ'].shape)
        nfracobj = 100.0 * nobj / np.prod(self.images['MASK_OBJ'].shape)
        min_nsky_abs, min_nsky_rel = nskywarn
        if nsky < min_nsky_abs or nfracsky < min_nsky_rel:
            self._logger.warning('Sky Mask is too small. Size is %d spaxel or '
                                 '%.1f %% of total area', nsky, nfracsky)
        self.add_attr('FSFMSK', fsf, 'HST Mask Conv Gauss FWHM in arcsec')
        self.add_attr('NSKYMSK', nsky, 'Size of MASK_SKY in spaxels')
        self.add_attr('FSKYMSK', nfracsky, 'Relative Size of MASK_SKY in %')
        self.add_attr('NOBJMSK', nobj, 'Size of MASK_OBJ in spaxels')
        self.add_attr('FOBJMSK', nfracobj, 'Relative Size of MASK_OBJ in %')
        return nobj, nfracobj, nsky, nfracobj

    def add_mask_from_hst(self, hstsegmap, idlist, fsf=0.6, thres=(0.05, 0.1),
                          nskywarn=(50, 5)):
        """
        hstsegmap: hst segmentation map
        idlist: list of IDs
        fsf: fwhm of fsf in arcsec
        thres: threshold value for masking
        nskywarn: used to write up warnings if sky mask is too small
        """
        if 'MUSE_WHITE' not in self.images:
            raise ValueError('MUSE_WHITE image not found')
        shape = self.images['MUSE_WHITE'].shape
        step = self.images['MUSE_WHITE'].wcs.get_step(unit=u.arcsec)
        if self.images['MUSE_WHITE'].wcs.sameStep(hstsegmap.wcs):
            size = shape
        else:
            size = step[0] * shape[0]

        seg_obj = objmask_from_hst((self.ra, self.dec), idlist, hstsegmap,
                                   self.images['MUSE_WHITE'], size, fsf=fsf,
                                   thres=thres)
        seg_sky = skymask_from_hst((self.ra, self.dec), hstsegmap,
                                   self.images['MUSE_WHITE'], size, fsf=fsf,
                                   thres=thres)

        # add segmentation map
        self.images['SEG_HST'] = seg_obj
        self.images['SEG_HST_ALL'] = seg_sky

        # create masks
        self.find_sky_mask(['SEG_HST_ALL'], 'MASK_SKY')
        self.find_union_mask(['SEG_HST'], 'MASK_OBJ')

        # delete temporary segmentation masks
        del self.images['SEG_HST_ALL'], self.images['SEG_HST']

        # compute surface of each masks and compare to field of view, save
        # values in header
        nsky = np.count_nonzero(self.images['MASK_SKY']._data)
        nobj = np.count_nonzero(self.images['MASK_OBJ']._data)
        nfracsky = 100.0 * nsky / np.prod(self.images['MASK_OBJ'].shape)
        nfracobj = 100.0 * nobj / np.prod(self.images['MASK_OBJ'].shape)
        min_nsky_abs, min_nsky_rel = nskywarn
        if nsky < min_nsky_abs or nfracsky < min_nsky_rel:
            self._logger.warning('Sky Mask is too small. Size is %d spaxel or '
                                 '%.1f %% of total area', nsky, nfracsky)
        self.add_attr('FSFMSK', fsf, 'HST Mask Conv Gauss FWHM in arcsec')
        self.add_attr('NSKYMSK', nsky, 'Size of MASK_SKY in spaxels')
        self.add_attr('FSKYMSK', nfracsky, 'Relative Size of MASK_SKY in %')
        self.add_attr('NOBJMSK', nobj, 'Size of MASK_OBJ in spaxels')
        self.add_attr('FOBJMSK', nfracobj, 'Relative Size of MASK_OBJ in %')
        self.add_attr('MASKT1', thres[0], 'Mask relative threshold T1')
        self.add_attr('MASKT2', thres[1], 'Mask relative threshold T2')
        return nobj, nfracobj, nsky, nfracobj

    def add_mask(self, mask_mode, segmap=None, iden_hst=None, radius=0):
        self._logger.debug('Computing mask, mode %s', mask_mode)
        if mask_mode == 'HST':
            # compute mask from HST
            if segmap and iden_hst is not None:
                self.add_mask_from_hst(segmap, iden_hst)
            else:
                self._logger.error('No HST segmentation map found')
        elif mask_mode == 'WHITE':
            self.add_mask_from_white()
        elif mask_mode == 'APERTURE':
            self.add_mask_from_aperture(radius)
        elif mask_mode == 'APERHST':
            self.add_mask_from_aperture(radius, segmap)


class SourceListX(SourceList):

    source_class = SourceX

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def from_coords(cls, coords, origin=None, idname='ID', raname='RA',
                    decname='DEC', srcvers=''):
        origin = origin or ('MuseX', __version__, '', '')
        srclist = cls()
        for res in coords:
            src = cls.source_class.from_data(res[idname], res[raname],
                                             res[decname], origin)
            src.SRC_V = srcvers
            srclist.append(src)
        return srclist

    def add_datasets(self, muse_dataset, additional_datasets=None, size=5):
        logger = self.logger

        if additional_datasets is None:
            additional_datasets = []

        # TODO: ProgressBar, parallelize ?
        for src in self:
            # set PA: FIXME - useful ?
            # src.set_pa(muse_dataset.cube)

            logger.debug('Adding Datacube and white light image')
            src.default_size = size
            src.SIZE = size
            src.add_cube(muse_dataset.cube, f'{muse_dataset.prefix}_CUBE',
                         size=size, unit_wave=None, add_white=True)
            src.CUBE = basename(muse_dataset.settings['datacube'])
            src.CUBE_V = muse_dataset.version

            # add expmap image + average and dispersion value of expmap
            logger.debug('Adding expmap image')
            src.add_image(muse_dataset.expima, f'{muse_dataset.prefix}_EXPMAP')
            ima = src.images[f'{muse_dataset.prefix}_EXPMAP']
            src.EXPMEAN = (np.ma.mean(ima.data), 'Mean value of EXPMAP')
            src.EXPMIN = (np.ma.min(ima.data), 'Minimum value of EXPMAP')
            src.EXPMAX = (np.ma.max(ima.data), 'Maximum value of EXPMAP')
            logger.debug('Expmap mean %.2f min %.2f max %.2f',
                         src.EXPMEAN, src.EXPMIN, src.EXPMAX)

            # add fsf info
            if muse_dataset.cube.primary_header.get('FSFMODE') == 'MOFFAT1':
                logger.debug('Adding FSF info from the datacube')
                src.add_FSF(muse_dataset.cube)

            for ds in [muse_dataset] + additional_datasets:
                for name, img in getattr(ds, 'images', {}).items():
                    name = name.upper()
                    if name == 'WHITE':  # white image is handled separately
                        continue
                    tagname = getattr(img, 'name', name)
                    logger.debug('Adding image %s', tagname)
                    order = 0 if name == 'SEGMAP' else 1
                    src.add_image(img, f'{ds.prefix}_{tagname}',
                                  rotate=True, order=order)

    def add_catalog(self, cat, select_in_image, name='CAT', **select_kw):
        for src in self:
            wcs = src.images[select_in_image].wcs
            scat = cat.select(wcs, **select_kw)
            dist = scat.edgedist(wcs, **select_kw)
            scat.add_column(Column(name='DIST', data=dist))
            # FIXME: is it the same ?
            # cat = in_catalog(cat, src.images['HST_F775W_E'], quiet=True)
            self.logger.debug('Adding catalog %s (%d rows)', name, len(scat))
            src.add_table(scat, name)

    def extract_spectra(self, apertures):
        self.logger.debug('Extract spectra for apertures %s', apertures)
        for src in self:
            src.extract_all_spectra(src.cubes['MUSE_CUBE'],
                                    apertures=apertures)
