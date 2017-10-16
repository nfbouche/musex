import astropy.units as u
import logging
import numpy as np
from mpdaf.sdetect import Source, SourceList

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

    def extract_spectra(self, apertures):
        self.logger.debug('Extract spectra for apertures %s', apertures)
        for src in self:
            src.extract_all_spectra(src.cubes['MUSE_CUBE'],
                                    apertures=apertures)
