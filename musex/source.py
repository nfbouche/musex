import astropy.units as u
import logging
import numpy as np
import os
import shutil
from mpdaf.sdetect import Source, SourceList

# from .hstutils import skymask_from_hst, objmask_from_hst
from .pdf import create_pdf
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

    def get_fsf(self):
        if 'FSFMODE' not in self.header:
            return
        for field in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99]:
            if 'FSF{:02d}BET'.format(field) in self.header:
                beta = self.header['FSF{:02d}BET'.format(field)]
                a = self.header['FSF{:02d}FWA'.format(field)]
                b = self.header['FSF{:02d}FWB'.format(field)]
                return a, b, beta, field

    def extract_all_spectra(self, cube=None, apertures=None):
        self._logger.debug('Extract spectra for apertures %s', apertures)
        cube = cube or self.cubes['MUSE_CUBE']
        kw = dict(obj_mask='MASK_OBJ', sky_mask='MASK_SKY', unit_wave=None)
        self.extract_spectra(cube, skysub=False, apertures=apertures, **kw)
        self.extract_spectra(cube, skysub=True, apertures=apertures, **kw)
        if 'FSFMODE' in self.header:
            a, b, beta, field = self.get_fsf()
            fwhm = b * cube.wave.coord() + a
            self.extract_spectra(cube, skysub=False, psf=fwhm, beta=beta,
                                 apertures=None, **kw)
            self.extract_spectra(cube, skysub=True, psf=fwhm, beta=beta,
                                 apertures=None, **kw)

        # FIXME: handle this correctly... (set_refspec)
        self.add_attr('REFSPEC', 'MUSE_PSF_SKYSUB',
                      desc='Name of reference spectra')


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

    def export_to_pdf(self, name, white, path='.'):
        path = os.path.join(os.path.normpath(path), name)
        os.makedirs(path, exist_ok=True)

        # TODO: pdf filename with infos
        # if info:
        #     # f = os.path.splitext(os.path.basename(src.filename))[0]
        #     confid = str(src.CONFID) if hasattr(src, 'CONFID') else 'u'
        #     stype = str(src.TYPE) if hasattr(src, 'TYPE') else 'u'
        #     z = src.get_zmuse()
        #     if z is None:
        #         zval = 'uu'
        #     else:
        #         if z < 0:
        #             zval = '00'
        #         else:
        #             zval = '{:.1f}'.format(z)
        #             zval = zval[0] + zval[2]
        #     # outfile = '{}_t{}_c{}_z{}.pdf'.format(f, stype, confid, zval)

        info = self.logger.info
        nfiles = len(self)
        for k, src in enumerate(self):
            outf = f'{path}/{name}-{src.ID:04d}.pdf'
            info('%d/%d: PDF source %s -> %s', k + 1, nfiles, src.ID, outf)
            create_pdf(src, white, outf)
