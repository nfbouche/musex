# import astropy.units as u
import logging
# import numpy as np
import os
# import shutil
from mpdaf.sdetect import Source, SourceList

from .pdf import create_pdf
from .version import __version__


class SourceX(Source):

    def get_fsf(self):
        if 'FSFMODE' not in self.header:
            return
        for field in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99):
            if f'FSF{field:02d}BET' in self.header:
                beta = self.header[f'FSF{field:02d}BET']
                a = self.header[f'FSF{field:02d}FWA']
                b = self.header[f'FSF{field:02d}FWB']
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
