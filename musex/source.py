from mpdaf.sdetect import Source

from .pdf import create_pdf


class SourceX(Source):

    def extract_all_spectra(self, cube=None, apertures=None):
        self._logger.debug('Extract spectra for apertures %s', apertures)
        cube = cube or self.cubes['MUSE_CUBE']
        kw = dict(obj_mask='MASK_OBJ', sky_mask='MASK_SKY', unit_wave=None)
        self.extract_spectra(cube, skysub=False, apertures=apertures, **kw)
        self.extract_spectra(cube, skysub=True, apertures=apertures, **kw)
        if 'FSFMODE' in self.header:
            a, b, beta, field = self.get_FSF()
            fwhm = b * cube.wave.coord() + a
            self.extract_spectra(cube, skysub=False, psf=fwhm, beta=beta,
                                 apertures=None, **kw)
            self.extract_spectra(cube, skysub=True, psf=fwhm, beta=beta,
                                 apertures=None, **kw)

    def to_pdf(self, filename, white, **kwargs):
        create_pdf(self, white, filename, **kwargs)

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
