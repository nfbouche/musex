from mpdaf.MUSE.PSF import create_psf_cube
from mpdaf.sdetect import Source

from .pdf import create_pdf

__all__ = ['SourceX']


class SourceX(Source):
    """A Source class inherited from `mpdaf.sdetect.Source`, and customized
    with a few additional methods.
    """

    def get_zmuse(self):
        """Return the MUSE redshift if available."""
        sel = self.z[self.z['Z_DESC'] == 'MUSE']
        if len(sel) > 0:
            return sel['Z'][0]

    def extract_all_spectra(self, cube=None, apertures=None):
        self._logger.debug('Extract spectra for apertures %s', apertures)
        cube = cube or self.cubes['MUSE_CUBE']
        kw = dict(obj_mask='MASK_OBJ', sky_mask='MASK_SKY',
                  apertures=apertures, unit_wave=None)

        if 'FSFMODE' in self.header:
            a, b, beta, field = self.get_FSF()
            kw['beta'] = beta
            psf = b * cube.wave.coord() + a
            kw['psf'] = create_psf_cube(cube.shape, psf, beta=beta,
                                        wcs=cube.wcs)

        self.extract_spectra(cube, skysub=False, **kw)
        self.extract_spectra(cube, skysub=True, **kw)

    @property
    def refcat(self):
        try:
            return self.tables[self.REFCAT]
        except KeyError:
            self._logger.debug('Ref catalog "%s" not found', self.REFCAT)

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
