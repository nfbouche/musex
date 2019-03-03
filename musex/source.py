import logging

from astropy.io import fits
from astropy.table import Table

from mpdaf.MUSE.PSF import create_psf_cube
from mpdaf.sdetect import Source

import numpy as np

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


def sources_to_marz(src_list, out_file, *, save_src_to=None,
                    check_keyword=None):
    """Export a list of source to a MarZ input file.

    Parameters
    ----------
    src_list : list of mpdaf.obj.Source
        List or generator of mpdaf sources.
    out_file : str
        Filename for the FITS file to use as input to MarZ.
    save_src_to : str, optional
        None or a template string that is formated with the source as `src` to
        get the name of the file to save the source to (for instance
        `/path/to/source-{src.ID:05d}.fits`).
    check_keyword: tuple, optional
        If a tuple (keyword, value) is given, each source header will be
        checked that it contains the keyword with the value. If the keyword is
        not here, a KeyError will be raise, if the value is not the expected
        one, a ValueError is raised.

    """
    logger = logging.getLogger(__name__)

    wave, data, stat, sky, meta = [], [], [], [], []

    for src in src_list:
        if check_keyword is not None:
            try:
                if src.header[check_keyword[0]] != check_keyword[1]:
                    raise ValueError("The source was not made from the good "
                                     "catalog: %s = %s", (check_keyword))
            except KeyError:
                raise KeyError("The source has no %s keyword.",
                               check_keyword[0])

        sp = src.spectra[src.REFSPEC]
        wave.append(sp.wave.coord())
        data.append(sp.data.filled(np.nan))
        stat.append(sp.var.filled(np.nan))
        sky.append(src.spectra['MUSE_SKY'].data.filled(np.nan))

        # TODO: The following comments were in the list of source to MarZ code
        # when it was into the musex.export_marz method.
        # if args.selmode == 'udf':
        #     zmuse = s.z[s.z['Z_DESC'] == 'MUSE']
        #     z = 0 if len(zmuse) == 0 else zmuse['Z'][0]
        #     band1 = s.mag[s.mag['BAND'] == 'F775W']
        #     mag1 = -99 if len(band1)==0  else band1['MAG'][0]
        #     band2 = s.mag[s.mag['BAND'] == 'F125W']
        #     mag2 = -99 if len(band2)==0  else band2['MAG'][0]
        z = 0
        mag1 = -99
        mag2 = -99
        meta.append(('%05d' % src.ID, src.RA, src.DEC, z,
                     src.header.get('CONFID', 0),
                     src.header.get('TYPE', 0), mag1, mag2, src.REFSPEC))

        if save_src_to is not None:
            fname = save_src_to.format(src=src)
            src.write(fname)
            logger.info('fits written to %s', fname)

    t = Table(rows=meta, names=['NAME', 'RA', 'DEC', 'Z', 'CONFID', 'TYPE',
                                'F775W', 'F125W', 'REFSPEC'],
              meta={'CUBE_V': src.CUBE_V, 'SRC_V': src.SRC_V})
    hdulist = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(name='WAVE', data=np.vstack(wave)),
        fits.ImageHDU(name='DATA', data=np.vstack(data)),
        fits.ImageHDU(name='STAT', data=np.vstack(stat)),
        fits.ImageHDU(name='SKY', data=np.vstack(sky)),
        fits.BinTableHDU(name='DETAILS', data=t)
    ])
    logger.info('Writing %s', out_file)
    hdulist.writeto(out_file, overwrite=True, output_verify='silentfix')
