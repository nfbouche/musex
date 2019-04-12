import logging
import numpy as np

from astropy.io import fits
from astropy.table import Table
from mpdaf.MUSE.PSF import create_psf_cube
from mpdaf.sdetect import Source

from .pdf import create_pdf
from .utils import extract_subimage
from .version import __version__

__all__ = ('SourceX', 'create_source', 'sources_to_marz')


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


def create_source(iden, ra, dec, size, skyim, maskim, datasets, apertures,
                  verbose):
    logger = logging.getLogger(__name__)
    if not verbose:
        logging.getLogger('musex').setLevel('WARNING')

    # minsize = min(*size) // 2
    minsize = 0.
    nskywarn = (50, 5)
    origin = ('MuseX', __version__, '', '')

    src = SourceX.from_data(iden, ra, dec, origin)
    logger.info('source %05d (%.5f, %.5f)', src.ID, src.DEC, src.RA)
    src.default_size = size
    src.SIZE = size

    for ds in datasets:
        ds.add_to_source(src)

    center = (src.DEC, src.RA)

    if skyim is not None and maskim is not None:
        # FIXME: use Source.add_image instead ?
        src.images['MASK_SKY'] = extract_subimage(
            skyim, center, (size, size), minsize=minsize)

        # FIXME: check that center is inside mask
        # centerpix = maskim.wcs.sky2pix(center)[0]
        # debug('center: (%.5f, %.5f) -> (%.2f, %.2f)', *center,
        #       *centerpix.tolist())
        src.images['MASK_OBJ'] = extract_subimage(
            maskim, center, (size, size), minsize=minsize)

        # compute surface of each masks and compare to field of view, save
        # values in header
        nsky = np.count_nonzero(src.images['MASK_SKY']._data)
        nobj = np.count_nonzero(src.images['MASK_OBJ']._data)
        nfracsky = 100.0 * nsky / np.prod(src.images['MASK_OBJ'].shape)
        nfracobj = 100.0 * nobj / np.prod(src.images['MASK_OBJ'].shape)
        min_nsky_abs, min_nsky_rel = nskywarn
        if nsky < min_nsky_abs or nfracsky < min_nsky_rel:
            logger.warning('sky mask is too small. Size is %d spaxel '
                           'or %.1f %% of total area', nsky, nfracsky)

        src.add_attr('NSKYMSK', nsky, 'Size of MASK_SKY in spaxels')
        src.add_attr('FSKYMSK', nfracsky, 'Relative Size of MASK_SKY in %')
        src.add_attr('NOBJMSK', nobj, 'Size of MASK_OBJ in spaxels')
        src.add_attr('FOBJMSK', nfracobj, 'Relative Size of MASK_OBJ in %')
        # src.add_attr('MASKT1', thres[0], 'Mask relative threshold T1')
        # src.add_attr('MASKT2', thres[1], 'Mask relative threshold T2')
        # return nobj, nfracobj, nsky, nfracobj
        logger.debug('MASKS: SKY: %.1f%%, OBJ: %.1f%%', nfracsky, nfracobj)

        src.extract_all_spectra(apertures=apertures)

    # Joblib has a memmap reducer that does not work with astropy.io.fits
    # memmaps. So here we copy the arrays to avoid relying the memmaps.
    for name, im in src.images.items():
        src.images[name] = im.copy()
    for name, cube in src.cubes.items():
        src.cubes[name] = cube.copy()

    return src


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
    if save_src_to is not None:
        logger.info('Saving sources to %s', save_src_to)

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
