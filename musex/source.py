import logging
import numpy as np
import re
import textwrap

from astropy.io import fits
from astropy.table import Table
from mpdaf.MUSE.PSF import create_psf_cube
from mpdaf.sdetect import Source

from .pdf import create_pdf
from .utils import extract_subimage
from .version import __version__

__all__ = ('SourceX', 'create_source', 'sources_to_marz')

# Default comments for FITS keywords
HEADER_COMMENTS = dict(
    CONFID='Z Confidence Flag',
    BLEND='Blending flag',
    DEFECT='Defect flag',
    REVISIT='Reconciliation Revisit Flag',
    TYPE='Object classification',
    REFSPEC='Name of reference spectra',
)


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

    def add_z_from_settings(self, redshifts, row):
        """Add redshifts from a row using the settings definition."""
        for name, val in redshifts.items():
            try:
                if isinstance(val, str):
                    z, errz = row[val], 0
                else:
                    z, errz = row[val[0]], (row[val[1]], row[val[2]])
            except KeyError:
                continue

            if z is not None and 0 <= z < 50:
                self._logger.debug('Add redshift %s=%.2f (err=%r)',
                                   name, z, errz)
                self.add_z(name, z, errz=errz)

    def add_mag_from_settings(self, mags, row):
        """Add magnitudes from a row using the settings definition."""
        for name, val in mags.items():
            try:
                if isinstance(val, str):
                    mag, magerr = row[val], 0
                else:
                    mag, magerr = row[val[0]], row[val[1]]
            except KeyError:
                continue

            if mag is not None and 0 <= mag < 50:
                self._logger.debug('Add mag %s=%.2f (err=%r)',
                                   name, mag, magerr)
                self.add_mag(name, mag, magerr)

    def add_header_from_settings(self, header_columns, row):
        """Add keywords from a row using the settings definition."""
        for key, colname in header_columns.items():
            if row.get(colname) is not None:
                if key == 'COMMENT':
                    # truncate comment if too long
                    com = re.sub(r'[^\s!-~]', '', row[colname])
                    for txt in textwrap.wrap(com, 50):
                        self.add_comment(txt, '')
                else:
                    self._logger.debug('Add %s=%r', key, row[colname])
                    comment = HEADER_COMMENTS.get(key)
                    self.header[key] = (row[colname], comment)

    def add_masks_from_dataset(self, maskds, center, size, minsize=0,
                               nskywarn=(50, 5)):
        skyim = maskds.get_skymask_file(self.ID)
        maskim = maskds.get_objmask_file(self.ID)

        # FIXME: use Source.add_image instead ?
        self.images['MASK_SKY'] = extract_subimage(
            skyim, center, (size, size), minsize=minsize)

        # FIXME: check that center is inside mask
        # centerpix = maskim.wcs.sky2pix(center)[0]
        # debug('center: (%.5f, %.5f) -> (%.2f, %.2f)', *center,
        #       *centerpix.tolist())
        self.images['MASK_OBJ'] = extract_subimage(
            maskim, center, (size, size), minsize=minsize)

        # compute surface of each masks and compare to field of view, save
        # values in header
        nsky = np.count_nonzero(self.images['MASK_SKY']._data)
        nobj = np.count_nonzero(self.images['MASK_OBJ']._data)
        fracsky = 100.0 * nsky / np.prod(self.images['MASK_OBJ'].shape)
        fracobj = 100.0 * nobj / np.prod(self.images['MASK_OBJ'].shape)
        min_nsky_abs, min_nsky_rel = nskywarn
        if nsky < min_nsky_abs or fracsky < min_nsky_rel:
            self._logger.warning('sky mask is too small. Size is %d spaxel '
                                 'or %.1f %% of total area', nsky, fracsky)

        self.add_attr('NSKYMSK', nsky, 'Size of MASK_SKY in spaxels')
        self.add_attr('FSKYMSK', fracsky, 'Relative Size of MASK_SKY in %')
        self.add_attr('NOBJMSK', nobj, 'Size of MASK_OBJ in spaxels')
        self.add_attr('FOBJMSK', fracobj, 'Relative Size of MASK_OBJ in %')
        self._logger.debug('MASKS: SKY: %.1f%%, OBJ: %.1f%%', fracsky, fracobj)


def create_source(row, idname, raname, decname, size, refspec, history,
                  segmap=None, datasets=None, maskds=None, apertures=None,
                  header=None, header_columns=None, redshifts=None, mags=None,
                  verbose=False, **kwargs):
    logger = logging.getLogger(__name__)
    if not verbose:
        logging.getLogger('musex').setLevel('WARNING')

    origin = ('MuseX', __version__, '', '')

    src = SourceX.from_data(row[idname], row[raname], row[decname], origin,
                            default_size=size)
    logger.debug('create source %05d (%.5f, %.5f)', src.ID, src.DEC, src.RA)
    src.SIZE = size
    if header:
        src.header.update(header)

    if datasets:
        for ds, names in datasets.items():
            logger.debug('Add dataset %s', ds.name)
            ds.add_to_source(src, names=names)

    # Add keywords from columns
    if header_columns:
        src.add_header_from_settings(header_columns, row)

    if src.header.get('REFSPEC') is None:
        logger.warning('REFSPEC column not found, using %s instead', refspec)
        src.add_attr('REFSPEC', refspec, desc=HEADER_COMMENTS['REFSPEC'])

    # Add redshifts
    if redshifts:
        src.add_z_from_settings(redshifts, row)

    # Add magnitudes
    if mags:
        src.add_mag_from_settings(mags, row)

    if segmap:
        logger.debug('Add segmap %s', segmap[0])
        src.SEGMAP = segmap[0]
        src.add_image(segmap[1], segmap[0], rotate=True, order=0)

    if history:
        for args in history:
            src.add_history(*args)

    # FIXME: masks could be added from sources
    if maskds is not None:
        src.add_masks_from_dataset(maskds, (src.DEC, src.RA), size)
        src.extract_all_spectra(apertures=apertures)

    return src


def sources_to_marz(src_list, outfile, refspec=None, skyspec='MUSE_SKY',
                    save_src_to=None, check_keyword=None):
    """Export a list of source to a MarZ input file.

    Parameters
    ----------
    src_list : iterable of `mpdaf.sdetect.Source`
        List of mpdaf sources.
    outfile : str
        Filename for the FITS file to use as input to MarZ.
    refspec : str, optional
        The spectrum to use, defaults to ``src.REFSPEC``.
    skyspec : str, optional
        The sky spectrum to use, defaults to ``MUSE_SKY``.
    save_src_to : str, optional
        None or a template string that is formated with the source as `src` to
        get the name of the file to save the source to (for instance
        `/path/to/source-{src.ID:05d}.fits`).
    check_keyword : tuple, optional
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
            key, val = check_keyword
            try:
                if src.header[key] != val:
                    raise ValueError("The source was not made from the good "
                                     f"catalog: {key} = {val}")
            except KeyError:
                raise KeyError(f"The source has no {key} keyword.")

        refsp = refspec or src.REFSPEC
        sp = src.spectra[refsp]
        wave.append(sp.wave.coord())
        data.append(sp.data.filled(np.nan))
        stat.append(sp.var.filled(np.nan))
        sky.append(src.spectra[skyspec].data.filled(np.nan))

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
                     src.header.get('TYPE', 0), mag1, mag2, refsp))

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
    logger.info('Writing %s', outfile)
    hdulist.writeto(outfile, overwrite=True, output_verify='silentfix')
