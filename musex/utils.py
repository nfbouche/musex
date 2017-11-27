import astropy.units as u
import numpy as np
from mpdaf.obj import Image


def extract_subimage(im, center, size, minsize=None, unit_size=u.arcsec):
    if isinstance(im, str):
        im = Image(im, copy=False)

    minsize = minsize or min(*size) // 2
    return im.subimage(center, size, minsize=minsize, unit_size=unit_size)


def align_with_image(img, other, inplace=False, order=1, fsf_conv=None):
    """Align `img` to match `ref` (rotation, fsf convolution, and resampling).
    """
    # Do nothing if the images are already aligned.
    if img.wcs.isEqual(other.wcs):
        return img if inplace else img.copy()

    # Determine the ranges of right-ascension and declination
    # covered by the target image grid plus an extra pixel at each edge.
    # TODO: add margin
    pixsky = other.wcs.pix2sky([[-1, -1],
                                [other.shape[0], -1],
                                [-1, other.shape[1]],
                                [other.shape[0], other.shape[1]]],
                               unit=u.deg)
    dec_min, ra_min = pixsky.min(axis=0)
    dec_max, ra_max = pixsky.max(axis=0)

    # Truncate the input image to just enclose the above ranges of
    # right-ascension and declination.
    out = img.truncate(dec_min, dec_max, ra_min, ra_max, mask=False,
                       unit=u.deg, inplace=inplace)

    # Rotate the image to have the same orientation as the other
    # image. Note that the rotate function has a side effect of
    # correcting the image for shear terms in the CD matrix, so we
    # perform this step even if no rotation is otherwise needed.
    out._rotate(other.wcs.get_rot() - out.wcs.get_rot(), reshape=True,
                regrid=True, flux=False, order=order)

    # convolve with FSF
    if fsf_conv:
        # TODO: replace with dilatation ?
        out.fftconvolve_gauss(fwhm=(fsf_conv, fsf_conv), inplace=True)

    # Get the pixel index and Dec,Ra coordinate at the center of
    # the image that we are aligning with.
    centerpix = np.asarray(other.shape) / 2.0
    centersky = other.wcs.pix2sky(centerpix)[0]

    # Re-sample the rotated image to have the same axis increments, offset and
    # number of pixels as the image that we are aligning it with.
    out.regrid(other.shape, centersky, centerpix,
               other.wcs.get_axis_increments(unit=u.deg),
               order=order, flux=False, unit_inc=u.deg, inplace=True)
    return out
