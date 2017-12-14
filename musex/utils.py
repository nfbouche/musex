import astropy.units as u
import numpy as np
from astropy.io import fits
from mpdaf.obj import Image, moffat_image


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


def align_mask_with_image(mask, other, inplace=True, fsf_conv=None,
                          threshold=0.1, inverse=False, outname=None):
    mask = align_with_image(mask, other, order=0, inplace=inplace,
                            fsf_conv=fsf_conv)
    data = mask.data.filled(0)
    data /= data.max()
    if inverse:
        mask._data = np.where(data > threshold, 0, 1)
    else:
        mask._data = np.where(data > threshold, 1, 0)

    if outname:
        mask.write(outname, savemask='none')

    return mask


def regrid_to_image(im, other, order=1, inplace=False, antialias=True,
                    size=None, unit_size=u.arcsec, **kwargs):
    im.data = im.data.astype(float)
    refpos = im.wcs.pix2sky([0, 0])[0]
    if size is not None:
        newdim = size / other.wcs.get_step(unit=unit_size)
    else:
        newdim = other.shape
    inc = other.wcs.get_axis_increments(unit=unit_size)
    im = im.regrid(newdim, refpos, [0, 0], inc, order=order,
                   unit_inc=unit_size, inplace=inplace, antialias=antialias)
    return im


def combine_masks(imlist, method='union', outname=None):
    images = [fits.getdata(f).astype(bool) for f in imlist]
    if method == 'union':
        data = np.logical_or.reduce(images)
    elif method == 'intersection':
        data = np.logical_and.reduce(images)
    else:
        raise ValueError("unknown method, must be 'union' or 'intersection'")

    out = Image(imlist[0], copy=False)
    out._data = data.astype(np.uint8)
    if outname:
        out.write(outname, savemask='none')
    return out


def struct_from_moffat_fwhm(wcs, fwhm, psf_threshold=0.5, beta=2.5):
    """Compute a structuring element for the dilatation, to simulate
    a convolution with a psf."""
    # image size will be twice the full-width, to account for
    # psf_threshold < 0.5
    size = int(round(fwhm / wcs.get_step(u.arcsec)[0])) * 2 + 1
    if size % 2 == 0:
        size += 1

    psf = moffat_image(fwhm=(fwhm, fwhm), n=beta, peak=True,
                       wcs=wcs[:size, :size])

    # remove useless zeros on the edges.
    psf.mask_selection(psf._data < psf_threshold)
    psf.crop()
    assert tuple(np.array(psf.shape) % 2) == (1, 1)
    return ~psf.mask
