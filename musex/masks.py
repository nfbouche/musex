"""Mask related tools."""
import logging

from astropy.io.fits import ImageHDU
from astropy.nddata.utils import NoOverlapError, overlap_slices
from astropy.wcs import WCS
import numpy as np


def _same_origin(m1, m2):
    """Find if two masks where extracted from the same image.

    We compare WCS information in the two masks:
    - the projection type (CTYPE)
    - the position of the reference pixel (CRVAL)
    - the transformation matrix
    All must be the same.
    """
    w1, w2 = WCS(m1), WCS(m2)

    return (
        w1.wcs.ctype[0] == w2.wcs.ctype[0] and
        w1.wcs.ctype[1] == w2.wcs.ctype[1] and
        np.all(w1.wcs.crval == w2.wcs.crval) and
        np.all(w1.pixel_scale_matrix == w2.pixel_scale_matrix)
    )


def merge_masks_on_area(ra, dec, size, mask_list, *, is_sky=False):
    """Merge sky masks on a given area.

    This function generate a mask at the given position (center) and of the
    given size combining several masks on the area. The masks are FITS HDUs
    containing 0 or 1, and must have been extracted on the same image. If the
    WCS information of the masks shows that they were extracted on different
    maps, a ValueError exception is raised.

    By default, the masks are combined with the OR function; this is valid for
    source masks. If the is_sky mask is set to true, the masks are combined
    with the AND function. This is because sources coming from ORIGIN may have
    different sky masks on the same area and we want the pixels that are sky in
    all the sources.

    Parameters
    ----------
    ra: float
        Right Ascension of the center of the result mask in decimal degrees.
    dec: float
        Declination of the center of the result mask in decimal degrees.
    size: (int, int)
        Size of the mask (x size, y, size) in pixels. The size of a pixel is
        the same as in the passed masks.
     mask_list: List[astropy.io.fits.hdu.image.ImageHDU]
        List of the masks to combine.
    is_sky: bool
        Set to True when the masks are sky ones.

    Returns
    -------
    astropy.io.fits.hdu.image.ImageHDU

    """
    logger = logging.getLogger(__name__)

    # Astropy overlap_slices has a specific way to compute offsets and image
    # centers to have consistent cutouts between arrays of odd and even
    # dimensions. Instead of re-implementing the offset computation, we just
    # take it from the first mask we use.
    result_wcs = None

    # If we are combining source masks, we start with an initial Boolean mask
    # to 0 and combine all the individual mask to this one with a logical OR
    # operation: we want want all the points in any of the original masks.
    # If we are combining sky masks, we want the pixels that are marked as sky
    # in all the original masks (because ORIGIN will given different sky masks
    # for nearby sources). We could start with a Boolean mask set to 1 and
    # combine with AND, but if we ask for a too big mask, the pixels not
    # covered by any of the original masks would stay to 0. To avoid that, we
    # start with and integer 0 mask and add all the individual masks. The final
    # sky mask it all the pixels with a value equal to the number of mask
    # combined.
    if is_sky:
        result_data = np.zeros((size[1], size[0]), dtype=np.uint8)
        combine = np.add
    else:
        result_data = np.zeros((size[1], size[0]), dtype=bool)
        combine = np.logical_or

    for mask in mask_list:
        if (result_wcs is not None and
                not _same_origin(mask, result_wcs.to_header())):
            raise ValueError("Not all the masks used in merge_masks_on_area "
                             "were extracted on the same image.")

        mask_size = mask.data.shape
        result_center = WCS(mask).all_world2pix(ra, dec, 0)
        try:
            mask_slice, result_slice = overlap_slices(
                mask_size, size, result_center)

            if result_wcs is None:
                result_wcs = WCS(mask).copy()
                offset = (
                    mask_slice[0].start - result_slice[0].start,
                    mask_slice[1].start - result_slice[1].start)
                result_wcs.wcs.crpix -= offset

            result_data[result_slice[::-1]] = combine(
                result_data[result_slice[::-1]],
                mask.data[mask_slice[::-1]].astype(bool))

        except NoOverlapError:
            # TODO: Add better warning.
            logger.warning("The mask does not overlap with the target mask.")

    # If result_wcs is still None, that means that none of the merged masks
    # were overlapping the region.
    if result_wcs is None:
        raise ValueError("None of the provided mask was overlapping the "
                         "given position.")

    if is_sky:
        result_data = (result_data == len(mask_list)).astype(np.uint8)

    return ImageHDU(header=result_wcs.to_header(),
                    data=result_data.astype(np.uint8))
