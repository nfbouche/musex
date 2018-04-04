"""Mask related tools."""
import logging

from astropy.io.fits import ImageHDU
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


def _overlap_limits(size1, size2, offset):
    """Compute overlap limits of two 1D array with an offset.

    Given two array switched with an offset, this function compute the
    limit indexes en each array for the overlapping region. If there is no
    overlap, ValueError is raised.

    The offset is to go from array 1 to array 2.

    +-+-+-+-+-+-+
    A1 |0|1|2|3|4|5|
    +-+-+-+-+-+-+-+-+
    A2  offset |0|1|2|3|
        =4   +-+-+-+-+

    """
    if (offset >= size1) or (offset <= -size2):
        raise ValueError("The arrays do not overlap.")

    # Longer version easier to understand:
    # if offset >= 0:
    #     min1 = offset
    #     max1 = min(size1, size2 + offset)
    #     min2 = 0
    #     max2 = min(size2, size1 - offset)
    # else:
    #     min1 = 0
    #     max1 = min(size1, size2 + offset)
    #     min2 = -offset
    #     max2 = min(size2, size1 - offset)

    min1 = max(offset, 0)
    max1 = min(size1, size2 + offset)
    min2 = max(-offset, 0)
    max2 = min(size2, size1 - offset)

    return min1, max1, min2, max2


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

    def _offset(mask):
        """Compute the pixel offsets for a mask.

        Given a mask header, this function computes the pixel offsets to go
        from the mask to the new mask the merge_masks_on_area if creating.

        We compute the position of the center of the new mask. The offset is
        then the position of the (0, 0) pixel in the target mask, computed
        using its size.

        """
        wcs = WCS(mask)

        target_center_x, target_center_y = wcs.all_world2pix(ra, dec, 0)
        offset_x = int(target_center_x) - int(size[0] / 2)
        offset_y = int(target_center_y) - int(size[1] / 2)

        return offset_x, offset_y

    # WCS of the result mask. We take the WCS of the first mask and change the
    # the pixel position of the reference pixel (CRPIX).
    m0 = mask_list[0]
    result_mask_wcs = WCS(m0).copy()
    result_mask_wcs.wcs.crpix -= _offset(m0)

    # If we are processing source masks, we set the initial mask to 0 as we
    # combine with OR. If we are processing sky masks, we set the initial mask
    # to 1 as we combine with AND.
    if is_sky:
        result_data = np.ones((size[1], size[0]), dtype=bool)
        combine = np.logical_and
    else:
        result_data = np.zeros((size[1], size[0]), dtype=bool)
        combine = np.logical_or

    for mask in mask_list:
        if not _same_origin(mask, result_mask_wcs.to_header()):
            raise ValueError("Not all the masks used in merge_masks_on_area "
                             "were extracted on the same image.")

        mask_size = mask.data.shape
        offset = _offset(mask)
        try:
            x1_min, x1_max, x2_min, x2_max = _overlap_limits(
                mask_size[0], size[0], offset[0])
            y1_min, y1_max, y2_min, y2_max = _overlap_limits(
                mask_size[1], size[1], offset[1])
        except ValueError:
            # TODO: Add better warning.
            logger.warning("The mask does not overlap with the target mask.")
            continue

        result_data[y2_min:y2_max, x2_min:x2_max] = combine(
            result_data[y2_min:y2_max, x2_min:x2_max],
            mask.data[y1_min:y1_max, x1_min:x1_max].astype(bool))

    return ImageHDU(header=result_mask_wcs.to_header(),
                    data=result_data.astype(np.uint8))
