import astropy.units as u
import logging
import numpy as np
from mpdaf.obj import Image
from scipy import ndimage as ndi

from .utils import regrid_to_image, isiter


class SegMap:
    """
    Handle segmentation maps, where pixel values are sources ids.
    """

    def __init__(self, path=None, data=None, cut_header_after='D001VER'):
        self.path = path
        self.logger = logging.getLogger(__name__)
        if data is not None:
            if isinstance(data, Image):
                self.img = data
            else:
                self.img = Image(data=data, copy=False, mask=np.ma.nomask)
        else:
            self.img = Image(path, copy=False, mask=np.ma.nomask)

        if cut_header_after:
            if cut_header_after in self.img.data_header:
                idx = self.img.data_header.index(cut_header_after)
                self.img.data_header = self.img.data_header[:idx]
            if cut_header_after in self.img.primary_header:
                idx = self.img.primary_header.index(cut_header_after)
                self.img.primary_header = self.img.primary_header[:idx]

    def copy(self):
        im = self.__class__(path=self.path, data=self.img.copy())
        im._mask = np.ma.nomask
        return im

    def get_mask(self, value, dtype=np.uint8, dilate=None, inverse=False,
                 struct=None, regrid_to=None, outname=None):
        if inverse:
            data = (self.img._data != value)
        else:
            data = (self.img._data == value)
        if dilate:
            data = dilate_mask(data, niter=dilate, struct=struct)

        im = Image.new_from_obj(self.img, data)
        if regrid_to:
            im = regrid_to_image(im, regrid_to, inplace=True, order=0,
                                 antialias=False)
            np.around(im._data, out=im._data)

        im._data = im._data.astype(dtype)
        im._mask = np.ma.nomask
        if inverse:
            np.logical_not(im._data, out=im._data)

        if outname:
            im.write(outname, savemask='none')

        return im

    def get_source_mask(self, iden, center, size, minsize=None, dilate=None,
                        dtype=np.uint8, struct=None, unit_center=u.deg,
                        unit_size=u.arcsec, regrid_to=None, outname=None):
        if minsize is None:
            minsize = size

        im = self.img.subimage(center, size, minsize=minsize,
                               unit_center=unit_center, unit_size=unit_size)

        if isiter(iden):
            # combine the masks for multiple ids
            data = np.logical_or.reduce([(im._data == i) for i in iden])
        else:
            data = (im._data == iden)

        if dilate:
            data = dilate_mask(data, niter=dilate, struct=struct)

        if regrid_to:
            im._data = data.astype(float)
            im = regrid_to_image(im, regrid_to, size=size, order=0,
                                 inplace=True, antialias=False)
            data = np.around(im._data, out=im._data)

        im._data = data.astype(dtype)
        im._mask = np.ma.nomask

        self.logger.debug('source %s (%.5f, %.5f), extract mask (%d masked '
                          'pixels)', iden, center[1], center[0],
                          np.count_nonzero(im._data))
        if outname:
            im.write(outname, savemask='none')

        return im

    def align_with_image(self, other, inplace=False, truncate=False):
        """Rotate and truncate the segmap to match 'other'."""
        out = self if inplace else self.copy()
        rot = other.wcs.get_rot() - self.img.wcs.get_rot()
        if np.abs(rot) > 1.e-3:
            out.img = self.img.rotate(rot, reshape=True, regrid=True,
                                      flux=False, order=0, inplace=inplace)

        if truncate:
            pixsky = other.wcs.pix2sky([[-1, -1],
                                        [other.shape[0], -1],
                                        [-1, other.shape[1]],
                                        [other.shape[0], other.shape[1]]],
                                       unit=u.deg)
            pixcrd = out.img.wcs.sky2pix(pixsky)
            ymin, xmin = pixcrd.min(axis=0)
            ymax, xmax = pixcrd.max(axis=0)
            out.img.truncate(ymin, ymax, xmin, xmax, mask=False,
                             unit=None, inplace=True)

        out.img._data = np.around(out.img._data).astype(int)
        return out

    def cmap(self, background_color='#000000'):
        """matplotlib colormap with random colors.
        (taken from photutils' segmentation map class)"""
        return get_cmap(self.img.data.max() + 1,
                        background_color=background_color)


def dilate_mask(data, thres=0.5, niter=1, struct=None):
    if struct is None:
        struct = ndi.generate_binary_structure(2, 1)
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(0)
    maxval = data.max()
    if maxval != 1:
        data /= maxval
        data = data > 0.5
    return ndi.binary_dilation(data, structure=struct, iterations=niter)


def get_cmap(ncolors, background_color='#000000'):
    from matplotlib import colors
    prng = np.random.RandomState(42)
    h = prng.uniform(low=0.0, high=1.0, size=ncolors)
    s = prng.uniform(low=0.2, high=0.7, size=ncolors)
    v = prng.uniform(low=0.5, high=1.0, size=ncolors)
    hsv = np.dstack((h, s, v))
    rgb = np.squeeze(colors.hsv_to_rgb(hsv))
    cmap = colors.ListedColormap(rgb)

    if background_color is not None:
        cmap.colors[0] = colors.hex2color(background_color)

    return cmap
