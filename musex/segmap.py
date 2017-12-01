import astropy.units as u
import numpy as np
from mpdaf.obj import Image
from scipy import ndimage as ndi


class SegMap:
    """
    Handle segmentation maps, where pixel values are sources ids.
    """

    def __init__(self, path=None, data=None, cut_header_after='D001VER'):
        self.path = path
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
        return self.__class__(path=self.path, data=self.img.copy())

    def get_mask(self, value, dtype=np.uint8, dilate=None, inverse=False,
                 struct=None):
        if inverse:
            data = (self.img._data != value)
        else:
            data = (self.img._data == value)
        if dilate:
            data = dilate_mask(data, niter=dilate, struct=struct)
        return Image.new_from_obj(self.img, data.astype(dtype))

    def get_source_mask(self, iden, center, size, minsize=None, dilate=None,
                        dtype=np.uint8, struct=None, unit_center=u.deg,
                        unit_size=u.arcsec):
        if minsize is None:
            minsize = size
        im = self.img.subimage(center, size, minsize=minsize,
                               unit_center=unit_center, unit_size=unit_size)
        data = (im._data == iden)
        if dilate:
            data = dilate_mask(data, niter=dilate, struct=struct)
        im.data = data.astype(dtype)
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
            dec_min, ra_min = pixsky.min(axis=0)
            dec_max, ra_max = pixsky.max(axis=0)
            out.img.truncate(dec_min, dec_max, ra_min, ra_max, mask=False,
                             unit=u.deg, inplace=True)

        out.img.data = np.around(out.img.data).astype(int)
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
