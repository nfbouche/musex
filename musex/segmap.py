import astropy.units as u
import numpy as np
from mpdaf.obj import Image


class SegMap:
    """Handle segmentation maps where pixel values are sources ids.

    TODO:
    Add methods to a source mask, possibly with convolution with the FWHM.

    """

    def __init__(self, path, cut_header_after='D001VER'):
        self.path = path
        self.img = Image(path, copy=False, mask=np.ma.nomask)
        if cut_header_after and cut_header_after in self.img.data_header:
            idx = self.img.data_header.index(cut_header_after)
            self.img.data_header = self.img.data_header[:idx]

    def get_mask(self, value, dtype=np.uint8):
        return Image.new_from_obj(
            self.img, (self.img._data == value).astype(dtype))

    def get_inverse_mask(self, value, dtype=np.uint8):
        return Image.new_from_obj(
            self.img, (self.img._data != value).astype(dtype))

    def get_source_mask(self, iden, center, size, minsize=None,
                        unit_center=u.deg, unit_size=u.arcsec):
        if minsize is None:
            minsize = size
        im = self.img.subimage(center, size, minsize=minsize,
                               unit_center=unit_center, unit_size=unit_size)
        im.data = (im._data == iden).astype(np.uint8)
        return im
