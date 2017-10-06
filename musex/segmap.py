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
        if cut_header_after:
            idx = self.img.data_header.index(cut_header_after)
            self.img.data_header = self.img.data_header[:idx]

    def get_mask(self, value):
        return Image.new_from_obj(self.img, self.img._data == value)

    def get_source_mask(self, iden, center, size, unit_center=u.deg,
                        unit_size=u.arcsec):
        im = self.img.subimage(center, size, minsize=size,
                               unit_center=unit_center, unit_size=unit_size)
        im.data = im._data == iden
        return im
