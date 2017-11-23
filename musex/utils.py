import astropy.units as u
from mpdaf.obj import Image


def extract_subimage(im, center, size, minsize=None, unit_size=u.arcsec):
    if isinstance(im, str):
        im = Image(im, copy=False)

    minsize = minsize or min(*size) // 2
    return im.subimage(center, size, minsize=minsize, unit_size=unit_size)
