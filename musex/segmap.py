from mpdaf.obj import Image


class SegMap:
    """Handle segmentation maps where pixel values are sources ids.

    TODO:
    Add methods to a source mask, possibly with convolution with the FWHM.

    """

    def __init__(self, path, cut_header_after='D001VER'):
        self.path = path
        self.img = Image(path, copy=False)
        if cut_header_after:
            idx = self.img.data_header.index(cut_header_after)
            self.img.data_header = self.img.data_header[:idx]

    def create_sky_mask(self, outpath, sky_value=0):
        sky = Image.new_from_obj(self.img,
                                 self.img.data.filled(-1) == sky_value)
        sky.write(outpath, savemask='none')
