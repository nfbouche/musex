import logging
import numpy as np
from astropy.utils.decorators import lazyproperty
from mpdaf.obj import Image, Cube
from os.path import basename


def load_datasets(settings):
    datasets = {}
    for name, conf in settings['datasets'].items():
        datasets[name] = DataSet(name, settings=conf)
    return datasets


class DataSet:

    def __init__(self, name, settings):
        self.name = name
        self.settings = settings
        self.settings.setdefault('images', {})
        self.logger = logging.getLogger(__name__)
        for key in ('prefix', 'version'):
            setattr(self, key, self.settings.get(key))

    @lazyproperty
    def images(self):
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', AstropyWarning)
        # TODO ? strip header
        return {k: Image(v, copy=False)
                for k, v in self.settings['images'].items()}

    def add_to_source(self, src, size):
        names = list(self.images.keys())
        # white image is handled separately
        if 'WHITE' in names:
            names.remove('WHITE')
        if not names:
            return
        self.logger.debug('Adding images: %s', ','.join(names))

        for name, img in self.images.items():
            name = name.upper()
            tagname = getattr(img, 'name', name)
            order = 0 if name == 'SEGMAP' else 1
            src.add_image(img, f'{self.prefix}_{tagname}', rotate=True,
                          order=order)


class MuseDataSet(DataSet):

    @lazyproperty
    def cube(self):
        return Cube(self.settings['datacube'], copy=False)

    @lazyproperty
    def white(self):
        return Image(self.settings['white'], copy=False)

    # @lazyproperty
    # def expcube(self):
    #     return Cube(self.settings['expcube'], copy=False)

    @lazyproperty
    def expima(self):
        return Image(self.settings['expima'], copy=False)

    def add_to_source(self, src, size):
        # set PA: FIXME - useful ?
        # src.set_pa(self.cube)

        self.logger.debug('Adding Datacube and white light image')
        src.default_size = size
        src.SIZE = size
        src.add_cube(self.cube, f'{self.prefix}_CUBE',
                     size=size, unit_wave=None, add_white=True)
        src.CUBE = basename(self.settings['datacube'])
        src.CUBE_V = self.version

        # add expmap image + average and dispersion value of expmap
        src.add_image(self.expima, f'{self.prefix}_EXPMAP')
        ima = src.images[f'{self.prefix}_EXPMAP']
        src.EXPMEAN = (np.ma.mean(ima.data), 'Mean value of EXPMAP')
        src.EXPMIN = (np.ma.min(ima.data), 'Minimum value of EXPMAP')
        src.EXPMAX = (np.ma.max(ima.data), 'Maximum value of EXPMAP')
        self.logger.debug('Adding expmap image, mean %.2f min %.2f max %.2f',
                          src.EXPMEAN, src.EXPMIN, src.EXPMAX)

        # add fsf info
        if self.cube.primary_header.get('FSFMODE') == 'MOFFAT1':
            self.logger.debug('Adding FSF info from the datacube')
            src.add_FSF(self.cube, fieldmap=self.settings.get('fieldmap'))

        super().add_to_source(src, size)
