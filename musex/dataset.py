import logging
import numpy as np
from astropy.utils.decorators import lazyproperty
from mpdaf.obj import Image, Cube
from os.path import basename

__all__ = ['DataSet', 'MuseDataSet']


def load_datasets(settings):
    """Load all datasets defined in the settings."""
    datasets = {}
    for name, conf in settings['datasets'].items():
        datasets[name] = DataSet(name, settings=conf)
    return datasets


class DataSet:
    """Manage a dataset defined in the settings file."""

    def __init__(self, name, settings):
        self.name = name
        self.settings = settings
        self.settings.setdefault('images', {})
        self.logger = logging.getLogger(__name__)
        for key in ('prefix', 'version'):
            setattr(self, key, self.settings.get(key))

    def __getstate__(self):
        state = self.__dict__.copy()
        # remove un-pickable objects
        state['logger'] = None
        return state

    def __setstate__(self, state):
        for slot, value in state.items():
            setattr(self, slot, value)
        self.logger = logging.getLogger(__name__)

    @lazyproperty
    def images(self):
        """Return a dictionary with the images."""
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', AstropyWarning)
        # TODO ? strip header
        return {k: Image(v, copy=False)
                for k, v in self.settings['images'].items()}

    def add_to_source(self, src, **kwargs):
        """Add stamp images to a source."""
        for name, img in self.images.items():
            name = name.upper()
            tagname = getattr(img, 'name', name)
            order = 0 if name == 'SEGMAP' else 1
            src.add_image(img, f'{self.prefix}_{tagname}', rotate=True,
                          order=order)


class MuseDataSet(DataSet):
    """Subclass from `DataSet` for MUSE datasets.

    In addition to images, it can also manage a datacube, white-light image,
    and exposure map image.

    """

    def __init__(self, name, settings):
        super().__init__(name, settings)
        margin = self.settings.get('margin')
        if margin is None:
            self.margin = 0  # default
        else:
            self.margin = margin

    @lazyproperty
    def cube(self):
        """The datacube."""
        return Cube(self.settings['datacube'], copy=False)

    @lazyproperty
    def white(self):
        """The white-light image."""
        return Image(self.settings['white'], copy=False)

    # @lazyproperty
    # def expcube(self):
    #     return Cube(self.settings['expcube'], copy=False)

    @lazyproperty
    def expima(self):
        """The exposure map image."""
        return Image(self.settings['expima'], copy=False)

    def add_to_source(self, src, **kwargs):
        """Add subcube and images to a source."""
        # set PA: FIXME - useful ?
        # src.set_pa(self.cube)

        src.add_cube(self.cube, f'{self.prefix}_CUBE',
                     size=src.SIZE, unit_wave=None, add_white=True)
        src.CUBE = basename(self.settings['datacube'])
        src.CUBE_V = self.version

        # add expmap image + average and dispersion value of expmap
        src.add_image(self.expima, f'{self.prefix}_EXPMAP', minsize=0.)
        ima = src.images[f'{self.prefix}_EXPMAP']
        src.EXPMEAN = (np.ma.mean(ima.data), 'Mean value of EXPMAP')
        src.EXPMIN = (np.ma.min(ima.data), 'Minimum value of EXPMAP')
        src.EXPMAX = (np.ma.max(ima.data), 'Maximum value of EXPMAP')

        # add fsf info
        if self.cube.primary_header.get('FSFMODE') == 'MOFFAT1':
            self.logger.info('Adding FSF info from the datacube')
            try:
                src.add_FSF(self.cube, fieldmap=self.settings.get('fieldmap'))
            except TypeError:
                self.logger.warning('Could not use fieldmap with MPDAF')
                # fieldmap arg not available in MPDAF 2.4
                src.add_FSF(self.cube)

        super().add_to_source(src, **kwargs)
