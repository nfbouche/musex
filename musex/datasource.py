import logging
from astropy.utils.decorators import lazyproperty
from mpdaf.obj import Image, Cube


def load_datasources(settings):
    datasources = {}
    for name, conf in settings['datasources'].items():
        datasources[name] = DataSource(name, settings=conf)
    return datasources


class DataSource:

    def __init__(self, name, settings):
        self.name = name
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    # @lazyproperty
    # def settings(self):
    #     from .settings import load_yaml_config
    #     return load_yaml_config(self.settings)

    @lazyproperty
    def images(self):
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', AstropyWarning)
        # TODO ? strip header
        return {k: Image(v, copy=False)
                for k, v in self.settings['images'].items()}

    @lazyproperty
    def datacube(self):
        return Cube(self.settings['datacube'], copy=False)

    @lazyproperty
    def expcube(self):
        return Cube(self.settings['expcube'], copy=False)

    @lazyproperty
    def expima(self):
        return Image(self.settings['expima'], copy=False)
