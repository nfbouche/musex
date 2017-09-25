import importlib
import logging
import os
from astropy.utils.decorators import lazyproperty
from mpdaf.obj import Image, Cube

from .settings import load_yaml_config, conf_dir


def load_datasources(settings):
    datasources = {}
    for name, ext in settings['datasources'].items():
        config_file = os.path.join(conf_dir, ext['settings'])
        # mod, class_ = ext['class'].rsplit('.', 1)
        # mod = importlib.import_module(mod)
        # inst = getattr(mod, class_)(name, config_file=config_file)
        datasources[name] = DataSource(name, config_file=config_file)
    return datasources


class DataSource:

    def __init__(self, name, config_file=None):
        self.name = name
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)

    @lazyproperty
    def conf(self):
        return load_yaml_config(self.config_file)

    @lazyproperty
    def images(self):
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', AstropyWarning)
        # TODO ? strip header
        return {k: Image(v, copy=False)
                for k, v in self.conf['images'].items()}

    @lazyproperty
    def datacube(self):
        return Cube(self.conf['datacube'], copy=False)

    @lazyproperty
    def expcube(self):
        return Cube(self.conf['expcube'], copy=False)

    @lazyproperty
    def expima(self):
        return Image(self.conf['expima'], copy=False)
