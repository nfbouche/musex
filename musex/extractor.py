import importlib
import logging
import os
from astropy.utils.decorators import lazyproperty
from mpdaf.obj import Image

from .settings import load_yaml_config


def load_extractors(settings_dir, settings):
    extractors = []
    for name, ext in settings['extractors'].items():
        mod, class_ = ext['class'].rsplit('.', 1)
        mod = importlib.import_module(mod)
        config_file = os.path.join(settings_dir, ext['settings'])
        inst = getattr(mod, class_)(name, config_file=config_file)
        extractors.append(inst)
    return extractors


class Extractor:

    def __init__(self, name, config_file=None):
        self.name = name
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)

    @lazyproperty
    def conf(self):
        return load_yaml_config(self.config_file)

    @lazyproperty
    def images(self):
        return {k: Image(v, copy=False)
                for k, v in self.conf['images'].items()}
