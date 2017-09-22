import importlib
import logging
import os

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
        self._conf = None

    @property
    def conf(self):
        if self._conf is None and self.config_file is not None:
            self._conf = load_yaml_config(self.config_file)
        return self._conf
