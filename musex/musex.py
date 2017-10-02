import logging
import os

from .dataset import load_datasets
from .catalog import load_catalogs
from .settings import load_db, load_yaml_config
from .version import __version__, __description__


class MuseX:

    def __init__(self, settings_file=None, **kwargs):
        if settings_file is None:
            dirname = os.path.abspath(os.path.dirname(__file__))
            settings_file = os.path.join(dirname, 'udf', 'settings.yaml')

        self.logger = logging.getLogger(__name__)
        self.logger.debug('Loading settings %s', settings_file)
        self.conf = load_yaml_config(settings_file)
        self.conf.update(kwargs)
        self.db = load_db(self.conf['db'])
        self.datasets = load_datasets(self.conf)
        self.catalogs = load_catalogs(self.conf, self.db)

        if self.conf['show_banner']:
            self.info()

    def info(self):
        print(f"""
MUSEX, {__description__} - v{__version__}

datasets: {', '.join(self.datasets.keys())}
catalogs: {', '.join(self.catalogs.keys())}
""")
