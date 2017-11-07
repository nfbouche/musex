import logging
import os

from .dataset import load_datasets, MuseDataSet
from .catalog import load_catalogs
from .settings import load_db, load_yaml_config
from .source import SourceListX
from .version import __version__, __description__


class MuseX:
    """
    TODO:
    - mean to choose catalog
    - save catalog name in source header (src.CATALOG)
    - save history of operations in source

    """

    def __init__(self, settings_file=None, muse_dataset=None, **kwargs):
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

        settings = self.conf['muse_datasets']
        muse_dataset = muse_dataset or settings['default']
        self.muse_dataset = MuseDataSet(muse_dataset,
                                        settings=settings[muse_dataset])

        if self.conf['show_banner']:
            self.info()

    def info(self):
        print(f"""
MUSEX, {__description__} - v{__version__}

muse:     {self.muse_dataset.name}
datasets: {', '.join(self.datasets.keys())}
catalogs: {', '.join(self.catalogs.keys())}
""")

    def preprocess(self, skip=True):
        for cat in self.catalogs.values():
            cat.preprocess(self.muse_dataset, skip=skip)

    def export_resultset(self, resultset, size=5):
        """Export a catalog selection (`ResultSet`) to a SourceList."""
        settings = self.conf['extraction']
        cat = resultset.catalog
        slist = SourceListX.from_coords(resultset, **cat.colnames)

        self.logger.info('Exporting results with %s dataset, size=%.1f',
                         self.muse_dataset.name, size)
        datasets = [self.muse_dataset]
        add_datasets = settings.get('additional_datasets')
        if add_datasets:
            if not isinstance(add_datasets, (list, tuple)):
                add_datasets = [add_datasets]
            datasets += [self.datasets[a] for a in add_datasets]

        for src in slist:
            self.logger.info('source %05d', src.ID)
            for ds in datasets:
                ds.add_to_source(src, size)

            cat.add_to_source(src, self.muse_dataset)
            src.extract_all_spectra(apertures=settings['apertures'])

        return slist
