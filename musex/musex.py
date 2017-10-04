import logging
import os
from mpdaf.sdetect import Catalog

from .dataset import load_datasets
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

    def export_resultset(self, resultset, muse_dataset=None):
        settings = self.conf['extraction']
        slist = SourceListX.from_coords(resultset,
                                        **resultset.catalog.colnames)

        muse_dataset = muse_dataset or settings['dataset']
        add_datasets = settings.get('additional_datasets')
        if add_datasets:
            if not isinstance(add_datasets, (list, tuple)):
                add_datasets = [add_datasets]
            add_datasets = [self.datasets[a] for a in add_datasets]

        slist.add_datasets(self.datasets[muse_dataset],
                           additional_datasets=add_datasets,
                           extended_images=settings.get('extended_images'))

        if 'catalog' in settings:
            conf = settings['catalog']
            cat = self.catalogs[conf['from']]
            res = cat.select(columns=conf['columns'])
            scat = Catalog(rows=[list(r.values()) for r in res],
                           names=list(res[0].keys()))
            slist.add_catalog(scat, select_in_image=conf['select_in'],
                              name=conf['name'],
                              ra=cat.raname, dec=cat.decname)

        return slist
