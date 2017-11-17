import logging
import os
import sys
from collections import OrderedDict

from .dataset import load_datasets, MuseDataSet
from .catalog import load_input_catalogs, Catalog
from .settings import load_db, load_yaml_config
from .source import SourceListX
from .version import __version__, __description__


class MuseX:
    """
    TODO:
    - mean to choose catalog
    - save history of operations in source

    """

    def __init__(self, settings_file=None, muse_dataset=None, **kwargs):
        if settings_file is None:
            settings_file = os.path.expanduser('~/.musex/settings.yaml')
            if not os.path.exists(settings_file):
                dirname = os.path.abspath(os.path.dirname(__file__))
                settings_file = os.path.join(dirname, 'udf', 'settings.yaml')

        self.logger = logging.getLogger(__name__)
        self.logger.debug('Loading settings %s', settings_file)
        self.conf = load_yaml_config(settings_file)
        self.conf.update(kwargs)
        self.db = load_db(self.conf['db'])
        self.datasets = load_datasets(self.conf)
        self.input_catalogs = load_input_catalogs(self.conf, self.db)
        self.catalogs = {}
        # TODO: load catalogs

        settings = self.conf['muse_datasets']
        muse_dataset = muse_dataset or settings['default']
        self.muse_dataset = MuseDataSet(muse_dataset,
                                        settings=settings[muse_dataset])

        self.catalogs_table = self.db.create_table('catalogs')
        for row in self.catalogs_table.all():
            name = row['name']
            self.catalogs[name] = Catalog(name, self.db,
                                          workdir=self.conf['workdir'])

        if self.conf['show_banner']:
            self.info()

    def info(self, outstream=None):
        if outstream is None:
            outstream = sys.stdout
        outstream.write(r"""
  __  __               __  __
 |  \/  |_   _ ___  ___\ \/ /
 | |\/| | | | / __|/ _ \\  /
 | |  | | |_| \__ \  __//  \
 |_|  |_|\__,_|___/\___/_/\_\

""")
        outstream.write(f"""
{__description__} - v{__version__}

muse_dataset   : {self.muse_dataset.name}
datasets       : {', '.join(self.datasets.keys())}
input_catalogs : {', '.join(self.input_catalogs.keys())}
catalogs       : {', '.join(self.catalogs.keys())}
""")

    def preprocess(self, catalog_names=None, skip=True):
        for name in (catalog_names or self.input_catalogs):
            self.input_catalogs[name].preprocess(self.muse_dataset, skip=skip)

    def new_catalog_from_resultset(self, name, resultset,
                                   drop_if_exists=False):
        if name in self.db.tables:
            if drop_if_exists:
                self.db[name].drop()
            else:
                raise ValueError('table already exists')
        parent_cat = resultset.catalog
        cat = Catalog(name, self.db, workdir=self.conf['workdir'],
                      idname=parent_cat.idname, raname=parent_cat.raname,
                      decname=parent_cat.decname)
        cat.insert_rows(resultset)

        self.catalogs[name] = cat
        self.catalogs_table.insert(OrderedDict(
            name=name, creation_date='todo', parent_cat='todo'
        ))

    def export_resultset(self, resultset, size=5, srcvers=''):
        """Export a catalog selection (`ResultSet`) to a SourceList."""
        settings = self.conf['extraction']
        cat = resultset.catalog
        slist = SourceListX.from_coords(resultset, srcvers=srcvers,
                                        **cat.colnames)

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
            src.CATALOG = os.path.basename(cat.name)
            src.add_history('New source created',
                            author='')  # FIXME: how to get author here ?
            for ds in datasets:
                ds.add_to_source(src, size)

            cat.add_to_source(src, self.muse_dataset)
            src.extract_all_spectra(apertures=settings['apertures'])

        return slist
