import importlib
import logging
import os
from astropy.table import Table, Column
from astropy.utils.console import ProgressBar
from astropy.utils.decorators import lazyproperty
from collections import OrderedDict

from .settings import isnotebook

DIRNAME = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)

__all__ = ['load_catalogs', 'Catalog', 'PriorCatalog']


def load_catalogs(settings, db):
    catalogs = {}
    for name, conf in settings['catalogs'].items():
        mod, class_ = conf['class'].rsplit('.', 1)
        mod = importlib.import_module(mod)
        catalogs[name] = getattr(mod, class_)(name, conf, db)
    return catalogs


class ResultSet(list):

    def __init__(self, results, whereclause=None):
        self.results = list(results)
        self.whereclause = whereclause

    def __repr__(self):
        return (f'<{self.__class__.__name__}({self.whereclause})>, '
                f'{len(self)} results')

    def __len__(self):
        return len(self.results)

    def as_table(self):
        t = Table(data=self.results, names=self.results[0].keys())
        if '_id' in t.columns:
            t.remove_column('_id')
        return t

    def as_sourcelist(self):
        raise NotImplementedError


class Catalog:

    def __init__(self, name, settings, db):
        self.name = name
        self.settings = settings
        self.db = db
        self.logger = logging.getLogger(__name__)
        for key in ('catalog', 'id_name', 'version'):
            setattr(self, key, self.settings.get(key))

    @lazyproperty
    def table(self):
        return self.db.create_table(self.name, primary_id='_id')

    def ingest_catalog(self, limit=None):
        logger.info('ingesting catalog %s', self.catalog)
        cat = Table.read(self.catalog)
        if limit:
            logger.info('keeping only %d rows', limit)
            cat = cat[:limit]

        # TODO: Use ID as primary key ?
        cat.add_column(Column(name='version', data=[self.version] * len(cat)))

        table = self.table
        colnames = cat.colnames
        count_inserted = 0
        count_updated = 0
        rows = list(zip(*[c.tolist() for c in cat.columns.values()]))
        with self.db as tx:
            table = tx[self.name]
            for row in ProgressBar(rows, ipython_widget=isnotebook()):
                res = table.upsert(OrderedDict(zip(colnames, row)),
                                   [self.id_name, 'version'])
                if res is True:
                    count_updated += 1
                else:
                    count_inserted += 1

        table.create_index(self.id_name)
        logger.info('%d rows inserted, %s updated', count_inserted,
                    count_updated)

    @property
    def c(self):
        return self.table.table.c

    def select(self, whereclause=None, **params):
        return ResultSet(
            self.db.query(
                self.table.table.select(whereclause=whereclause, **params)),
            whereclause=whereclause
        )


class PriorCatalog(Catalog):

    @lazyproperty
    def segmap(self):
        from mpdaf.obj import Image
        im = Image(self.conf['segmap'], copy=False)
        idx = im.data_header.index('D001VER')
        im.data_header = im.data_header[:idx]
        return im
