import importlib
import logging
import os
from astropy.table import Table, Column
from astropy.utils.console import ProgressBar
from astropy.utils.decorators import lazyproperty
from collections import OrderedDict

from .settings import db

DIRNAME = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)

__all__ = ['load_catalogs', 'Catalog', 'PriorCatalog']


def load_catalogs(settings):
    catalogs = {}
    for name, conf in settings['catalogs'].items():
        mod, class_ = conf['class'].rsplit('.', 1)
        mod = importlib.import_module(mod)
        catalogs[name] = getattr(mod, class_)(name, conf)
    return catalogs


class Catalog:

    def __init__(self, name, settings):
        self.name = name
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        for key in ('catalog', 'id_name', 'version'):
            setattr(self, key, self.settings.get(key))

    @lazyproperty
    def table(self):
        return db.create_table(self.name, primary_id='_id')

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
        for row in ProgressBar(cat):
            res = table.upsert(OrderedDict(zip(colnames, row.as_void())),
                               [self.id_name, 'version'])
            if res is True:
                count_updated += 1
            else:
                count_inserted += 1

        table.create_index(self.id_name)
        logger.info('%d rows inserted, %s updated', count_inserted,
                    count_updated)


class PriorCatalog(Catalog):

    @lazyproperty
    def segmap(self):
        from mpdaf.obj import Image
        im = Image(self.conf['segmap'], copy=False)
        idx = im.data_header.index('D001VER')
        im.data_header = im.data_header[:idx]
        return im
