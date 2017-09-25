import logging
import os
from astropy.table import Table, Column
from astropy.utils.console import ProgressBar
from astropy.utils.decorators import lazyproperty
from collections import OrderedDict

from .extractor import Extractor
from .settings import db

DIRNAME = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)

__all__ = ['PriorExtractor']


class PriorExtractor(Extractor):

    def ingest_catalog(self):
        logger.info('ingesting catalog from %s', self._conf['catalog'])
        cat = Table.read(self._conf['catalog'])
        if self._conf.get('limit'):
            logger.info('keeping only %d rows', self._conf['limit'])
            cat = cat[:self._conf['limit']]

        # TODO: Use ID as primary index
        cat.rename_column('ID', 'RAF_ID')
        cat.add_column(Column(name='version',
                              data=[self._conf['version']] * len(cat)),
                       index=1)

        table = db[self.name]
        colnames = cat.colnames
        # TODO: skip already inserted sources
        for row in ProgressBar(cat):
            table.insert(OrderedDict(zip(colnames, row.as_void())))
        # table.insert_many((OrderedDict(zip(cat.colnames, row.as_void()))
        #                    for row in cat))
        table.create_index('RAF_ID')
        logger.info('%d rows ingested', len(cat))

    @lazyproperty
    def segmap(self):
        im = Image(self.conf['segmap'], copy=False)
        idx = im.data_header.index('D001VER')
        im.data_header = im.data_header[:idx]
        return im
