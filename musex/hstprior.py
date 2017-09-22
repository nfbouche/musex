import logging
import os
from astropy.table import Table, Column
from astropy.utils.console import ProgressBar
from collections import OrderedDict

from .extractor import Extractor
from .settings import db

DIRNAME = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)

__all__ = ['HstPriorExtractor']


class HstPriorExtractor(Extractor):

    def ingest_hstprior_data(self):
        cat = Table.read(self._conf['catalog'])
        cat.rename_column('ID', 'RAF_ID')
        cat.add_column(Column(name='version',
                              data=[self._conf['version']] * len(cat)),
                       index=1)

        # table = db.create_table(self._conf['tablename'])
        table = db[self.name]
        colnames = cat.colnames
        for row in ProgressBar(cat):
            table.insert(OrderedDict(zip(colnames, row.as_void())))
        # table.insert_many((OrderedDict(zip(cat.colnames, row.as_void()))
        #                    for row in cat))
        table.create_index('RAF_ID')
        logger.info('%d rows ingested', len(cat))
