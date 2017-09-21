import logging
import os
from astropy.table import Table, Column
from collections import OrderedDict

from .settings import db, load_yaml_config

DIRNAME = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)

__all__ = ['ingest_hstprior_data']


def ingest_hstprior_data():
    conf = load_yaml_config(os.path.join(DIRNAME, 'udf', 'hstprior.yaml'))
    cat = Table.read(conf['catalog'])
    cat.rename_column('ID', 'RAF_ID')
    cat.add_column(Column(name='version', data=[conf['version']] * len(cat)),
                   index=1)
    table = db.create_table(conf['tablename'])
    table.insert_many((OrderedDict(zip(cat.colnames, row.as_void()))
                       for row in cat))
    table.create_index('RAF_ID')
    logger.info('%d rows ingested', len(cat))
