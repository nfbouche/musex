from astropy.table import Table
from collections import OrderedDict

from .settings import conf, db

tablename = conf['HSTPRIOR']['tablename']


def ingest_catalog(catname, tablename):
    cat = Table.read(catname)
    db[tablename].insert_many((OrderedDict(zip(cat.colnames, row.as_void()))
                               for row in cat))
