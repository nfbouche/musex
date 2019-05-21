import logging
import os
from collections import OrderedDict

import dataset
import numpy as np
import yaml
from sqlalchemy import event, pool
from sqlalchemy.engine import Engine

import astropy.units as u
from mpdaf.obj import Image

__all__ = ('load_db', 'load_yaml_config', 'table_to_odict', 'extract_subimage')


def load_yaml_config(filename):
    """Load a YAML config file, with string substitution."""
    with open(filename, 'r') as f:
        conftext = f.read()
        conf = yaml.full_load(conftext)
        return yaml.full_load(conftext.format(**conf))


def load_db(filename=None, db_env=None, **kwargs):
    """Open a sqlite database with dataset."""

    kwargs.setdefault('engine_kwargs', {})

    if filename is not None:
        path = os.path.dirname(os.path.abspath(filename))
        if not os.path.isdir(path):
            raise ValueError(f'database path "{path}/" does not exist, you '
                             'should create it before running musered.')

        # Use a NullPool by default, which is sqlalchemy's default but dataset
        # uses instead a StaticPool.
        kwargs['engine_kwargs'].setdefault('poolclass', pool.NullPool)

        url = f'sqlite:///{filename}'
    elif db_env is not None:
        url = os.environ.get(db_env)
    else:
        raise ValueError('database url should be provided either with '
                         'filename or with db_env')

    logger = logging.getLogger(__name__)
    debug = os.getenv('SQLDEBUG')
    if debug is not None:
        logger.info('Activate debug mode')
        kwargs['engine_kwargs']['echo'] = True

    logger.debug('Connecting to %s', url)
    db = dataset.connect(url, **kwargs)

    if db.engine.driver == 'pysqlite':
        @event.listens_for(Engine, 'connect')
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute('PRAGMA foreign_keys = ON')
            cursor.execute('PRAGMA cache_size = -100000')
            # cursor.execute('PRAGMA journal_mode = WAL')
            cursor.close()

    return db


def table_to_odict(table):
    """Convert a `astropy.table.Table` to a list of `OrderedDict`."""
    colnames = table.colnames
    columns = []
    for c in table.columns.values():
        if c.dtype.kind == 'S':
            c = np.chararray.decode(c)
        columns.append(c.tolist())

    return [OrderedDict(zip(colnames, row)) for row in zip(*columns)]


def extract_subimage(im, center, size, minsize=None, unit_size=u.arcsec):
    if isinstance(im, str):
        im = Image(im, copy=False)

    if minsize is None:
        minsize = min(*size) // 2
    return im.subimage(center, size, minsize=minsize, unit_size=unit_size)
