import dataset
import logging
import os
import yaml
from sqlalchemy.engine import Engine
from sqlalchemy import event


__all__ = ('load_yaml_config', 'load_db')


def load_yaml_config(filename):
    """Load a YAML config file, with string substitution."""
    with open(filename, 'r') as f:
        conftext = f.read()
        conf = yaml.load(conftext)
        return yaml.load(conftext.format(**conf))


def load_db(filename, **kwargs):
    """Open a sqlite database with dataset."""

    debug = os.getenv('SQLDEBUG')
    if debug is not None:
        logging.getLogger(__name__).info('Activate debug mode')
        kwargs.setdefault('engine_kwargs', {})
        kwargs['engine_kwargs']['echo'] = True
    # if not verbose:
    #     dataset.persistence.database.log.addHandler(logging.NullHandler())
    db = dataset.connect('sqlite:///{}'.format(filename), **kwargs)

    @event.listens_for(Engine, 'connect')
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute('PRAGMA foreign_keys = ON')
        cursor.execute('PRAGMA cache_size = -100000')
        cursor.execute('PRAGMA journal_mode = WAL')
        cursor.close()

    return db
