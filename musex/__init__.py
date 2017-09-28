import logging
import os
import sys
from mpdaf.log import setup_logging

from .dataset import load_datasets
from .catalog import load_catalogs
from .settings import load_db, load_yaml_config
from .version import __version__, __description__

setup_logging(__name__, level=logging.DEBUG, stream=sys.stdout)


def init(settings_file=None, verbose=True):
    if settings_file is None:
        dirname = os.path.abspath(os.path.dirname(__file__))
        settings_file = os.path.join(dirname, 'udf', 'settings.yaml')

    conf = load_yaml_config(settings_file)
    db = load_db(conf['db'], verbose=True)
    datasets = load_datasets(conf)
    catalogs = load_catalogs(conf, db)

    if verbose and conf['show_banner']:
        banner = f"""
MUSEX, {__description__} - v{__version__}

datasets: {', '.join(datasets.keys())}
catalogs: {', '.join(catalogs.keys())}
    """
        # banner += textwrap.indent('\n'.join(datasets.keys()), '- ')
        print(banner)

    return datasets, catalogs
