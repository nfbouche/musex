import logging
import sys
from mpdaf.log import setup_logging

from .datasets import load_datasets
from .catalog import load_catalogs
from .settings import conf
from .version import __version__, __description__

setup_logging(__name__, level=logging.DEBUG, stream=sys.stdout)

datasets = load_datasets(conf)
catalogs = load_catalogs(conf)

if conf['show_banner']:
    banner = f"""
MUSEX, {__description__} - v{__version__}

datasets: {', '.join(datasets.keys())}
catalogs: {', '.join(catalogs.keys())}
"""
    # banner += textwrap.indent('\n'.join(datasets.keys()), '- ')
    print(banner)
