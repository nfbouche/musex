import logging
import textwrap
from mpdaf.log import setup_logging

from .datasource import load_datasources
from .settings import conf
from .version import __version__, __description__

setup_logging(__name__, level=logging.DEBUG)

datasources = load_datasources(conf)

if conf['show_banner']:
    banner = f"""
MUSEX, {__description__} - v{__version__}

Available datasources:
"""
    banner += textwrap.indent('\n'.join(datasources.keys()), '- ')
    print(banner)
