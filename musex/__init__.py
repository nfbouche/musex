import logging
import sys
from mpdaf.log import setup_logging

from .datasource import load_datasources
from .catalog import load_catalogs
from .settings import conf
from .version import __version__, __description__

setup_logging(__name__, level=logging.DEBUG, stream=sys.stdout)

datasources = load_datasources(conf)
catalogs = load_catalogs(conf)

if conf['show_banner']:
    banner = f"""
MUSEX, {__description__} - v{__version__}

datasources: {', '.join(datasources.keys())}
catalogs: {', '.join(catalogs.keys())}
"""
    # banner += textwrap.indent('\n'.join(datasources.keys()), '- ')
    print(banner)
