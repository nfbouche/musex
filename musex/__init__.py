import logging
import sys
from mpdaf.log import setup_logging

from .datasource import load_datasources
from .detection import load_sourcefinders
from .settings import conf
from .version import __version__, __description__

setup_logging(__name__, level=logging.DEBUG, stream=sys.stdout)

datasources = load_datasources(conf)
sourcefinders = load_sourcefinders(conf)

if conf['show_banner']:
    banner = f"""
MUSEX, {__description__} - v{__version__}

datasources: {', '.join(datasources.keys())}
sourcefinders: {', '.join(sourcefinders.keys())}
"""
    # banner += textwrap.indent('\n'.join(datasources.keys()), '- ')
    print(banner)
