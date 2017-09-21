import logging
from mpdaf.log import setup_logging

from .hstprior import *  # noqa
from .settings import *  # noqa
from .version import __version__, __description__

setup_logging(__name__, level=logging.DEBUG)

if conf['show_banner']:  # noqa
    print("""
    MUSEX, {} - v{}
    """.format(__description__, __version__))
