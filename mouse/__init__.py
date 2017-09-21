import logging
from mpdaf.log import setup_logging

from .hstprior import *  # noqa
from .settings import *  # noqa

setup_logging(__name__, level=logging.DEBUG)
