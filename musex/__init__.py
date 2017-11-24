import logging
import sys
from mpdaf.log import setup_logging

from .musex import MuseX
from .segmap import SegMap
from .version import __version__, __description__

setup_logging(__name__, level=logging.DEBUG, stream=sys.stdout)
