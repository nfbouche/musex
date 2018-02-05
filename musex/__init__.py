from .catalog import *
from .dataset import *
from .musex import *
from .segmap import *
from .source import *
from .version import __version__, __description__

# __all__ = ['MuseX', 'SegMap']


def _setup_logging():
    import logging
    import sys
    from mpdaf.log import setup_logging
    setup_logging('mpdaf', level=logging.INFO, stream=sys.stdout)
    setup_logging(__name__, level=logging.DEBUG, stream=sys.stdout)


_setup_logging()
