from .catalog import *
from .crossmatching import *
from .dataset import *
from .musex import *
from .source import *
from .version import __description__, __version__


def _setup_logging():
    import logging
    import sys
    from mpdaf.log import setup_logging, clear_loggers
    logging.getLogger('alembic').setLevel('WARNING')
    clear_loggers('mpdaf')
    
    try:
        import pyplatefit
    except ImportError:
        pass
    else:
        clear_loggers('pyplatefit')
        
    setup_logging(name='', level='INFO', color=True, stream=sys.stdout,
                  fmt='%(levelname)s %(message)s')


_setup_logging()
