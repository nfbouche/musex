import logging
from mpdaf.log import setup_logging

from .extractor import load_extractors
from .settings import conf, conf_dir
from .version import __version__, __description__

setup_logging(__name__, level=logging.DEBUG)

_extractors = load_extractors(conf_dir, conf)
banner = f"""
MUSEX, {__description__} - v{__version__}

Available extractors:
"""
for _ext in _extractors:
    locals()[_ext.name] = _ext
    banner += f'- {_ext.name}\n'

if conf['show_banner']:
    print(banner)
