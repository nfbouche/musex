# -*- coding: utf-8 -*-

__version__ = '0.1.dev'
__description__ = 'The MUse Source EXtractor :)'

try:
    from ._githash import __githash__, __dev_value__
    if '.dev' in __version__:
        __version__ += __dev_value__
except Exception:
    pass
