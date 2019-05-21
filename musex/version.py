from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution('musex').version
except DistributionNotFound:
    # package is not installed
    pass

__description__ = 'The MUse Source EXtractor :)'
