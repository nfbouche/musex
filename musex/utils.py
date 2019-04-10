import astropy.units as u
import dataset
import logging
import numpy as np
import os
import yaml

from mpdaf.obj import Image, moffat_image
from sqlalchemy.engine import Engine
from sqlalchemy import event, pool


def load_yaml_config(filename):
    """Load a YAML config file, with string substitution."""
    with open(filename, 'r') as f:
        conftext = f.read()
        conf = yaml.full_load(conftext)
        return yaml.full_load(conftext.format(**conf))


def load_db(filename=None, db_env=None, **kwargs):
    """Open a sqlite database with dataset."""

    kwargs.setdefault('engine_kwargs', {})

    if filename is not None:
        path = os.path.dirname(os.path.abspath(filename))
        if not os.path.isdir(path):
            raise ValueError(f'database path "{path}/" does not exist, you '
                             'should create it before running musered.')

        # Use a NullPool by default, which is sqlalchemy's default but dataset
        # uses instead a StaticPool.
        kwargs['engine_kwargs'].setdefault('poolclass', pool.NullPool)

        url = f'sqlite:///{filename}'
    elif db_env is not None:
        url = os.environ.get(db_env)
    else:
        raise ValueError('database url should be provided either with '
                         'filename or with db_env')

    logger = logging.getLogger(__name__)
    debug = os.getenv('SQLDEBUG')
    if debug is not None:
        logger.info('Activate debug mode')
        kwargs['engine_kwargs']['echo'] = True

    logger.debug('Connecting to %s', url)
    db = dataset.connect(url, **kwargs)

    if db.engine.driver == 'pysqlite':
        @event.listens_for(Engine, 'connect')
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute('PRAGMA foreign_keys = ON')
            cursor.execute('PRAGMA cache_size = -100000')
            # cursor.execute('PRAGMA journal_mode = WAL')
            cursor.close()

    return db


def isiter(val):
    try:
        iter(val)
    except TypeError:
        return False
    else:
        return True


def isnotebook():  # pragma: no cover
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def progressbar(*args, **kwargs):
    from tqdm import tqdm, tqdm_notebook
    func = tqdm_notebook if isnotebook() else tqdm
    return func(*args, **kwargs)


def extract_subimage(im, center, size, minsize=None, unit_size=u.arcsec):
    if isinstance(im, str):
        im = Image(im, copy=False)

    if minsize is None:
        minsize = min(*size) // 2
    return im.subimage(center, size, minsize=minsize, unit_size=unit_size)


def regrid_to_image(im, other, order=1, inplace=False, antialias=True,
                    size=None, unit_size=u.arcsec, **kwargs):
    im.data = im.data.astype(float)
    refpos = other.wcs.pix2sky([0, 0])[0]
    if size is not None:
        newdim = size / other.wcs.get_step(unit=unit_size)
    else:
        newdim = other.shape
    inc = other.wcs.get_axis_increments(unit=unit_size)
    im = im.regrid(newdim, refpos, [0, 0], inc, order=order,
                   unit_inc=unit_size, inplace=inplace, antialias=antialias)
    return im


def struct_from_moffat_fwhm(wcs, fwhm, psf_threshold=0.5, beta=2.5):
    """Compute a structuring element for the dilatation, to simulate
    a convolution with a psf."""
    # image size will be twice the full-width, to account for
    # psf_threshold < 0.5
    size = int(round(fwhm / wcs.get_step(u.arcsec)[0])) * 2 + 1
    if size % 2 == 0:
        size += 1

    psf = moffat_image(fwhm=(fwhm, fwhm), n=beta, peak=True,
                       wcs=wcs[:size, :size])

    # remove useless zeros on the edges.
    psf.mask_selection(psf._data < psf_threshold)
    psf.crop()
    assert tuple(np.array(psf.shape) % 2) == (1, 1)
    return ~psf.mask
