import logging
import numpy as np
from astropy.utils.decorators import lazyproperty
from mpdaf.obj import Image, Cube
from mpdaf.sdetect import Source
from os.path import basename

__all__ = ['DataSet', 'MuseDataSet']


def load_datasets(settings):
    """Load all datasets defined in the settings."""
    datasets = {}
    for name, conf in settings['datasets'].items():
        datasets[name] = DataSet(name, settings=conf)
    return datasets


class DataSet:
    """Manage a dataset defined in the settings file."""

    def __init__(self, name, settings):
        self.name = name
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        for key in ('prefix', 'version'):
            setattr(self, key, self.settings.get(key))

    def __repr__(self):
        out = f'<{self.__class__.__name__}('
        if self.prefix:
            out += f'prefix={self.prefix}, '
        if self.version:
            out += f'version={self.version}, '
        for k, v in self.settings.items():
            if k not in ('version', 'prefix'):
                if isinstance(v, dict):
                    if len(v) > 0:
                        out += f'{len(v)} {k}, '
                else:
                    out += f'1 {k}, '
        if out.endswith(', '):
            out = out[:-2]
        return out + ')>'

    def __getstate__(self):
        state = self.__dict__.copy()
        # remove un-pickable objects
        state['logger'] = None
        return state

    def __setstate__(self, state):
        for slot, value in state.items():
            setattr(self, slot, value)
        self.logger = logging.getLogger(__name__)

    @lazyproperty
    def images(self):
        """Return a dictionary with the images."""
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', AstropyWarning)
        # TODO ? strip header
        conf = self.settings.get('images', {})
        return {k: Image(v) for k, v in conf.items()}

    @lazyproperty
    def cubes(self):
        """Return a dictionary with the cubes."""
        conf = self.settings.get('cubes', {})
        return {k: Cube(v) for k, v in conf.items()}

    def get_skymask_file(self, id_):
        """Return the sky mask, optionally for a given ID.

        The mask file must be defined in the settings::

            masks:
                skymask: '{tmpdir}/masks/hdfs/mask-sky.fits'

        Or with a template for the mask of each source::

            masks:
                skymask_tpl: '{datadir}/origin_masks/sky-mask-%05d.fits'

        """
        masks = self.settings.get('masks', {})
        if 'skymask_tpl' in masks:
            return masks['skymask_tpl'] % id_
        elif 'skymask' in masks:
            return masks['skymask']

    def get_objmask_file(self, id_):
        """Return the source mask for a given ID.

        The mask file template must be defined in the settings::

            masks:
                mask_tpl: '{tmpdir}/masks/hdfs/mask-source-%05d.fits'

        """
        masks = self.settings.get('masks', {})
        if 'mask_tpl' in masks:
            return masks['mask_tpl'] % id_

    @property
    def src_path(self):
        return self.settings.get('sources')

    def get_source(self, id_):
        if self.src_path:
            return Source.from_file(self.src_path % id_)

    def add_to_source(self, src, names=None, **kwargs):
        """Add stamp images to a source."""

        debug = self.logger.debug
        # Images
        for name, img in self.images.items():
            name = name.upper()
            # tagname = getattr(img, 'name', name)
            if names is not None and name not in names:
                continue
            order = 0 if name == 'SEGMAP' else 1
            debug('Adding image: %s_%s', self.prefix, name)
            src.add_image(img, f'{self.prefix}_{name}', rotate=True,
                          order=order)

        # Cubes
        for name, cube in self.cubes.items():
            name = name.upper()
            if names is not None and name not in names:
                continue
            debug('Adding cube: %s_%s', self.prefix, name)
            src.add_cube(cube, f'{self.prefix}_{name}')

        # Sources
        s = self.get_source(src.ID)
        if s is not None:
            for ext in ('images', 'spectra', 'cubes', 'tables'):
                sdata = getattr(s, ext)
                srcdata = getattr(src, ext)
                for name, img in sdata.items():
                    if names is not None and name not in names:
                        continue
                    if name in srcdata:
                        debug('Not overriding %s from source %s', name, ext)
                    else:
                        debug('Adding source %s: %s', ext, name)
                        srcdata[name] = img


class MuseDataSet(DataSet):
    """Subclass from `DataSet` for MUSE datasets.

    In addition to images, it can also manage a datacube, white-light image,
    and exposure map image.

    """

    def __init__(self, name, settings):
        super().__init__(name, settings)
        self.margin = self.settings.get('margin', 0)

    @lazyproperty
    def cube(self):
        """The datacube."""
        return Cube(self.settings['datacube'], copy=False)

    @lazyproperty
    def white(self):
        """The white-light image."""
        return Image(self.settings['white'], copy=False)

    # @lazyproperty
    # def expcube(self):
    #     return Cube(self.settings['expcube'], copy=False)

    @lazyproperty
    def expima(self):
        """The exposure map image."""
        return Image(self.settings['expima'], copy=False)

    def add_to_source(self, src, names=None, **kwargs):
        """Add subcube and images to a source."""
        # set PA: FIXME - useful ?
        # src.set_pa(self.cube)

        src.add_cube(self.cube, f'{self.prefix}_CUBE',
                     size=src.SIZE, unit_wave=None, add_white=True)
        src.CUBE = basename(self.settings['datacube'])
        src.CUBE_V = self.version

        # add expmap image + average and dispersion value of expmap
        src.add_image(self.expima, f'{self.prefix}_EXPMAP', minsize=0.)
        ima = src.images[f'{self.prefix}_EXPMAP']
        src.EXPMEAN = (np.ma.mean(ima.data), 'Mean value of EXPMAP')
        src.EXPMIN = (np.ma.min(ima.data), 'Minimum value of EXPMAP')
        src.EXPMAX = (np.ma.max(ima.data), 'Maximum value of EXPMAP')

        # add fsf info
        if self.cube.primary_header.get('FSFMODE') == 'MOFFAT1':
            self.logger.info('Adding FSF info from the datacube')
            try:
                src.add_FSF(self.cube, fieldmap=self.settings.get('fieldmap'))
            except TypeError:
                self.logger.warning('Could not use fieldmap with MPDAF')
                # fieldmap arg not available in MPDAF 2.4
                src.add_FSF(self.cube)

        super().add_to_source(src, names=names, **kwargs)
