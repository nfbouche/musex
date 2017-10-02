import logging
import numpy as np
from mpdaf.sdetect import Source, SourceList

from .version import __version__


class SourceListX(SourceList):

    @classmethod
    def from_coords(cls, coords, origin=None, idname='ID'):
        origin = origin or ('MuseX', __version__, '', '')
        srclist = cls()
        for res in coords:
            src = cls.source_class.from_data(res[idname], res['RA'],
                                             res['DEC'], origin)
            srclist.append(src)
        return srclist

    def add_datasets(self, muse_dataset, additional_datasets=None, size=5):
        logger = logging.getLogger(__name__)

        if additional_datasets is None:
            additional_datasets = []
        elif not isinstance(additional_datasets, (list, tuple)):
            additional_datasets = [additional_datasets]

        for src in self:
            # compute and add white light image
            logger.debug('Adding white light image')
            src.add_white_image(muse_dataset.cube, size)

            # set PA
            # src.set_pa(muse_dataset.cube)

            logger.debug('Adding Datacube')
            src.add_cube(muse_dataset.cube, f'{muse_dataset.prefix}_CUBE',
                         size=size, unit_wave=None)

            # add expmap image
            logger.debug('Adding expmap image')
            src.add_image(muse_dataset.expima, f'{muse_dataset.prefix}_EXPMAP')

            # compute average and dispersion value of expmap
            ima = src.images['MUSE_EXPMAP']
            src.EXPMEAN = (np.ma.mean(ima.data), 'Mean value of EXPMAP')
            src.EXPMIN = (np.ma.min(ima.data), 'Minimum value of EXPMAP')
            src.EXPMAX = (np.ma.max(ima.data), 'Maximum value of EXPMAP')
            logger.debug('Expmap mean %.2f min %.2f max %.2f',
                         src.EXPMEAN, src.EXPMIN, src.EXPMAX)

            for dataset in [muse_dataset] + additional_datasets:
                for name, img in getattr(dataset, 'images', {}).items():
                    tagname = getattr(img, 'name', name.upper())
                    logger.debug('Adding image %s', tagname)
                    src.add_image(img, f'{dataset.prefix}_{tagname}')
