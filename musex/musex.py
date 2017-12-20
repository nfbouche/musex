import logging
import numpy as np
import os
import sys
from astropy.table import Table
from mpdaf.obj import Image

from .dataset import load_datasets, MuseDataSet
from .catalog import load_input_catalogs, Catalog, ResultSet, table_to_odict
from .settings import load_db, load_yaml_config
from .source import SourceListX
from .utils import extract_subimage
from .version import __version__, __description__

LOGO = r"""
  __  __               __  __
  |  \/  |_   _ ___  ___\ \/ /
  | |\/| | | | / __|/ _ \\  /
  | |  | | |_| \__ \  __//  \
  |_|  |_|\__,_|___/\___/_/\_\

"""


class MuseX:
    """
    TODO:
    - mean to choose catalog
    - save history of operations in source

    """

    def __init__(self, settings_file=None, muse_dataset=None, **kwargs):
        if settings_file is None:
            settings_file = os.path.expanduser('~/.musex/settings.yaml')
            if not os.path.exists(settings_file):
                dirname = os.path.abspath(os.path.dirname(__file__))
                settings_file = os.path.join(dirname, 'udf', 'settings.yaml')

        self.logger = logging.getLogger(__name__)
        self.logger.debug('Loading settings %s', settings_file)
        self.conf = load_yaml_config(settings_file)
        self.conf.update(kwargs)
        self.db = load_db(self.conf['db'])

        # Load datasets
        self.datasets = load_datasets(self.conf)
        settings = self.conf['muse_datasets']
        muse_dataset = muse_dataset or settings['default']
        self.muse_dataset = MuseDataSet(muse_dataset,
                                        settings=settings[muse_dataset])

        # Load catalogs
        self.input_catalogs = load_input_catalogs(self.conf, self.db)
        self.catalogs = {}
        catalogs_table = self.db.create_table('catalogs')
        for row in catalogs_table.find(type='user'):
            name = row['name']
            self.catalogs[name] = Catalog(
                name, self.db, workdir=self.conf['workdir'],
                idname=row['idname'], raname=row['raname'],
                decname=row['decname'], segmap=row['segmap'])

        if self.conf['show_banner']:
            self.info()

    def info(self, outstream=None):
        if outstream is None:
            outstream = sys.stdout
        outstream.write(LOGO)
        outstream.write(f"""
{__description__} - v{__version__}

muse_dataset   : {self.muse_dataset.name}
datasets       : {', '.join(self.datasets.keys())}
input_catalogs : {', '.join(self.input_catalogs.keys())}
catalogs       : {', '.join(self.catalogs.keys())}
""")

    def new_catalog_from_resultset(self, name, resultset,
                                   drop_if_exists=False):
        """Create a new user catalog from a query result.

        Parameters
        ----------
        name: str
            Name of the catalog.
        resultset: ResultSet
            Result from a query.
        drop_if_exists: bool
            Drop the catalog if it already exists.

        """
        if not isinstance(resultset, (Table, ResultSet)):
            raise ValueError('unknown input type, resultset must be a '
                             'ResultSet or Table object')

        if name in self.db.tables:
            if name in self.input_catalogs or name not in self.catalogs:
                raise ValueError('a table with the same name already exists, '
                                 'and cannot be dropped since it is not a '
                                 'user catalog. Please choose another name.')
            if drop_if_exists:
                self.db[name].drop()
            else:
                raise ValueError('table already exists')

        parent_cat = resultset.catalog
        wherecl = resultset.whereclause
        cat = Catalog(name, self.db, workdir=self.conf['workdir'],
                      idname=parent_cat.idname, raname=parent_cat.raname,
                      decname=parent_cat.decname, segmap=parent_cat.segmap)

        if isinstance(resultset, Table):
            resultset = table_to_odict(resultset)

        cat.insert_rows(resultset)

        query = (str(wherecl.compile(compile_kwargs={"literal_binds": True}))
                 if wherecl is not None else None)
        cat.update_meta(type='user', parent_cat=parent_cat.name,
                        segmap=parent_cat.segmap, query=query)
        self.catalogs[name] = cat

    def export_catalog(self, catalog, only_active=True, **kwargs):
        """Export a catalog to a list of Sources. See `export_resultset` for
        the additional parameters.
        """
        if only_active and 'active' in catalog.c:
            resultset = catalog.select(catalog.c.active.isnot(False))
        else:
            resultset = catalog.select()
        return self.export_resultset(resultset, **kwargs)

    def export_resultset(self, resultset, size=5, srcvers='', apertures=None,
                         datasets=None):
        """Export a catalog selection (`ResultSet`) to a SourceList.

        Parameters
        ----------
        size: float
            Size of the images (in arcseconds) added in the sources.
        srcvers: str
            Version of the sources (SRC_V).
        apertures: list of float
            List of aperture radii for spectra extraction.
        datasets: list of str
            List of dataset names to use for the sources. By default all
            datasets are used.

        """
        cat = resultset.catalog
        slist = SourceListX.from_coords(resultset, srcvers=srcvers,
                                        idname=cat.idname, raname=cat.raname,
                                        decname=cat.decname)

        # FIXME: not sure here, but the current dataset should be the same as
        # the one used to produce the masks
        assert cat.meta['dataset'] == self.muse_dataset.name

        debug = self.logger.debug
        info = self.logger.info
        info('Exporting %s sources with %s dataset, size=%.1f',
             len(resultset), self.muse_dataset.name, size)
        use_datasets = [self.muse_dataset]
        if datasets:
            if not isinstance(datasets, (list, tuple)):
                datasets = [datasets]
            use_datasets += [self.datasets[a] for a in datasets]
        else:
            use_datasets += list(self.datasets.values())

        try:
            parent_cat = self.input_catalogs[cat.meta['parent_cat']]
        except TypeError:
            parent_cat = cat

        # minsize = min(*size) // 2
        minsize = size // 2
        nskywarn = (50, 5)
        refskyf = resultset[0]['mask_sky']
        refskyim = Image(str(cat.workdir / refskyf), copy=False)

        for row, src in zip(resultset, slist):
            info('source %05d', src.ID)
            src.CATALOG = os.path.basename(parent_cat.name)
            src.add_history('New source created', author=self.conf['author'])
            for ds in use_datasets:
                ds.add_to_source(src, size)

            cat.add_segmap_to_source(src, parent_cat.extract,
                                     dataset=self.muse_dataset)
            parent_cat.add_to_source(src, parent_cat.extract)

            center = (src.DEC, src.RA)
            # If mask_sky is always the same, reuse it instead of reloading
            skyim = (refskyim if row['mask_sky'] == refskyf else
                     str(cat.workdir / row['mask_sky']))
            src.images['MASK_SKY'] = extract_subimage(
                skyim, center, (size, size), minsize=minsize)

            maskim = Image(str(cat.workdir / row['mask_obj']), copy=False)
            centerpix = maskim.wcs.sky2pix(center)[0]
            debug('center: %r -> %r', center, centerpix.tolist())
            src.images['MASK_OBJ'] = extract_subimage(
                maskim, center, (size, size), minsize=minsize)

            # compute surface of each masks and compare to field of view, save
            # values in header
            nsky = np.count_nonzero(src.images['MASK_SKY']._data)
            nobj = np.count_nonzero(src.images['MASK_OBJ']._data)
            nfracsky = 100.0 * nsky / np.prod(src.images['MASK_OBJ'].shape)
            nfracobj = 100.0 * nobj / np.prod(src.images['MASK_OBJ'].shape)
            min_nsky_abs, min_nsky_rel = nskywarn
            if nsky < min_nsky_abs or nfracsky < min_nsky_rel:
                self.logger.warning('sky mask is too small. Size is %d spaxel '
                                    'or %.1f %% of total area', nsky, nfracsky)

            src.add_attr('FSFMSK', cat.meta['convolve_fwhm'],
                         'Mask Conv Gauss FWHM in arcsec')
            src.add_attr('NSKYMSK', nsky, 'Size of MASK_SKY in spaxels')
            src.add_attr('FSKYMSK', nfracsky, 'Relative Size of MASK_SKY in %')
            src.add_attr('NOBJMSK', nobj, 'Size of MASK_OBJ in spaxels')
            src.add_attr('FOBJMSK', nfracobj, 'Relative Size of MASK_OBJ in %')
            # src.add_attr('MASKT1', thres[0], 'Mask relative threshold T1')
            # src.add_attr('MASKT2', thres[1], 'Mask relative threshold T2')
            # return nobj, nfracobj, nsky, nfracobj
            info('MASK_SKY: %.1f%%, MASK_OBJ: %.1f%%', nfracsky, nfracobj)

            src.extract_all_spectra(apertures=apertures)

        return slist

    def delete_user_cat(self, name):
        if name not in self.db.tables or name not in self.catalogs:
            raise ValueError('not a valid catalog name')
        self.catalogs[name].drop()
        del self.catalogs[name]
