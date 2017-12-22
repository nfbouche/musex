import logging
import numpy as np
import os
import sys
from astropy.table import Table
from mpdaf.obj import Image

from .dataset import load_datasets, MuseDataSet
from .catalog import load_input_catalogs, Catalog, ResultSet, table_to_odict
from .settings import load_db, load_yaml_config
from .source import SourceX
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
        self.workdir = self.conf['workdir']
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
                name, self.db, workdir=self.workdir, idname=row['idname'],
                raname=row['raname'], decname=row['decname'],
                segmap=row['segmap'])

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
        cat = Catalog(name, self.db, workdir=self.workdir,
                      idname=parent_cat.idname, raname=parent_cat.raname,
                      decname=parent_cat.decname, segmap=parent_cat.segmap)

        wherecl = resultset.whereclause
        if isinstance(resultset, Table):
            resultset = table_to_odict(resultset)

        cat.insert_rows(resultset)
        cat.update_meta(type='user', parent_cat=parent_cat.name,
                        segmap=parent_cat.segmap, query=wherecl)
        self.catalogs[name] = cat

    def to_sources(self, res_or_cat, size=5, srcvers='', apertures=None,
                   datasets=None, only_active=True):
        """Export a catalog or selection to sources (SourceX).

        Parameters
        ----------
        res_or_cat: `ResultSet` or `Catalog`
            Either a result from a query or a catalog to export.
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
        if isinstance(res_or_cat, Catalog):
            if only_active and 'active' in res_or_cat.c:
                resultset = res_or_cat.select(res_or_cat.c.active.isnot(False))
            else:
                resultset = res_or_cat.select()
        elif isinstance(res_or_cat, ResultSet):
            # TODO: filter active sources
            resultset = res_or_cat
        else:
            raise ValueError('invalid input for res_or_cat')

        cat = resultset.catalog
        origin = ('MuseX', __version__, '', '')

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
        idname, raname, decname = cat.idname, cat.raname, cat.decname

        for row in resultset:
            src = SourceX.from_data(row[idname], row[raname], row[decname],
                                    origin)
            src.SRC_V = srcvers
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
            debug('center: (%.5f, %.5f) -> (%.2f, %.2f)', *center,
                  *centerpix.tolist())
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
            yield src

    def export_sources(self, res_or_cat, create_pdf=False, outdir=None,
                       outname='source-{src.ID:05d}', **kwargs):
        """Save a catalog or selection to sources (SourceX).

        See `MuseX.to_sources` for the additional arguments.

        Parameters
        ----------
        res_or_cat: `ResultSet` or `Catalog`
            Either a result from a query or a catalog to export.
        create_pdf: bool
            If True, create a pdf for each source.
        outdir: str
            Output directory. If None the default is
            `'{self.workdir}/export/{cname}/{self.muse_dataset.name}'`.
        outdir: str
            Output directory. If None the default is `'source-{src.ID:05d}'`.

        """
        if outdir is None:
            if isinstance(res_or_cat, Catalog):
                cname = res_or_cat.name
            elif isinstance(res_or_cat, ResultSet):
                cname = res_or_cat.catalog.name
            else:
                raise ValueError('invalid input for res_or_cat')
            outdir = f'{self.workdir}/export/{cname}/{self.muse_dataset.name}'
        os.makedirs(outdir, exist_ok=True)

        try:
            conf = self.conf['export']['pdf']
        except KeyError:
            conf = {}
        white = self.muse_dataset.white
        info = self.logger.info
        ima2 = conf.get('image', 'HST_F775W')

        for src in self.to_sources(res_or_cat, **kwargs):
            outn = outname.format(src=src)
            fname = f'{outdir}/{outn}.fits'
            src.write(fname)
            info('fits written to %s', fname)
            if create_pdf:
                fname = f'{outdir}/{outn}.pdf'
                src.to_pdf(fname, white, ima2=ima2)
                info('pdf written to %s', fname)

    def delete_user_cat(self, name):
        if name not in self.db.tables or name not in self.catalogs:
            raise ValueError('not a valid catalog name')
        self.catalogs[name].drop()
        del self.catalogs[name]
