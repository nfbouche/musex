import importlib
import logging
import numpy as np
import os
import re
import sys
import textwrap
from astropy.table import Table
from collections import OrderedDict, Sized, Iterable
from contextlib import contextmanager
from joblib import delayed, Parallel
from mpdaf.log import setup_logging
from mpdaf.obj import Image
from mpdaf.sdetect import create_masks_from_segmap, Catalog as MpdafCatalog
from mpdaf.tools import progressbar, isiter

from .dataset import load_datasets, MuseDataSet
from .catalog import (Catalog, InputCatalog, ResultSet, MarzCatalog,
                      IdMapping, get_result_table)
from .crossmatching import CrossMatch, gen_crossmatch
from .source import sources_to_marz, create_source
from .utils import load_db, load_yaml_config, table_to_odict
from .version import __version__, __description__

__all__ = ['MuseX']

LOGO = r"""
  __  __               __  __
  |  \/  |_   _ ___  ___\ \/ /
  | |\/| | | | / __|/ _ \\  /
  | |  | | |_| \__ \  __//  \
  |_|  |_|\__,_|___/\___/_/\_\

"""

# Default comments for FITS keywords
HEADER_COMMENTS = dict(
    CONFID='Z Confidence Flag',
    BLEND='Blending flag',
    DEFECT='Defect flag',
    REVISIT='Reconciliation Revisit Flag',
    TYPE='Object classification',
    REFSPEC='Name of reference spectra',
)


def _create_catalogs_table(db):
    """Create the 'catalogs' table which stores metadata about catalogs."""
    # Return the table if it exists
    if 'catalogs' in db:
        return db['catalogs']

    row = OrderedDict([
        ('name', ''),
        ('creation_date', ''),
        ('type', 'id'),
        ('parent_cat', ''),
        ('raname', 'RA'),
        ('decname', 'DEC'),
        ('idname', 'ID'),
        ('maxid', 1),
        ('query', ''),
    ])

    # Create the table
    table = db.create_table('catalogs')
    # Force the creation of the SQLATable
    assert table.table is not None
    # and make sure that all columns exists
    table._sync_columns(row, True)
    return table


class MuseX:
    """The main MuseX class.

    This class is the central part of MuseX, it gives access to all the
    catalogs and datasets, and provides some high-level operations.

    Parameters
    ----------
    settings_file: str
        Path of the settings file.
    muse_dataset: str
        Name of the Muse dataset to work on.
    id_mapping: str
        Name of a IdMapping catalog to use.

    """

    def __init__(self, settings_file, muse_dataset=None, id_mapping=None,
                 **kwargs):
        self.logger = logging.getLogger(__name__)
        self.settings_file = settings_file
        self.conf = load_yaml_config(settings_file)
        self.conf.update(kwargs)
        self.workdir = self.conf['workdir']
        self.conf.setdefault('export', {})
        self.conf['export'].setdefault('path', f'{self.workdir}/export')
        self.db = load_db(filename=self.conf.get('db'),
                          db_env=self.conf.get('db_env'))

        # Creating the IdMapping object if required
        self.id_mapping = None
        id_mapping = id_mapping or self.conf.get('idmapping')
        if id_mapping:
            self.create_id_mapping(id_mapping)

        # Table to store history of operations
        self.history = self.db.create_table('history', primary_id='_id')

        # Load datasets
        self.datasets = load_datasets(self.conf)
        settings = self.conf['muse_datasets']
        muse_dataset = muse_dataset or settings['default']
        self.muse_dataset = MuseDataSet(muse_dataset,
                                        settings=settings[muse_dataset])

        # Load catalogs table
        self.catalogs_table = _create_catalogs_table(self.db)
        self._load_input_catalogs()
        self._load_user_catalogs()

        # Marz
        self.marzcat = MarzCatalog('marz', self.db, primary_id='_id')
        # FIXME: handle properly version / revision
        self.marzcat.version = '1'

        if self.conf['show_banner']:
            self.info()

    def _load_input_catalogs(self):
        """Load input catalogs defined in the settings."""
        self.input_catalogs = {}
        for name, conf in self.conf['catalogs'].items():
            if 'class' in conf:
                mod, class_ = conf['class'].rsplit('.', 1)
                mod = importlib.import_module(mod)
                cls = getattr(mod, class_)
            else:
                cls = InputCatalog
            self.input_catalogs[name] = cls.from_settings(name, self.db,
                                                          **conf)
        self.logger.info("Input catalogs loaded")

    def _load_user_catalogs(self):
        """Load user generated catalogs."""
        self.catalogs = {}

        # User catalogs
        for row in self.catalogs_table.find(type='user'):
            name = row['name']
            self.catalogs[name] = Catalog(
                name, self.db, idname=row['idname'],
                raname=row['raname'], decname=row['decname'])
            # Restore the associated line catalog.
            line_tablename = f'{name}_lines'
            line_meta = self.catalogs_table.find_one(name=line_tablename)
            if line_meta:
                self.catalogs[name].create_lines(
                    line_idname=line_meta['idname'],
                    line_src_idname=line_meta['src_idname']
                )

        # Cross-match catalogs
        for row in self.catalogs_table.find(type='cross-match'):
            name = row['name']
            self.catalogs[name] = CrossMatch(name, self.db)
            cat1_name = self.catalogs[name].meta.get('cat1_name')
            cat2_name = self.catalogs[name].meta.get('cat2_name')
            if cat1_name is not None:
                try:
                    self.catalogs[name].cat1 = self.catalogs[cat1_name]
                except KeyError:
                    self.catalogs[name].cat1 = self.input_catalogs[cat1_name]
            if cat2_name is not None:
                try:
                    self.catalogs[name].cat2 = self.catalogs[cat2_name]
                except KeyError:
                    self.catalogs[name].cat2 = self.input_catalogs[cat2_name]

        self.logger.info("User catalogs loaded")

    def set_loglevel(self, level):
        """Set the logging level for the root logger."""
        logger = logging.getLogger('')
        level = level.upper()
        logger.setLevel(level)
        logger.handlers[0].setLevel(level)

    @contextmanager
    def use_loglevel(self, level):
        """Context manager to set the logging level for the root logger."""
        logger = logging.getLogger('')
        level = level.upper()
        oldlevel = logger.getEffectiveLevel()
        logger.setLevel(level)
        logger.handlers[0].setLevel(level)
        yield
        logger.setLevel(oldlevel)
        logger.handlers[0].setLevel(oldlevel)

    def info(self, outstream=None):
        """Print all available information."""
        if outstream is None:
            outstream = sys.stdout
        outstream.write(LOGO)
        outstream.write(textwrap.dedent(f"""
            {__description__} - v{__version__}

            settings file  : {self.settings_file}
            muse_dataset   : {self.muse_dataset.name}
            datasets       : {', '.join(self.datasets.keys())}
            input_catalogs : {', '.join(self.input_catalogs.keys())}
            catalogs       : {', '.join(self.catalogs.keys())}
            """))
        if self.id_mapping is not None:
            outstream.write(f"id_mapping     : {self.id_mapping.name}")

    @property
    def exportdir(self):
        """The directory where files are exported."""
        exportdir = self.conf['export']['path']
        return f'{exportdir}/{self.muse_dataset.name}'

    def find_parent_cat(self, cat):
        """Find the parent catalog of a given catalog."""
        current_cat = cat
        while True:
            parent = current_cat.meta['parent_cat']
            if parent is None:
                parent_cat = current_cat
                break
            elif parent in self.input_catalogs:
                parent_cat = self.input_catalogs[parent]
                break
            elif parent in self.catalogs:
                current_cat = self.catalogs[parent]
            else:
                raise ValueError('parent catalog not found')
        return parent_cat

    def new_catalog(self, name, idname='ID', raname='RA', decname='DEC',
                    drop_if_exists=False):
        """Create a new user catalog.

        Parameters
        ----------
        name: str
            Name of the catalog.
        idname: str
            Name of the 'id' column.
        raname: str
            Name of the 'ra' column.
        decname: str
            Name of the 'dec' column.
        drop_if_exists: bool
            Drop the catalog if it already exists.

        """
        if name in self.db.tables:
            if name in self.input_catalogs or name not in self.catalogs:
                raise ValueError('a table with the same name already exists, '
                                 'and cannot be dropped since it is not a '
                                 'user catalog. Please choose another name.')
            if drop_if_exists:
                self.db[name].drop()
            else:
                raise ValueError('table already exists')

        self.catalogs[name] = Catalog(name, self.db, idname=idname,
                                      raname=raname, decname=decname)
        return self.catalogs[name]

    def new_catalog_from_resultset(self, name, resultset, primary_id=None,
                                   drop_if_exists=False):
        """Create a new user catalog from a query result.

        Parameters
        ----------
        name: str
            Name of the catalog.
        resultset: `musex.ResultSet`
            Result from a query.
        primary_id: str
            The primary id for the SQL table, must be a column name. Defaults
            to 'ID'.
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
        whereclause = resultset.whereclause
        self.catalogs[name] = cat = Catalog.from_parent_cat(
            parent_cat, name, whereclause, primary_id=primary_id)

        if isinstance(resultset, Table):
            resultset = table_to_odict(resultset)
        cat.insert(resultset)
        return cat

    def create_id_mapping(self, name):
        """Create or get an IdMapping object."""
        self.id_mapping = IdMapping(name, self.db)

    def create_masks_from_segmap(self, cat, maskdir, limit=None, n_jobs=-1,
                                 skip_existing=True, margin=0,
                                 mask_size=(20, 20), convolve_fwhm=0):
        """Create binary masks from a segmentation map.

        Parameters
        ----------
        cat : `Catalog`
            The catalog with sources for which masks are computed.
        maskdir : str
            The output directory.
        n_jobs : int
            Number of parallel processes (for joblib).
        skip_existing : bool
            If True, skip sources for which the mask file exists.
        margin : float
            Margin from the edges (pixels), for the sources selection and the
            segmap alignment.
        mask_size : tuple
            Size of the source masks (arcsec).
        convolve_fwhm : float
            FWHM for the PSF convolution (arcsec).

        """
        ref_image = self.muse_dataset.white
        catfile = cat.params['catalog']
        tbl = MpdafCatalog.read(catfile)
        self.logger.info('read catalog %s with %d sources', catfile, len(tbl))

        tbl = tbl.select(ref_image.wcs, ra=cat.raname, dec=cat.decname,
                         margin=margin)
        self.logger.info('selected %d sources in dataset footprint', len(tbl))

        if limit:
            tbl = tbl[:limit]

        os.makedirs(maskdir, exist_ok=True)
        create_masks_from_segmap(
            cat.params['segmap'], tbl[:10], ref_image, n_jobs=n_jobs,
            skip_existing=skip_existing,
            masksky_name=f'{maskdir}/mask-sky.fits',
            maskobj_name=f'{maskdir}/mask-source-%05d.fits',
            idname=cat.idname, raname=cat.raname, decname=cat.decname,
            margin=margin, mask_size=mask_size, convolve_fwhm=convolve_fwhm)

    def to_sources(self, res_or_cat, size=5, srcvers='', apertures=None,
                   datasets=None, only_active=True, refspec='MUSE_TOT_SKYSUB',
                   content=('parentcat', 'segmap', 'history'), verbose=False,
                   n_jobs=1, masks_dataset=None):
        """Export a catalog or selection to sources (SourceX).

        Parameters
        ----------
        res_or_cat : `ResultSet`, `Catalog`, `Table`
            Either a result from a query or a catalog to export.
        size : float or list of float
            Size of the images (in arcseconds) added in the sources.
        srcvers : str
            Version of the sources (SRC_V).
        apertures : list of float
            List of aperture radii for spectra extraction.
        datasets : iterable or dict
            List of dataset names to use for the sources, or dictionary with
            dataset names associated to a list of tags to use. By default all
            datasets are used, and all tags.
        masks_dataset : str
            Name of the dataset from which the source and sky masks are taken.
            If missing, no spectra will be extracted from the source cube.

        """
        if isinstance(res_or_cat, Catalog):
            if only_active and 'active' in res_or_cat.c:
                resultset = res_or_cat.select(res_or_cat.c.active.isnot(False))
            else:
                resultset = res_or_cat.select()
        elif isinstance(res_or_cat, (ResultSet, Table)):
            resultset = res_or_cat
        else:
            raise ValueError('invalid input for res_or_cat')

        if isinstance(resultset, ResultSet):
            # To simplify things below, make sure we have a Table
            # TODO: filter active sources
            resultset = resultset.as_table()

        nrows = len(resultset)
        cat = resultset.catalog
        parent_cat = self.find_parent_cat(cat)

        debug = self.logger.debug
        info = self.logger.info

        if isinstance(size, (float, int)):
            info('Exporting %s sources with %s dataset, size=%.1f',
                 nrows, self.muse_dataset.name, size)
            size = [size] * nrows
        elif isinstance(size, Iterable) and isinstance(size, Sized):
            if nrows != len(size):
                raise ValueError(f"Length of res_or_cat ({nrows}) does not "
                                 f"match length of size ({len(size)})")

            info('Exporting %s sources with %s dataset, %.1f<=size<=%.1f',
                 nrows, self.muse_dataset.name, np.min(size), np.max(size))
        else:
            raise ValueError("'size' should be a float or list of floats")

        use_datasets = {self.muse_dataset: None}
        if datasets is not None:
            if isinstance(datasets, dict):
                use_datasets.update({self.datasets[name]: val
                                     for name, val in datasets.items()})
            elif isiter(datasets):
                use_datasets.update({self.datasets[name]: None
                                     for name in datasets})
        else:
            for ds in self.datasets.values():
                if ds.linked_cat and ds.linked_cat != parent_cat.name:
                    # if the dataset is linked to a catalog which is not the
                    # one used here, skip it
                    continue
                use_datasets[ds] = None

        info('Using datasets: %s', ','.join(ds.name for ds in use_datasets))

        idname, raname, decname = cat.idname, cat.raname, cat.decname
        author = self.conf['author']

        segmap = parent_cat.params.get('segmap')
        if segmap:
            segmap = Image(segmap)
            segmap_tag = parent_cat.params['extract']['prefix'] + "_SEGMAP"

        redshifts = self.conf['export'].get('redshifts', {})
        header_columns = self.conf['export'].get('header_columns', {})
        for key, colname in header_columns.items():
            if colname not in resultset.colnames:
                self.logger.warning(
                    "'%s' column not found, though it is specified in the "
                    "settings file for the %s keyword", colname, key)

        if masks_dataset is not None:
            maskds = self.datasets[masks_dataset]
        else:
            info('no masks specified, spectra will not be extracted')
            maskds = None

        rows = [dict(zip(row.colnames, row.as_void().tolist()))
                for row in resultset]
        to_compute = []
        for row, src_size in zip(rows, size):
            id_ = row[idname]
            if maskds:
                skyim = maskds.get_skymask_file(id_)
                maskim = maskds.get_objmask_file(id_)
            else:
                skyim, maskim = None, None

            args = (id_, row[raname], row[decname], src_size, skyim,
                    maskim, use_datasets, apertures, verbose)
            to_compute.append(delayed(create_source)(*args))

        if verbose is False:
            to_compute = progressbar(to_compute)
            setup_logging('musex', level=logging.WARNING, stream=sys.stdout)

        sources = Parallel(n_jobs=n_jobs,
                           verbose=50 if verbose else 0)(to_compute)

        if verbose is False:
            rows = progressbar(rows)

        for row, src, src_size in zip(rows, sources, size):
            src.SRC_V = (srcvers, 'Source Version')
            info('source %05d (%.5f, %.5f)', src.ID, src.DEC, src.RA)
            src.CATALOG = os.path.basename(parent_cat.name)
            # src.add_attr('FSFMSK', cat.meta['convolve_fwhm'],
            #              'Mask Conv Gauss FWHM in arcsec')

            # Add keywords from columns
            for key, colname in header_columns.items():
                if row.get(colname) is not None:
                    if key == 'COMMENT':
                        # truncate comment if too long
                        com = re.sub(r'[^\s!-~]', '', row[colname])
                        for txt in textwrap.wrap(com, 50):
                            src.add_comment(txt, '')
                    else:
                        debug('Add %s=%r', key, row[colname])
                        comment = HEADER_COMMENTS.get(key)
                        src.header[key] = (row[colname], comment)

            if src.header.get('REFSPEC') is None:
                self.logger.warning(
                    'REFSPEC column not found, using %s instead', refspec)
                src.add_attr('REFSPEC', refspec,
                             desc=HEADER_COMMENTS['REFSPEC'])

            # Add reshifts
            for key, colname in redshifts.items():
                if row.get(colname) is not None:
                    debug('Add redshift %s=%.2f', key, row[colname])
                    src.add_z(key, row[colname])

            if 'history' in content:
                for o in cat.get_log(src.ID):
                    src.add_history(o['msg'], author=author, date=o['date'])
                src.add_history('source created', author=author)

            if 'segmap' in content and segmap:
                src.SEGMAP = segmap_tag
                src.add_image(segmap, segmap_tag, rotate=True, order=0)

            if 'parentcat' in content:
                parent_cat.add_to_source(src, row,
                                         **parent_cat.params['extract'])

            debug('IMAGES: %s', ', '.join(src.images.keys()))
            debug('SPECTRA: %s', ', '.join(src.spectra.keys()))
            yield src

        if verbose is False:
            # Reset logger
            # TODO: find a better way to do this!
            setup_logging('musex', level=logging.DEBUG, stream=sys.stdout)

    def export_sources(self, res_or_cat, create_pdf=False, outdir=None,
                       outname='source-{src.ID:05d}', **kwargs):
        """Save a catalog or selection to sources (SourceX).

        See `MuseX.to_sources` for the additional arguments.

        Parameters
        ----------
        res_or_cat: `ResultSet`, `Catalog`, `Table`
            Either a result from a query or a catalog to export.
        create_pdf: bool
            If True, create a pdf for each source.
        outdir: str
            Output directory. If None the default is
            `'{conf[export][path]}/{self.muse_dataset.name}/{cname}/sources'`.
        outdir: str
            Output directory. If None the default is `'source-{src.ID:05d}'`.

        """
        cat = get_result_table(res_or_cat)
        if outdir is None:
            outdir = f'{self.exportdir}/{cat.catalog.name}/sources'
        os.makedirs(outdir, exist_ok=True)

        pdfconf = self.conf['export'].get('pdf', {})
        info = self.logger.info

        for src in self.to_sources(res_or_cat, **kwargs):
            outn = outname.format(src=src)
            fname = f'{outdir}/{outn}.fits'
            src.write(fname)
            info('fits written to %s', fname)
            if create_pdf:
                fname = f'{outdir}/{outn}.pdf'
                src.to_pdf(fname, self.muse_dataset.white,
                           ima2=pdfconf.get('image'), mastercat=cat)
                info('pdf written to %s', fname)

    def export_marz(self, res_or_cat, outfile=None, export_sources=False,
                    datasets=None, sources_dataset=None, outdir=None,
                    srcname='source-{src.ID:05d}', skyspec='MUSE_SKY',
                    **kwargs):
        """Export a catalog or selection for MarZ.

        Pre-generated sources may be used by specifying the related dataset.
        Each object in the catalog (or the selection) must have an existing
        source, and each source must have a REFSPEC attribute designating the
        spectrum to use in MarZ.

        If the catalog (or the resultset) comes from ODHIN, the `source_tpl`
        points to the ODHIN group source and the catalog must contain
        a `group_id` column.

        Parameters
        ----------
        res_or_cat: `ResultSet` or `Catalog`
            Either a result from a query or a catalog to export.
        outfile: str
            Output file. If None the default is
            `'{conf[export][path]}/marz-{cat.name}-{muse_dataset.name}.fits'`.
        export_sources: bool
            If True, the source files are also exported.
        outdir: str
            Output directory. If None the default is
            `'{conf[export][path]}/{self.muse_dataset.name}/{cname}/marz'`.
        srcname: str
            Output directory. If None the default is `'source-{src.ID:05d}'`.

        """
        cat = get_result_table(res_or_cat)
        cname = cat.catalog.name
        if outdir is None:
            outdir = f'{self.exportdir}/{cname}/marz'
        os.makedirs(outdir, exist_ok=True)
        if outfile is None:
            outfile = f'{outdir}/marz-{cname}-{self.muse_dataset.name}.fits'

        # Keyword to check in the sources for ORIGIN.
        version_meta = cat.meta.get('version_meta', None)
        if version_meta is not None:
            check_keyword = (version_meta, cat.meta[version_meta])
        else:
            check_keyword = None

        srcdir = None
        if sources_dataset:
            ds = self.datasets[sources_dataset]

            def _src_func():
                for id_ in cat[cat.meta['idname']]:
                    yield ds.get_source(id_, check_keyword=check_keyword)
            src_iter = _src_func()
        else:
            if export_sources:
                # If sources must be exported, we need to tell .to_sources
                # what to put inside the sources (all datasets by default
                # and the segmap)
                content = ('segmap', 'parentcat')
                srcdir = f'{outdir}/{srcname}.fits'
            else:
                if datasets is None:
                    # skip using additional datasets. In this case the
                    # muse_dataset will be used, with spectra extracted
                    # from the cube.
                    datasets = []
                content = tuple()

            src_iter = self.to_sources(cat, datasets=datasets,
                                       content=content, **kwargs)

        sources_to_marz(src_iter, outfile, skyspec=skyspec, save_src_to=srcdir)

    def import_marz(self, catfile, catalog, **kwargs):
        """Import a MarZ catalog.

        Parameters
        ----------
        catfile: str
            The catalog to import from Marz.
        catalog: str or `Catalog`
            The MuseX catalog to which the results are attached.

        """
        if isinstance(catalog, Catalog):
            catalog = catalog.name
        if catalog not in self.catalogs:
            raise ValueError('catalog must be a valid user catalog')

        cat = Table.read(catfile, format='ascii', delimiter=',',
                         header_start=2, encoding='utf8')
        # The real ID column in Marz is "Name"
        cat.remove_column('ID')
        cat.rename_column('Name', 'ID')
        cat['catalog'] = catalog
        keys = ['ID', 'version', 'catalog']
        self.marzcat.ingest_input_catalog(catalog=cat, keys=keys, **kwargs)

    def delete_user_cat(self, name):
        """Delete a user catalog."""
        if name not in self.db.tables or name not in self.catalogs:
            raise ValueError('not a valid catalog name')
        self.catalogs[name].drop()
        del self.catalogs[name]

    def cross_match(self, name, cat1, cat2, radius=1.):
        """Cross-match two catalogs and creates a CrossMatch catalog.

        Parameters
        ----------
        name: str
            Name of the `musex.catalog.CrossMatch` catalog in the database.
        cat1: `musex.catalog.Catalog`
            The first catalog to cross-match.
        cat2. `musex.catalog.Catalog`
            The second catalog.
        radius: float
            The cross-match radius in arc-seconds.

        """
        if name in self.catalogs:
            raise ValueError("A catalog with this name already exists.")
        return gen_crossmatch(name, self.db, cat1, cat2, radius)
