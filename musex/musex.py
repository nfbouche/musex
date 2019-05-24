import importlib
import itertools
import logging
import multiprocessing
import datetime
import os
import sys
import textwrap
from collections import Iterable, Sized
from contextlib import contextmanager

import numpy as np

from astropy.table import Table
from mpdaf.obj import Image
from mpdaf.sdetect import Catalog as MpdafCatalog
from mpdaf.sdetect import create_masks_from_segmap
from mpdaf.tools import isiter, progressbar

from .catalog import (
    Catalog,
    IdMapping,
    InputCatalog,
    MarzCatalog,
    ResultSet,
    get_cat_name,
    get_result_table,
)
from .crossmatching import CrossMatch, gen_crossmatch
from .dataset import MuseDataSet, load_datasets
from .source import create_source, sources_to_marz
from .utils import load_db, load_yaml_config, table_to_odict
from .version import __description__, __version__

__all__ = ['MuseX']

LOGO = r"""
  __  __               __  __
  |  \/  |_   _ ___  ___\ \/ /
  | |\/| | | | / __|/ _ \\  /
  | |  | | |_| \__ \  __//  \
  |_|  |_|\__,_|___/\___/_/\_\

"""


def _create_catalogs_table(db):
    """Create the 'catalogs' table which stores metadata about catalogs."""
    # Return the table if it exists
    if 'catalogs' in db:
        return db['catalogs']

    # Create the table
    table = db.create_table('catalogs')
    # Force the creation of the SQLATable
    assert table.table is not None

    # and make sure that all columns exists
    row = {
        'name': '',
        'creation_date': '',
        'type': '',
        'parent_cat': '',
        'raname': '',
        'decname': '',
        'zname': '',
        'zconfname': '',
        'idname': '',
        'maxid': 1,
        'query': '',
    }
    table._sync_columns(row, True)
    return table


def _worker_export(args):
    try:
        return create_source(*args[0], **args[1])
    except KeyboardInterrupt:
        pass


class MuseX:
    """The main MuseX class.

    This class is the central part of MuseX, it gives access to all the
    catalogs and datasets, and provides some high-level operations.

    Parameters
    ----------
    settings_file : str
        Path of the settings file.
    muse_dataset : str
        Name of the Muse dataset to work on.
    id_mapping : str
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
        self._muse_dataset = None
        self.muse_dataset = muse_dataset or settings['default']

        # Load catalogs table
        self.catalogs_table = _create_catalogs_table(self.db)
        self._load_input_catalogs()
        self._load_user_catalogs()

        # Marz
        self.marzcat = MarzCatalog('marz', self.db, primary_id='_id',
                                   author=self.conf['author'])
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

            self.input_catalogs[name] = cls.from_settings(
                name, self.db, author=self.conf['author'], **conf
            )
        self.logger.info("Input catalogs loaded")

    def _load_user_catalogs(self):
        """Load user generated catalogs."""
        self.catalogs = {}

        # User catalogs
        for row in self.catalogs_table.find(type='user'):
            name = row['name']
            self.catalogs[name] = Catalog(
                name, self.db,
                idname=row['idname'],
                raname=row['raname'],
                decname=row['decname'],
                zname=row['zname'],
                zconfname=row['zconfname'],
                primary_id=row.get('primary_id'),
                author=self.conf['author'],
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

            database       : {self.db}
            settings file  : {self.settings_file}
            muse_dataset   : {self.muse_dataset.name}
            """))

        maxlen = max(map(len, itertools.chain(
            self.datasets, self.input_catalogs, self.catalogs)))

        outstream.write('datasets       :\n')
        for name, ds in self.datasets.items():
            desc = ds.description or ''
            outstream.write(f"    - {name:{maxlen}s} : {desc}\n")

        outstream.write('input_catalogs :\n')
        for name, cat in self.input_catalogs.items():
            outstream.write(f"    - {name:{maxlen}s} : {len(cat)} rows\n")

        outstream.write('catalogs       :\n')
        for name, cat in self.catalogs.items():
            outstream.write(f"    - {name:{maxlen}s} : {len(cat)} rows\n")

        if self.id_mapping is not None:
            outstream.write(f"id_mapping     : {self.id_mapping.name}")

    @property
    def muse_dataset(self):
        return self._muse_dataset

    @muse_dataset.setter
    def muse_dataset(self, name):
        conf = self.conf['muse_datasets']
        if name not in conf:
            raise ValueError('invalid dataset name')
        self._muse_dataset = MuseDataSet(name, settings=conf[name])

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

    def new_catalog(self, name, idname='ID', primary_id=None, raname=None, decname=None,
                    zname=None, zconfname=None, drop_if_exists=False):
        """Create a new user catalog.

        Parameters
        ----------
        name : str
            Name of the catalog.
        idname : str, optional
            Name of the 'id' column.
        raname : str, optional
            Name of the 'ra' column.
        decname : str, optional
            Name of the 'dec' column.
        zname : str, optional
            Name of the 'z' column.
        zconfname : str, optional
            Name of the 'confid' column.
        drop_if_exists : bool, optional
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
                                      primary_id=primary_id,
                                      raname=raname, decname=decname,
                                      zname=zname, zconfname=zconfname,
                                      author=self.conf['author'])
        return self.catalogs[name]

    def new_catalog_from_resultset(self, name, resultset, primary_id=None,
                                   drop_if_exists=False):
        """Create a new user catalog from a query result.

        Parameters
        ----------
        name : str
            Name of the catalog.
        resultset : `musex.ResultSet`
            Result from a query.
        primary_id : str
            The primary id for the SQL table, must be a column name. Defaults
            to 'ID'.
        drop_if_exists : bool
            Drop the catalog if it already exists.

        """
        if not isinstance(resultset, (Table, ResultSet)):
            raise ValueError('unknown input type, resultset must be a '
                             'ResultSet or Table object')

        if name in self.db.tables:
            if drop_if_exists:
                self.delete_user_cat(name)
            else:
                raise ValueError('table already exists')

        parent_cat = resultset.catalog
        whereclause = resultset.whereclause
        self.catalogs[name] = cat = Catalog.from_parent_cat(
            parent_cat, name, whereclause, primary_id=primary_id
        )

        if isinstance(resultset, Table):
            resultset = table_to_odict(resultset)
        cat.insert(resultset)
        return cat

    def create_id_mapping(self, name):
        """Create or get an IdMapping object."""
        self.id_mapping = IdMapping(name, self.db)

    def create_masks_from_segmap(self, cat, maskdir, limit=None, n_jobs=-1,
                                 skip_existing=True, margin=0,
                                 psf_threshold=0.5, mask_size=(20, 20),
                                 convolve_fwhm=0):
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
        psf_threshold : float
            Threshold applied to the PSF to get a binary image.
        mask_size : tuple
            Size of the source masks (arcsec).
        convolve_fwhm : float
            FWHM for the PSF convolution (arcsec).

        """
        ref_image = self.muse_dataset.white
        catfile = cat.params['catalog']
        tbl = MpdafCatalog.read(catfile)
        self.logger.info('read catalog %s with %d sources', catfile, len(tbl))

        tbl = tbl.select(ref_image.wcs, ra=cat.raname, dec=cat.decname, margin=margin)
        self.logger.info('selected %d sources in dataset footprint', len(tbl))

        if limit:
            tbl = tbl[:limit]

        os.makedirs(maskdir, exist_ok=True)
        create_masks_from_segmap(
            cat.params['segmap'], tbl, ref_image, n_jobs=n_jobs,
            skip_existing=skip_existing,
            masksky_name=f'{maskdir}/mask-sky.fits',
            maskobj_name=f'{maskdir}/mask-source-%05d.fits',
            idname=cat.idname, raname=cat.raname, decname=cat.decname,
            margin=margin, mask_size=mask_size, convolve_fwhm=convolve_fwhm,
            psf_threshold=psf_threshold)

    def _prepare_datasets(self, datasets, catname):
        # compute the list of datasets to use
        use_datasets = {self.muse_dataset: None}
        if datasets is not None:
            # if datasets is specified, use it:
            # - of a dict, use tags specified in the dict
            # - if a list of datasets, use all tags from these datasets
            if isinstance(datasets, dict):
                use_datasets.update({self.datasets[name]: val
                                     for name, val in datasets.items()})
            elif isiter(datasets):
                use_datasets.update({self.datasets[name]: None
                                     for name in datasets})
        else:
            # otherwise we use all datasets, except the ones linked to
            # another catalog
            for ds in self.datasets.values():
                if ds.linked_cat and catname and ds.linked_cat != catname:
                    continue
                use_datasets[ds] = None
        return use_datasets

    def to_sources(
            self,
            res_or_cat,
            apertures=None,
            catalogs=None,
            datasets=None,
            extra_header=None,
            history=True,
            masks_dataset=None,
            n_jobs=1,
            only_active=True,
            outdir=None,
            outname=None,
            refspec=None,
            segmap=False,
            size=5,
            srcvers='',
            user_func=None,
            user_func_kw=None,
            verbose=False):
        """Export a catalog or selection to sources (SourceX).

        This is the main function to export sources, so it does a lot of
        things to prepare all the data that will be added to the sources.
        This function is a generator, so to get the sources you must iterate
        on the result of the function.

        Parameters
        ----------
        res_or_cat : `ResultSet`, `Catalog`, `Table`
            Either a result from a query or a catalog to export.
        apertures : list of float
            List of aperture radii for spectra extraction.
        catalogs : list of `Catalog`
            List of catalogs that will be added to the source as table
            extension, with a selection done on the WCS.
        datasets : iterable or dict
            List of dataset names to use for the sources, or dictionary with
            dataset names associated to a list of tags to use. By default all
            datasets are used, and all tags.
        extra_header : dict
            Dict with additional keywords/values to add to the Source header.
        history : bool
            If True, add the log of operations done on each source, which is
            saved in HISTORY keywords.
        masks_dataset : str
            Name of the dataset from which the source and sky masks are taken.
            If missing, no spectra will be extracted from the source cube.
        n_jobs : int
            Number of parallel processes.
        only_active : bool
            If True the inactive sources, i.e. the sources that have been
            merged into another, are filtered.
        outdir : str
            Output directory.
        outname : str
            Output file name.
        refspec : str
            Name of the reference spectra. Used if refspec is not specified in
            the catalog, and mapped to the REFSPEC keyword using the
            ``header_columns`` block in the settings.
        segmap : bool
            If True, add the segmentation map defined in the parent catalog.
        size : float or list of float
            Size of the images (in arcseconds) added in the sources.
        srcvers : str
            Version of the sources (SRC_V).
        user_func : callable
            User specified function that is called at the end of the source
            creation process, with the source object as first argument, and the
            catalog row as second argument.
        user_func_kw : dict
            Dict with additional keywords/values to pass to the user_func
        verbose : bool
            Verbose flag.

        """
        resultset = get_result_table(res_or_cat, filter_active=only_active)
        nrows = len(resultset)
        cat = resultset.catalog
        info = self.logger.info

        # find attributes of the parent catalog
        parent_catname = None
        parent_extract = None
        parent_prefix = None
        try:
            parent_cat = self.find_parent_cat(cat)
        except ValueError:
            parent_cat = None
        else:
            parent_catname = parent_cat.name
            if hasattr(parent_cat, 'params'):
                parent_extract = parent_cat.params.get('extract', {})
                parent_prefix = parent_extract.get('prefix')

        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)

        # make sure that size is a list with the same length as the catalog
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

        # compute the list of datasets to use
        use_datasets = self._prepare_datasets(datasets, parent_catname)
        info('using datasets: %s', ', '.join(ds.name for ds in use_datasets))

        # keywords added to the source
        header = {'SRC_V': (srcvers, 'Source Version')}
        if parent_cat:
            header['CATALOG'] = os.path.basename(parent_catname),
        if extra_header:
            header.update(extra_header)

        # export parameters (mags, redshits, header_columns, etc.)
        kw = {
            **self.conf['export'],
            'apertures': apertures,       # list of apertures for spectra
            'datasets': use_datasets,     # datasets to use
            'header': header,             # additional keywords
            'outdir': outdir,             # output directory
            'outname': outname,           # output filename
            'user_func': user_func,       # user function
            'user_func_kw': user_func_kw, # user function dictionnary
        }

        # segmap from the parent cat
        if segmap and parent_cat:
            segmapfile = parent_cat.params.get('segmap')
            if segmapfile is None:
                info('could not find segmap from %s', parent_cat.name)
            else:
                segmap_tag = parent_prefix + "_SEGMAP" if parent_prefix else "SEGMAP"
                kw['segmap'] = (segmap_tag, Image(segmapfile))

        # check if header_columns are available in the resultset
        header_columns = kw.get('header_columns', {})
        for key, colname in header_columns.items():
            if colname not in resultset.colnames:
                self.logger.warning(
                    "'%s' column not found, though it is specified in the "
                    "settings file for the %s keyword", colname, key)

        # additional catalogs
        if catalogs:
            kw['catalogs'] = {}
            # special treatment for the parent of the exported catalog
            if parent_cat:
                kw['header']['REFCAT'] = f"{parent_prefix}_CAT"

            for pcat in catalogs:
                parent = self.find_parent_cat(pcat)
                if hasattr(parent, 'params'):
                    pcat_extract = parent.params.get('extract', {})
                    pcat_prefix = pcat_extract.get('prefix')
                    if pcat_prefix:
                        columns = pcat_extract.get('columns')
                        pcat = parent.select(columns=columns).as_table()
                        pcat.meta.update(pcat_extract)
                        kw['catalogs'][f"{pcat_prefix}_CAT"] = pcat

        author = self.conf['author']

        # build the list of sources to compute
        to_compute = []
        for res, src_size in zip(resultset, size):
            row = dict(zip(res.colnames, tuple(res)))

            # dataset for masks
            if 'mask_dataset' in row:
                maskds = self.datasets[row['mask_dataset']]
            elif masks_dataset is not None:
                maskds = self.datasets[masks_dataset]
            else:
                maskds = None

            if history:
                hist_items = [(o['msg'], author, o['date'])
                              for o in cat.get_log(row[cat.idname])]
                hist_items.append(('source created', author,
                                   datetime.datetime.now().isoformat()))
            else:
                hist_items = None

            to_compute.append(((row, cat.idname, cat.raname, cat.decname,
                                src_size, refspec, hist_items, maskds), kw))

        # create the sources, either with multiprocessing or directly with
        # create_source
        if n_jobs > 1:
            # multiprocessing.log_to_stderr('DEBUG')
            pool = multiprocessing.Pool(n_jobs, maxtasksperchild=50)
            chunksize = min(20, nrows // n_jobs + 1)
            sources = pool.imap_unordered(_worker_export, to_compute,
                                          chunksize=chunksize)
        else:
            sources = (create_source(*args, **kw) for args, kw in to_compute)

        try:
            if verbose is False:
                ctx = self.use_loglevel('WARNING')
                ctx.__enter__()
                bar = progressbar(sources, total=nrows)

            for src in sources:
                yield src

                if verbose is False:
                    bar.update()

            if n_jobs > 1:
                pool.close()
                pool.join()
        except KeyboardInterrupt:
            if n_jobs > 1:
                pool.terminate()
        finally:
            if verbose is False:
                ctx.__exit__(*sys.exc_info())

    def export_sources(self, res_or_cat, outdir=None,
                       outname='source-{src.ID:05d}', **kwargs):
        """Save a catalog or selection to sources (SourceX).

        See `MuseX.to_sources` for the additional arguments.

        Parameters
        ----------
        res_or_cat : `ResultSet`, `Catalog`, `Table`
            Either a result from a query or a catalog to export.
        outdir : str
            Output directory. If None the default is
            `'{conf[export][path]}/{self.muse_dataset.name}/{cname}/sources'`.
        outname : str
            Output file name. If None the default is `'source-{src.ID:05d}'`.

        """
        if outdir is None:
            catname = get_cat_name(res_or_cat)
            outdir = f'{self.exportdir}/{catname}/sources'
        return list(self.to_sources(res_or_cat, outdir=outdir,
                                    outname=outname, **kwargs))

    def export_marz(self, res_or_cat, outfile=None, export_sources=False,
                    datasets=None, sources_dataset=None, outdir=None,
                    srcname='source-{src.ID:05d}', skyspec='MUSE_SKY',
                    **kwargs):
        """Export a catalog or selection for MarZ.

        Pre-generated sources may be used by specifying the related dataset.
        Each object in the catalog (or the selection) must have an existing
        source, and each source must have a REFSPEC attribute designating the
        spectrum to use in MarZ.

        Parameters
        ----------
        res_or_cat : `ResultSet`, `Catalog`, `Table`
            Either a result from a query or a catalog to export.
        outfile : str
            Output file. If None the default is
            `'{conf[export][path]}/marz-{cat.name}-{muse_dataset.name}.fits'`.
        export_sources : bool
            If True, the source files are also exported.
        datasets : iterable or dict
            List of dataset names to use for the sources, or dictionary with
            dataset names associated to a list of tags to use. By default all
            datasets are used, and all tags.
        sources_dataset : str
            Name of the dataset from which the sources are taken.
        outdir : str
            Output directory. If None the default is
            `'{conf[export][path]}/{self.muse_dataset.name}/{cname}/marz'`.
        srcname : str
            Output file name. If None the default is `'source-{src.ID:05d}'`.
        skyspec : str
            The tag name to find the sky spectrum in the sources.

        """
        cat = get_result_table(res_or_cat)
        cname = cat.catalog.name
        parent_cat = self.find_parent_cat(cat.catalog)
        if outdir is None:
            outdir = f'{self.exportdir}/{cname}/marz'
        os.makedirs(outdir, exist_ok=True)
        if outfile is None:
            outfile = f'{outdir}/marz-{cname}-{self.muse_dataset.name}.fits'

        # Keyword to check in the sources for ORIGIN.
        version_meta = parent_cat.meta.get('version_meta', None)
        if version_meta is not None:
            check_keyword = (version_meta, parent_cat.meta[version_meta])
        else:
            check_keyword = None

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
                kwargs.setdefault('segmap', True)
                kwargs['outdir'] = outdir
                kwargs['outname'] = srcname
            else:
                if datasets is None:
                    # skip using additional datasets. In this case the
                    # muse_dataset will be used, with spectra extracted
                    # from the cube.
                    datasets = []
                    kwargs['history'] = False
                    kwargs['segmap'] = False

            src_iter = self.to_sources(cat, datasets=datasets, **kwargs)

        sources_to_marz(src_iter, outfile, skyspec=skyspec)

    def import_marz(self, catfile, catalog, **kwargs):
        """Import a MarZ catalog.

        Parameters
        ----------
        catfile : str
            The catalog to import from Marz.
        catalog : str or `Catalog`
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
        name : str
            Name of the `musex.catalog.CrossMatch` catalog in the database.
        cat1 : `musex.Catalog`
            The first catalog to cross-match.
        cat2 : `musex.Catalog`
            The second catalog.
        radius : float
            The cross-match radius in arc-seconds.

        """
        if name in self.catalogs:
            raise ValueError("A catalog with this name already exists.")
        return gen_crossmatch(name, self.db, cat1, cat2, radius)
