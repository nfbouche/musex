import importlib
import itertools
import logging
import multiprocessing
import numpy as np
import os
import sys
import textwrap

from astropy.table import Table
from collections import OrderedDict, Sized, Iterable
from contextlib import contextmanager
from mpdaf.obj import Image
from mpdaf.sdetect import create_masks_from_segmap, Catalog as MpdafCatalog
from mpdaf.tools import progressbar, isiter

from .dataset import load_datasets, MuseDataSet
from .catalog import (Catalog, InputCatalog, ResultSet, MarzCatalog,
                      IdMapping, get_result_table, get_cat_name)
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

    def new_catalog(self, name, idname='ID', raname='RA', decname='DEC',
                    drop_if_exists=False):
        """Create a new user catalog.

        Parameters
        ----------
        name : str
            Name of the catalog.
        idname : str
            Name of the 'id' column.
        raname : str
            Name of the 'ra' column.
        decname : str
            Name of the 'dec' column.
        drop_if_exists : bool
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

        tbl = tbl.select(ref_image.wcs, ra=cat.raname, dec=cat.decname,
                         margin=margin)
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

    def to_sources(self, res_or_cat, size=5, srcvers='', apertures=None,
                   datasets=None, only_active=True, refspec='MUSE_TOT_SKYSUB',
                   content=('parentcat', 'segmap', 'history'), verbose=False,
                   n_jobs=1, masks_dataset=None, outdir=None, outname=None,
                   create_pdf=False, user_func=None):
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
        only_active : bool
            If True the inactive sources, i.e. the sources that have been
            merged into another, are filtered.
        refspec : str
            Name of the reference spectra. Used if refspec is not specified in
            the catalog, and mapped to the REFSPEC keyword using the
            ``header_columns`` block in the settings.
        content : list of str
            This allows to select specific kind of data that can be added to
            the sources:

            - 'parentcat' for the parent catalog.
            - 'segmap' for the segmentation map.
            - 'history' for the log of operations done on each source, which is
              saved in HISTORY keywords.
        masks_dataset : str
            Name of the dataset from which the source and sky masks are taken.
            If missing, no spectra will be extracted from the source cube.
        outdir : str
            Output directory.
        outname : str
            Output file name.
        create_pdf : bool
            If True, create a pdf for each source.
        user_func : callable
            User specified function that is called at the end of the source
            creation process, with the source object as first argument.

        """
        resultset = get_result_table(res_or_cat, filter_active=only_active)
        nrows = len(resultset)
        cat = resultset.catalog
        parent_cat = self.find_parent_cat(cat)
        info = self.logger.info

        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)

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

        info('using datasets: %s', ', '.join(ds.name for ds in use_datasets))

        # keywords added to the source
        header = {
            'SRC_V': (srcvers, 'Source Version'),
            'CATALOG': os.path.basename(parent_cat.name),
        }

        # export parameters (mags, redshits, header_columns, etc.)
        kw = {
            **self.conf['export'],
            'apertures': apertures,     # list of apertures for spectra
            'datasets': use_datasets,   # datasets to use
            'header': header,           # additional keywords
            'verbose': verbose,
            'outdir': outdir,           # output directory
            'outname': outname,         # output filename
            'user_func': user_func,     # user function
        }

        # dataset for masks
        if masks_dataset is not None:
            kw['maskds'] = self.datasets[masks_dataset]
            info('using mask datasets: %s', kw['maskds'])
        else:
            info('no masks specified, spectra will not be extracted')

        # segmap from the parent cat
        segmap = parent_cat.params.get('segmap')
        if 'segmap' in content and segmap:
            segmap_tag = parent_cat.params['extract']['prefix'] + "_SEGMAP"
            kw['segmap'] = (segmap_tag, Image(segmap))

        # check if header_columns are available in the resultset
        header_columns = kw.get('header_columns', {})
        for key, colname in header_columns.items():
            if colname not in resultset.colnames:
                self.logger.warning(
                    "'%s' column not found, though it is specified in the "
                    "settings file for the %s keyword", colname, key)

        kw['catalogs'] = {}
        if 'parentcat' in content:
            parent_params = parent_cat.params.get('extract', {})
            if 'prefix' in parent_params:
                columns = parent_params.get('columns')
                prefix = parent_params.get('prefix')
                pcat = parent_cat.select(columns=columns).as_table()
                pcat.meta.update(parent_params)
                kw['header']['REFCAT'] = f"{prefix}_CAT"
                kw['catalogs'][f"{prefix}_CAT"] = pcat

        if create_pdf:
            kw['pdfconf'] = self.conf['export'].get('pdf', {})

        author = self.conf['author']
        to_compute = []
        for res, src_size in zip(resultset, size):
            row = dict(zip(res.colnames, res.as_void().tolist()))

            history = []
            if 'history' in content:
                for o in cat.get_log(row[cat.idname]):
                    history.append((o['msg'], author, o['date']))
                history.append(('source created', author))

            to_compute.append(((row, cat.idname, cat.raname, cat.decname,
                                src_size, refspec, history), kw))

        if n_jobs > 1:
            # multiprocessing.log_to_stderr('DEBUG')
            pool = multiprocessing.Pool(n_jobs, maxtasksperchild=100)
            sources = pool.imap_unordered(_worker_export, to_compute,
                                          chunksize=1)
        else:
            sources = (create_source(*args, **kw) for args, kw in to_compute)

        try:
            if verbose is False:
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
                kwargs.setdefault('content', ('segmap', 'parentcat'))
                kwargs['outdir'] = outdir
                kwargs['outname'] = srcname
            else:
                if datasets is None:
                    # skip using additional datasets. In this case the
                    # muse_dataset will be used, with spectra extracted
                    # from the cube.
                    datasets = []
                    kwargs['content'] = []

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
