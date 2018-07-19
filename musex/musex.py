import importlib
import logging
import numpy as np
import os
import re
import sys
import textwrap
from astropy.io import fits
from astropy.table import Table, Row
from collections import OrderedDict, Sized, Iterable
from contextlib import contextmanager
from mpdaf.log import setup_logging
from mpdaf.obj import Image

from .dataset import load_datasets, MuseDataSet
from .catalog import (Catalog, InputCatalog, ResultSet, table_to_odict,
                      MarzCatalog, IdMapping, get_cat_name)
from .crossmatching import CrossMatch, gen_crossmatch
from .source import SourceX
from .utils import extract_subimage, load_db, load_yaml_config, progressbar
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
        ('segmap', ''),
        ('query', ''),
        ('dataset', ''),
        ('convolve_fwhm', 1.0),
        ('psf_threshold', 1.0),
        ('mask_size_x', 1),
        ('mask_size_y', 1)
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
        Path of the settings file. If None, it defaults to
        ``~/.musex/settings.yaml`` if possible, otherwise to the
        ``udf/settings.yaml`` file which comes with MuseX.
    muse_dataset: str
        Name of the Muse dataset to work on.
    id_mapping: str
        Name of a IdMapping catalog to use.

    """

    def __init__(self, settings_file=None, muse_dataset=None, id_mapping=None,
                 **kwargs):
        if settings_file is None:
            settings_file = os.path.expanduser('~/.musex/settings.yaml')
            if not os.path.exists(settings_file):
                dirname = os.path.abspath(os.path.dirname(__file__))
                settings_file = os.path.join(dirname, 'udf', 'settings.yaml')

        self.logger = logging.getLogger(__name__)
        self.settings_file = settings_file
        self.conf = load_yaml_config(settings_file)
        self.conf.update(kwargs)
        self.workdir = self.conf['workdir']
        self.conf.setdefault('export', {})
        self.conf['export'].setdefault('path', f'{self.workdir}/export')
        self.db = db = load_db(self.conf['db'])

        # Creating the IdMapping object if required
        self.id_mapping = None
        id_mapping = id_mapping or self.conf.get('idmapping')
        if id_mapping:
            self.create_id_mapping(id_mapping)

        # Table to store history of operations
        self.history = db.create_table('history', primary_id='_id')

        # Load datasets
        self.datasets = load_datasets(self.conf)
        settings = self.conf['muse_datasets']
        muse_dataset = muse_dataset or settings['default']
        self.muse_dataset = MuseDataSet(muse_dataset,
                                        settings=settings[muse_dataset])

        # Load catalogs table
        self.catalogs_table = _create_catalogs_table(db)
        self._load_input_catalogs()
        self._load_user_catalogs()

        # Marz
        self.marzcat = MarzCatalog('marz', db, primary_id='_id')
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
        self.logger.info("Input catalogs loaded.")

    def _load_user_catalogs(self):
        """Load user generated catalogs."""
        self.catalogs = {}

        # User catalogs
        for row in self.catalogs_table.find(type='user'):
            name = row['name']
            self.catalogs[name] = Catalog(
                name, self.db, workdir=self.workdir, idname=row['idname'],
                raname=row['raname'], decname=row['decname'],
                segmap=row['segmap'])
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

        self.logger.info("User catalogs loaded.")

    def info(self, outstream=None):
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
        exportdir = self.conf['export']['path']
        return f'{exportdir}/{self.muse_dataset.name}'

    def find_parent_cat(self, cat):
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
        whereclause = resultset.whereclause
        self.catalogs[name] = cat = Catalog.from_parent_cat(
            parent_cat, name, self.workdir, whereclause)

        if isinstance(resultset, Table):
            resultset = table_to_odict(resultset)
        cat.insert(resultset)
        return cat

    def create_id_mapping(self, name):
        """Create or get an IdMapping object."""
        self.id_mapping = IdMapping(name, self.db)

    @contextmanager
    def use_id_mapping(self, cat):
        """Temporarily modifies ``cat`` to use the ``id_mapping``."""
        if self.id_mapping is None:
            raise ValueError('no id_mapping defined.')
        cat.idmap = self.id_mapping
        yield
        cat.idmap = None

    def to_sources(self, res_or_cat, size=5, srcvers='', apertures=None,
                   datasets=None, only_active=True, refspec='MUSE_TOT_SKYSUB',
                   content=('parentcat', 'segmap', 'history'), verbose=False):
        """Export a catalog or selection to sources (SourceX).

        Parameters
        ----------
        res_or_cat: `ResultSet`, `Catalog`, `Table`
            Either a result from a query or a catalog to export.
        size: float or list of float
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
        origin = ('MuseX', __version__, '', '')

        # FIXME: not sure here, but the current dataset should be the same as
        # the one used to produce the masks
        assert cat.meta['dataset'] == self.muse_dataset.name

        # debug = self.logger.debug
        debug = self.logger.debug
        info = self.logger.info

        if isinstance(size, (float, int)):
            info('Exporting %s sources with %s dataset, size=%.1f',
                 nrows, self.muse_dataset.name, size)
            size = [size] * nrows

        elif isinstance(size, Iterable) and isinstance(size, Sized):
            if nrows != len(size):
                msg = ("Length of res_or_cat (%d) does not match length of "
                       "size (%d)")
                raise ValueError(msg % (nrows, len(size)))

            info('Exporting %s sources with %s dataset, %.1f<=size<=%.1f',
                 nrows, self.muse_dataset.name, np.min(size), np.max(size))
        else:
            raise ValueError("'size' should be a float or list of floats")

        use_datasets = [self.muse_dataset]
        if datasets is not None:
            if not isinstance(datasets, (list, tuple)):
                datasets = [datasets]
            use_datasets += [self.datasets[a] for a in datasets]
        else:
            use_datasets += list(self.datasets.values())

        parent_cat = self.find_parent_cat(cat)
        # minsize = min(*size) // 2
        minsize = 0.
        nskywarn = (50, 5)
        refskyf = resultset[0]['mask_sky']
        refskyim = Image(str(cat.workdir / refskyf), copy=False)
        idname, raname, decname = cat.idname, cat.raname, cat.decname
        author = self.conf['author']

        redshifts = self.conf['export'].get('redshifts', {})
        header_columns = self.conf['export'].get('header_columns', {})
        for key, colname in header_columns.items():
            if colname not in resultset.colnames:
                self.logger.warning(
                    "'%s' column not found, though it is specified in the "
                    "settings file for the %s keyword", colname, key)

        if verbose is False:
            resultset = progressbar(resultset)
            setup_logging('musex', level=logging.WARNING, stream=sys.stdout)

        for row, src_size in zip(resultset, size):
            if isinstance(row, Row):
                # convert Astropy Row to dict
                row = dict(zip(row.colnames, row.as_void().tolist()))

            src = SourceX.from_data(row[idname], row[raname], row[decname],
                                    origin)
            src.SRC_V = (srcvers, 'Source Version')
            info('source %05d (%.5f, %.5f)', src.ID, src.DEC, src.RA)
            src.CATALOG = os.path.basename(parent_cat.name)
            src.default_size = src_size
            src.SIZE = src_size

            # Add keywords from columns
            for key, colname in header_columns.items():
                if row.get(colname) is not None:
                    if key == 'COMMENT':
                        # truncate comment if too long
                        com = re.sub('[^\s!-~]', '', row[colname])
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

            for ds in use_datasets:
                ds.add_to_source(src)

            if 'segmap' in content:
                cat.add_segmap_to_source(src, parent_cat.extract,
                                         dataset=self.muse_dataset)
            if 'parentcat' in content:
                parent_cat.add_to_source(src, row, **parent_cat.extract)

            debug('IMAGES: %s', ', '.join(src.images.keys()))

            center = (src.DEC, src.RA)
            # If mask_sky is always the same, reuse it instead of reloading
            skyim = (refskyim if row['mask_sky'] == refskyf else
                     str(cat.workdir / row['mask_sky']))
            src.images['MASK_SKY'] = extract_subimage(
                skyim, center, (src_size, src_size), minsize=minsize)

            maskim = Image(str(cat.workdir / row['mask_obj']), copy=False)
            # centerpix = maskim.wcs.sky2pix(center)[0]
            # debug('center: (%.5f, %.5f) -> (%.2f, %.2f)', *center,
            #       *centerpix.tolist())
            # FIXME: check that center is inside mask
            src.images['MASK_OBJ'] = extract_subimage(
                maskim, center, (src_size, src_size), minsize=minsize)

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
            debug('MASKS: SKY: %.1f%%, OBJ: %.1f%%', nfracsky, nfracobj)

            src.extract_all_spectra(apertures=apertures)
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
        if isinstance(res_or_cat, ResultSet):
            cat = res_or_cat.as_table()
        elif isinstance(res_or_cat, Catalog):
            cat = res_or_cat.select().as_table()
        elif isinstance(res_or_cat, Table):
            cat = res_or_cat
        else:
            raise ValueError('invalid input for res_or_cat')

        if outdir is None:
            cname = cat.catalog.name
            outdir = f'{self.exportdir}/{cname}/sources'
        os.makedirs(outdir, exist_ok=True)

        pdfconf = self.conf['export'].get('pdf', {})
        white = self.muse_dataset.white
        info = self.logger.info
        ima2 = pdfconf.get('image')

        for src in self.to_sources(res_or_cat, **kwargs):
            outn = outname.format(src=src)
            fname = f'{outdir}/{outn}.fits'
            src.write(fname)
            info('fits written to %s', fname)
            if create_pdf:
                fname = f'{outdir}/{outn}.pdf'
                src.to_pdf(fname, white, ima2=ima2, mastercat=cat)
                info('pdf written to %s', fname)

    def export_marz(self, res_or_cat, outfile=None, export_sources=False,
                    outdir=None, srcname='source-{src.ID:05d}', **kwargs):
        """Export a catalog or selection for MarZ.

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

        cname = get_cat_name(res_or_cat)
        if outdir is None:
            outdir = f'{self.exportdir}/{cname}/marz'
        os.makedirs(outdir, exist_ok=True)
        if outfile is None:
            outfile = f'{outdir}/marz-{cname}-{self.muse_dataset.name}.fits'

        wave, data, stat, sky, meta = [], [], [], [], []

        # If sources must be exported, we need to tell .to_sources what to put
        # inside the sources (all datasets and the segmap)
        datasets = None if export_sources else []
        content = ('segmap', 'parentcat') if export_sources else tuple()

        for s in self.to_sources(res_or_cat, datasets=datasets,
                                 content=content, **kwargs):
            # TODO: how to choose which spectrum to use ?
            # if args.selmode == 'udf':
            #     smag = s.mag[s.mag['BAND'] == 'F775W']
            #     mag = -99 if len(smag) == 0 else smag['MAG'][0]
            #     if 'MUSE_PSF_SKYSUB' in s.spectra:
            #         if hasattr(s,'FWHM'):
            #             if s.FWHM*0.03 > 0.7: # large object we use WHITE
            #                 sp = s.spectra['MUSE_WHITE_SKYSUB']
            #             else:
            #                 sp = s.spectra['MUSE_PSF_SKYSUB']
            #     else: # we use mag, no size available
            #         if mag < 26.5:
            #             sp = s.spectra['MUSE_WHITE_SKYSUB']
            # elif args.selmode == 'origin':
            #     if 'MUSE_PSF' in s.spectra:
            #         sp = s.spectra['MUSE_PSF']
            #     else:
            #         sp = s.spectra['MUSE_TOT']
            # else:
            #     self.logger.error('unknown selmode '+args.selmode)
            #     return

            sp = s.spectra[s.REFSPEC]
            wave.append(sp.wave.coord())
            data.append(sp.data.filled(np.nan))
            stat.append(sp.var.filled(np.nan))
            sky.append(s.spectra['MUSE_SKY'].data.filled(np.nan))

            # if args.selmode == 'udf':
            #     zmuse = s.z[s.z['Z_DESC'] == 'MUSE']
            #     z = 0 if len(zmuse) == 0 else zmuse['Z'][0]
            #     band1 = s.mag[s.mag['BAND'] == 'F775W']
            #     mag1 = -99 if len(band1)==0  else band1['MAG'][0]
            #     band2 = s.mag[s.mag['BAND'] == 'F125W']
            #     mag2 = -99 if len(band2)==0  else band2['MAG'][0]
            z = 0
            mag1 = -99
            mag2 = -99
            meta.append(('%05d' % s.ID, s.RA, s.DEC, z,
                        s.header.get('CONFID', 0),
                        s.header.get('TYPE', 0), mag1, mag2, s.REFSPEC))

            if export_sources:
                outn = srcname.format(src=s)
                fname = f'{outdir}/{outn}.fits'
                s.write(fname)
                self.logger.info('fits written to %s', fname)

        t = Table(rows=meta, names=['NAME', 'RA', 'DEC', 'Z', 'CONFID', 'TYPE',
                                    'F775W', 'F125W', 'REFSPEC'],
                  meta={'CUBE_V': s.CUBE_V, 'SRC_V': s.SRC_V})
        hdulist = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(name='WAVE', data=np.vstack(wave)),
            fits.ImageHDU(name='DATA', data=np.vstack(data)),
            fits.ImageHDU(name='STAT', data=np.vstack(stat)),
            fits.ImageHDU(name='SKY', data=np.vstack(sky)),
            fits.BinTableHDU(name='DETAILS', data=t)
        ])
        self.logger.info('Writing %s', outfile)
        hdulist.writeto(outfile, overwrite=True, output_verify='silentfix')

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
        if name not in self.db.tables or name not in self.catalogs:
            raise ValueError('not a valid catalog name')
        self.catalogs[name].drop()
        del self.catalogs[name]

    def cross_match(self, name, cat1, cat2, radius=1.):
        """Cross-match two catalogs.

        This function cross-match two catalogs and creates a CrossMatch catalog
        in the database.

        Parameters
        ----------
        name: str
            Name of the CrossMatch catalog in the database.
        cat1: `musex.catalog.SpatialCatalog`
            The first catalog to cross-match.
        cat2. `musex.catalog.SpatialCatalog`
            The second catalog.
        radius: float
            The cross-match radius in  arc-seconds.

        """
        if name in self.catalogs:
            raise ValueError("A catalog with this name already exists.")

        return gen_crossmatch(name, self.db, cat1, cat2, radius)
