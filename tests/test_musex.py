import numpy as np
import os
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from mpdaf.sdetect import Source
from musex import Catalog, MuseX, masks
from numpy.testing import assert_allclose, assert_array_equal
from sqlalchemy import desc

CURDIR = os.path.abspath(os.path.dirname(__file__))
DATADIR = os.path.join(CURDIR, 'data')


def test_settings(capsys, settings_file):
    mx = MuseX(settings_file=settings_file, author='Me')
    assert mx.muse_dataset.name == 'hdfs'
    assert mx.conf['author'] == 'Me'

    expected = """\
muse_dataset   : hdfs
datasets       : test
input_catalogs : photutils, origin
"""
    captured = capsys.readouterr()
    assert expected in captured.out


def test_ingest(mx):
    assert list(mx.input_catalogs.keys()) == ['photutils', 'origin']

    # Photutils catalog
    phot = mx.input_catalogs['photutils']
    assert phot.idname == 'id'
    assert phot.raname == 'ra'
    assert phot.decname == 'dec'
    assert phot.segmap.endswith('segmap.fits')

    phot.ingest_input_catalog(limit=3)
    assert len(phot.select()) == 3

    phot.ingest_input_catalog(upsert=True)
    assert len(phot.select()) == 13
    assert phot.meta['maxid'] == 13

    tbl = phot.select().as_table()
    assert tbl[phot.idname].max() == 13

    # Catalogue origin
    orig = mx.input_catalogs['origin']
    assert orig.idname == 'ID'
    assert orig.raname == 'ra'
    assert orig.decname == 'dec'
    assert orig.segmap is None
    assert orig.mask_tpl.endswith('source-mask-%05d.fits')
    assert orig.skymask_tpl.endswith('sky-mask-%05d.fits')

    orig.ingest_input_catalog(limit=1)
    assert len(orig.select()) == 1
    assert len(orig.lines.select()) == 1

    orig.ingest_input_catalog(upsert=True)
    assert len(orig.select()) == 10
    assert orig.meta['maxid'] == 9
    assert len(orig.lines.select()) == 13
    assert len(orig.lines.select_src_ids(2)) == 1

    tbl = orig.select().as_table()
    assert tbl[orig.idname].max() == 9


def test_catalog_name(settings_file):
    mx = MuseX(settings_file=settings_file, show_banner=False, db=':memory:')
    with pytest.warns(UserWarning,
                      match='catalog name should contain only ascii letters '):
        Catalog('dont-use-dash', mx.db, workdir=mx.workdir)


def test_catalog(mx):
    phot = mx.input_catalogs['photutils']
    assert len(phot) == 13
    assert repr(phot) == "<InputCatalog('photutils', 13 rows)>"
    assert phot.meta['maxid'] == 13
    assert phot.meta['type'] == 'input'

    res = phot.select(phot.c[phot.idname] < 5, columns=[phot.idname])
    assert res.whereclause == 'photutils.id < 5'
    assert repr(res) == "<ResultSet(photutils.id < 5)>, 4 results"
    assert len(res) == 4
    assert res[0][phot.idname] == 1

    # Test empty selection
    res = phot.select(phot.c[phot.idname] > 20)
    assert repr(res) == "<ResultSet(photutils.id > 20)>, 0 results"
    assert len(res) == 0
    assert len(res.as_table()) == 0

    # Test selection by ids
    res = phot.select_ids([1, 2, 3], columns=[phot.idname])
    tbl = res.as_table()
    assert tbl.whereclause == 'photutils.id IN (1, 2, 3)'
    assert tbl.catalog is phot
    assert len(tbl) == 3
    assert tbl.colnames == ['id']

    res = phot.select_ids(1, columns=[phot.idname])
    assert len(res) == 1
    assert res[0]['id'] == 1

    res = phot.select_ids(np.array([1, 2]), columns=[phot.idname])
    assert len(res) == 2
    assert res[0]['id'] == 1

    # Test selection with a wcs
    res = phot.select(wcs=mx.muse_dataset.white.wcs, columns=[phot.idname],
                      margin=2/0.2)
    assert len(res) == 8

    # Test selection with a WCS and a mask
    res = phot.select(wcs=mx.muse_dataset.white.wcs,
                      mask=np.ones(mx.muse_dataset.white.data.shape,
                                   dtype=int))
    assert len(res) == 0



def test_user_catalog(mx):
    phot = mx.input_catalogs['photutils']
    res = phot.select(phot.c[phot.idname] < 5)
    mx.new_catalog_from_resultset('my_cat', res, drop_if_exists=True)

    with pytest.raises(ValueError):
        mx.new_catalog_from_resultset('my_cat', res, drop_if_exists=False)

    tbl = res.as_table()
    mx.new_catalog_from_resultset('my_cat', tbl, drop_if_exists=True)

    mycat = mx.catalogs['my_cat']
    assert len(mycat) == 4
    assert mycat.meta['type'] == 'user'
    assert mycat.meta['query'] == 'photutils.id < 5'
    assert isinstance(mycat.skycoord(), SkyCoord)

    # Test with a user cat created from a user cat
    res = mycat.select(mycat.c[phot.idname] > 2)
    mx.new_catalog_from_resultset('my_cat_2', res, drop_if_exists=True)

    mycat = mx.catalogs['my_cat_2']
    assert len(mycat) == 2
    assert mycat.meta['type'] == 'user'
    assert mycat.meta['query'] == 'photutils.id < 5 AND my_cat.id > 2'


def test_drop_user_catalog(mx):
    phot = mx.input_catalogs['photutils']
    res = phot.select(phot.c[phot.idname] < 5)

    catname = 'my_cat_tmp'
    mx.new_catalog_from_resultset(catname, res, drop_if_exists=True)
    assert catname in mx.db.tables

    mycat = mx.catalogs[catname]
    assert len(mycat) == 4

    with pytest.raises(ValueError):
        mx.delete_user_cat('foocat')

    with pytest.raises(ValueError):
        mx.delete_user_cat('photutils')

    mx.delete_user_cat(catname)
    assert catname not in mx.db.tables
    assert catname not in mx.catalogs


def test_update_column(mx):
    # Test addition of a new column
    phot = mx.input_catalogs['photutils']
    res = phot.select(phot.c[phot.idname] < 5)
    mx.new_catalog_from_resultset('my_cat', res, drop_if_exists=True)
    mycat = mx.catalogs['my_cat']

    assert 'foobar' not in mycat.table.columns
    mycat.update_column('foobar', 1)
    assert 'foobar' in mycat.table.columns
    assert mycat.table.find_one()['foobar'] == 1

    mycat.update_column('foobar', range(len(mycat)))
    res = mycat.select_ids([1, 2, 3])
    tbl = res.as_table()
    assert_array_equal(tbl['foobar'], [0, 1, 2])


def test_update_rows(mx):
    phot = mx.input_catalogs['photutils']
    res = phot.select(phot.c[phot.idname] < 5)
    mx.new_catalog_from_resultset('my_cat', res, drop_if_exists=True)

    mycat = mx.catalogs['my_cat']

    # insert rows
    for i, r in enumerate(res):
        r['ra'] = i
    mycat.upsert(res, show_progress=False)
    assert len(mycat) == 4
    assert_allclose([o['ra'] for o in mycat.select(columns=['ra'])],
                    np.arange(4, dtype=float))

    # insert table
    tbl = res.as_table()
    tbl['dec'] = 2.0
    mycat.upsert(tbl, show_progress=False)
    assert len(mycat) == 4
    assert_allclose([o['dec'] for o in mycat.select(columns=['dec'])], 2.0)


def test_id_mapping(mx):
    mx.create_id_mapping('mapping_name')
    meta = mx.db['catalogs'].find_one(name='mapping_name')
    assert meta['type'] == 'id'
    assert mx.id_mapping is not None

    phot = mx.input_catalogs['photutils']
    res = phot.select(phot.c[phot.idname] > 8)
    mycat = mx.new_catalog_from_resultset('my_cat', res, drop_if_exists=True)

    mycat_ids = [x['id'] for x in mycat.select()]
    mx.id_mapping.add_ids(mycat_ids, mycat)
    assert mx.id_mapping.table.columns == ['ID', 'my_cat_id']
    assert mycat.table.count() == 5
    assert mx.id_mapping.table.count() == 5

    # By default insert row does not change id_mapping
    mycat.insert([{'ra': 1, 'dec': 2}])
    assert mycat.table.count() == 6
    assert mx.id_mapping.table.count() == 5

    # It does update with the context manager
    with mx.use_id_mapping(mycat):
        mycat.insert([{'id': 20, 'ra': 3, 'dec': 4}])
        assert mycat.table.count() == 7
        assert mx.id_mapping.table.count() == 6
        res = mx.id_mapping.select(limit=1, order_by=desc('ID')).results[0]
        assert res['ID'] == 6
        assert res['my_cat_id'] == 20

    # And same for select and update
    assert mycat.select_id(6) is None
    assert mycat.select_id(20) is not None

    mycat.update_id(10, area=200)
    assert mycat.select_id(10)['area'] == 200

    with mx.use_id_mapping(mycat):
        assert mycat.select_id(20) is None
        assert mycat.select_id(6)['id'] == 20

        mycat.update_id(6, area=100)
        assert mycat.select_id(6)['area'] == 100


def test_segmap(mx):
    mycat = mx.catalogs['my_cat']
    segmap = mycat.get_segmap_aligned(mx.muse_dataset)

    assert segmap.img.shape == (90, 90)
    assert segmap.img.dtype == np.int64
    assert np.max(segmap.img._data) == 13
    assert np.all(np.unique(segmap.img._data) == np.arange(14))


def test_merge_sources(mx):
    phot = mx.input_catalogs['photutils']
    res = phot.select_ids([9, 10, 11, 12, 13])
    mx.new_catalog_from_resultset('my_cat', res, drop_if_exists=True)

    mycat = mx.catalogs['my_cat']
    mycat.merge_sources([9, 10], weights_colname='source_sum')
    mycat.add_column_with_merged_ids('MERGID')

    s100 = mycat.table.find_one(id=100)
    assert s100['merged'] is True
    assert s100['MERGID'] == '9,10'
    assert mycat.get_ids_merged_in(100) == [9, 10]

    s11 = mycat.table.find_one(id=11)
    assert s11['merged'] is None
    assert s11['MERGID'] == '11'

    tbl = mycat.select(mycat.c.active.is_(False)).as_table()
    assert_array_equal(tbl['id'], [9, 10])
    assert_array_equal(tbl['active'], False)
    assert_array_equal(tbl['merged_in'], 100)

    # check that reserved colnames cannot be used
    with pytest.raises(ValueError):
        mycat.add_column_with_merged_ids('merged')

    # check that it can be called twice (TODO: and updated, to be checked)
    mycat.add_column_with_merged_ids('MERGID')


def test_attach_dataset(mx):
    """Test attaching a dataset with the merging of sources.

    - merge 9 & 10 before
    - merge 11, 12 & 13 after

    """
    phot = mx.input_catalogs['photutils']
    res = phot.select(phot.c[phot.idname] > 7)
    mx.new_catalog_from_resultset('my_cat', res, drop_if_exists=True)

    mycat = mx.catalogs['my_cat']
    mycat.merge_sources([9, 10])
    mycat.attach_dataset(mx.muse_dataset, skip_existing=False,
                         mask_size=(10, 10), n_jobs=2)

    outdir = mycat.workdir / mx.muse_dataset.name
    flist = sorted(os.listdir(str(outdir)))
    assert flist[0] == 'mask-sky.fits'
    assert flist[1:] == ['mask-source-%05d.fits' % i
                         for i in (8, 11, 12, 13, 100)]

    segm = fits.getdata(mycat.segmap)
    sky = fits.getdata(str(outdir / 'mask-sky.fits'))

    assert segm.shape == sky.shape
    assert_array_equal(np.unique(sky), [0, 1])

    # FIXME: check why these are slightly different
    # assert_array_equal((segm == 0).astype(np.uint8), sky)
    assert_array_equal((segm[:89, :] == 0).astype(np.uint8), sky[:89, :])

    mask = fits.getdata(outdir / 'mask-source-00012.fits')
    assert mask.shape == (50, 50)
    assert np.count_nonzero(mask) == 92

    mname = outdir / 'mask-source-%05d.fits'
    totmask = sum([np.count_nonzero(fits.getdata(str(mname) % i))
                   for i in (11, 12, 13)])

    # check that the new mask is computed if possible, and that it corresponds
    # to the union of the sources mask
    mycat.merge_sources([11, 12, 13], dataset=mx.muse_dataset)
    mask = fits.getdata(outdir / 'mask-source-00101.fits')
    assert mask.shape == (50, 50)
    assert np.count_nonzero(mask) == totmask


def test_attach_origin_dataset(mx):
    """Test attaching an origin dataset."""
    orig = mx.input_catalogs['origin']

    assert orig.mask_tpl is not None
    assert orig.skymask_tpl is not None

    res = orig.select()
    mx.new_catalog_from_resultset('my_oricat', res, drop_if_exists=True)

    mycat = mx.catalogs['my_oricat']
    mycat.attach_dataset(mx.muse_dataset, skip_existing=False)

    outdir = mycat.workdir / mx.muse_dataset.name
    flist = sorted(os.listdir(str(outdir)))
    assert sorted(flist) == sorted(
        ['mask-%s-%05d.fits' % (t, i)
         for i in range(10) for t in ['sky', 'source']])


def test_export_marz(mx):
    phot = mx.input_catalogs['photutils']
    res = phot.select(phot.c[phot.idname] > 7,
                      columns=[phot.idname, phot.raname, phot.decname])
    mx.new_catalog_from_resultset('my_cat', res, drop_if_exists=True)
    mx.new_catalog_from_resultset('my_cat2', res, drop_if_exists=True)

    mycat = mx.catalogs['my_cat']
    mycat.merge_sources([9, 10])
    mycat.merge_sources([11, 12, 13])
    mycat.attach_dataset(mx.muse_dataset, skip_existing=False,
                         mask_size=(10, 10))

    refspec = ['MUSE_PSF_SKYSUB', 'MUSE_TOT_SKYSUB'] * 4
    mycat.update_column('refspec', refspec)

    outdir = f'{mx.workdir}/export'
    os.makedirs(outdir, exist_ok=True)
    mx.export_marz(mycat, export_sources=True)

    outdir = f'{mx.workdir}/export/hdfs/my_cat/marz'
    assert sorted(os.listdir(f'{outdir}')) == [
        'marz-my_cat-hdfs.fits', 'source-00008.fits', 'source-00100.fits',
        'source-00101.fits']

    with fits.open(f'{outdir}/marz-my_cat-hdfs.fits') as hdul:
        assert [hdu.name for hdu in hdul] == [
            'PRIMARY', 'WAVE', 'DATA', 'STAT', 'SKY', 'DETAILS']
        for name in ['WAVE', 'DATA', 'STAT', 'SKY']:
            assert hdul[name].shape == (3, 200)
        assert hdul['DETAILS'].data.dtype.names == (
            'NAME', 'RA', 'DEC', 'Z', 'CONFID', 'TYPE', 'F775W', 'F125W',
            'REFSPEC')
        assert_array_equal(
            hdul['DETAILS'].data['REFSPEC'],
            ['MUSE_PSF_SKYSUB', 'MUSE_PSF_SKYSUB', 'MUSE_TOT_SKYSUB'])

    marzfile = os.path.join(DATADIR, 'marz-my-cat-hdfs_SCO.mz')
    with pytest.raises(ValueError):
        mx.import_marz(marzfile, 'foobar')

    mx.import_marz(marzfile, mycat)
    assert len(mx.marzcat) == 3
    mx.import_marz(marzfile, 'my_cat2')
    assert len(mx.marzcat) == 6

    res = mx.marzcat.select_ids(8)
    assert res[0]['ID'] == 8
    assert res[0]['Type'] == 6
    assert res[0]['catalog'] == 'my_cat'
    assert res[1]['catalog'] == 'my_cat2'


def test_export_sources(mx):
    phot = mx.input_catalogs['photutils']
    res = phot.select(phot.c[phot.idname] > 7)
    mx.new_catalog_from_resultset('my_cat', res, drop_if_exists=True)

    mycat = mx.catalogs['my_cat']
    mycat.merge_sources([9, 10])
    mycat.merge_sources([11, 12, 13])
    mycat.attach_dataset(mx.muse_dataset, skip_existing=False,
                         convolve_fwhm=0.5, mask_size=(10, 10))

    outdir = f'{mycat.workdir}/export'
    os.makedirs(outdir, exist_ok=True)

    mx.export_sources(mycat, outdir=outdir, create_pdf=True, srcvers='0.1',
                      apertures=None, refspec='MUSE_PSF_SKYSUB')

    flist = os.listdir(outdir)
    assert sorted(flist) == ['source-00008.fits', 'source-00008.pdf',
                             'source-00100.fits', 'source-00100.pdf',
                             'source-00101.fits', 'source-00101.pdf']

    src = Source.from_file(f'{outdir}/source-00008.fits')
    assert src.REFSPEC == 'MUSE_PSF_SKYSUB'

    assert list(src.tables.keys()) == ['PHU_CAT']
    assert_array_equal(src.tables['PHU_CAT']['id'], [7, 8])

    assert list(src.cubes.keys()) == ['MUSE_CUBE']
    assert src.cubes['MUSE_CUBE'].shape == (200, 25, 25)

    assert list(src.images.keys()) == [
        'MUSE_WHITE', 'MUSE_EXPMAP', 'TEST_FAKE', 'PHU_SEGMAP', 'MASK_SKY',
        'MASK_OBJ']
    assert list(src.spectra.keys()) == [
        'MUSE_TOT', 'MUSE_WHITE', 'MUSE_PSF', 'MUSE_SKY', 'MUSE_TOT_SKYSUB',
        'MUSE_WHITE_SKYSUB', 'MUSE_PSF_SKYSUB']

    ref_header = """\
ID      =                    8 / object ID u.unitless %d
RA      =    338.2289866204975 / RA u.degree %.7f
DEC     =   -60.56824280312122 / DEC u.degree %.7f
FROM    = 'MuseX   '           / detection software
CUBE    = 'cube.fits'          / datacube
CUBE_V  = '1.24    '           / version of the datacube
SRC_V   = '0.1     '           / Source Version
CATALOG = 'photutils'
REFSPEC = 'MUSE_PSF_SKYSUB'    / Name of reference spectra
SIZE    =                    5
EXPMEAN =                 52.0 / Mean value of EXPMAP
EXPMIN  =                   52 / Minimum value of EXPMAP
EXPMAX  =                   52 / Maximum value of EXPMAP
FSFMODE = 'MOFFAT1 '
FSF00BET=                  2.8
FSF00FWA=                  0.8
FSF00FWB=               -3E-05
SEGMAP  = 'PHU_SEGMAP'
REFCAT  = 'PHU_CAT '
FSFMSK  =                  0.5 / Mask Conv Gauss FWHM in arcsec
NSKYMSK =                  434 / Size of MASK_SKY in spaxels
FSKYMSK =                69.44 / Relative Size of MASK_SKY in %
NOBJMSK =                   43 / Size of MASK_OBJ in spaxels
FOBJMSK =                 6.88 / Relative Size of MASK_OBJ in %
AUTHOR  = 'MPDAF   '           / Origin of the file
FORMAT  = '0.5     '           / Version of the Source format
"""

    cards = [fits.Card.fromstring(s) for s in ref_header.splitlines()]
    hdr = src.header
    for card in cards:
        assert hdr[card.keyword] == card.value
        assert hdr.comments[card.keyword] == card.comment


def test_join(mx):
    mycat = mx.catalogs['my_cat']
    photcat = mx.find_parent_cat(mycat)
    res = mycat.join([photcat, mx.marzcat],
                     whereclause=(mx.marzcat.c.catalog == mycat.name),
                     debug=True)
    # This gives only one result because merged sources do not have
    # a corresponding id in photcat
    assert len(res) == 1
    assert res[0]['my_cat_id'] == 8

    res = mycat.join([photcat, mx.marzcat],
                     whereclause=(mx.marzcat.c.catalog == mycat.name),
                     isouter=True, debug=True)
    # Now with outer join we get all results
    assert len(res) == 3
    assert_array_equal(res.as_table()['my_cat_id'], [8, 100, 101])


def test_merge_masks_on_area():
    mask_list = [fits.open(f"{DATADIR}/origin_masks/mask_1.fits")[1],
                 fits.open(f"{DATADIR}/origin_masks/mask_2.fits")[1],
                 fits.open(f"{DATADIR}/origin_masks/mask_3.fits")[1]]
    mask = fits.open(f"{DATADIR}/origin_masks/combined_masks.fits")[1]
    skymask = fits.open(f"{DATADIR}/origin_masks/combined_skymasks.fits")[1]

    ra, dec = 53.16559, -27.78124
    size = (60, 50)

    assert (masks.merge_masks_on_area(ra, dec, size, mask_list).data ==
            mask.data).all()
    assert (
        masks.merge_masks_on_area(ra, dec, size, mask_list, is_sky=True).data
        == skymask.data).all()

    # Check that the mask is at the correct position.
    # Use rounding method from astropy.nddata.utils
    def _round(a):
        '''Always round up.

        ``np.round`` cannot be used here, because it rounds .5 to the nearest
        even number.
        '''
        return int(np.floor(a + 0.5))

    wcs = WCS(masks.merge_masks_on_area(ra, dec, size, mask_list))
    center = wcs.all_world2pix(ra,  dec, 0)
    assert _round(center[0]) == _round(size[1] / 2)
    assert _round(center[1]) == _round(size[0] / 2)


def test_matching(mx):
    orig = mx.input_catalogs['origin']
    phot = mx.input_catalogs['photutils']
    cross = mx.cross_match("cross_matching", orig, phot)
    assert len(cross) == 18
    assert cross.cat1 is orig
    assert cross.cat2 is phot
    assert len(cross.matched_table_with_more_than(0)) == 5
