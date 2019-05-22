import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from mpdaf.sdetect import Source
from musex import Catalog, MuseX, masks
from musex.catalog import get_cat_name, get_result_table

CURDIR = os.path.abspath(os.path.dirname(__file__))
DATADIR = os.path.join(CURDIR, 'data')


def test_settings(capsys, settings_file):
    mx = MuseX(settings_file=settings_file, author='Me')
    assert mx.muse_dataset.name == 'hdfs'
    assert mx.conf['author'] == 'Me'

    expected = """\
muse_dataset   : hdfs
datasets       :
    - test            : small test dataset with images
    - photutils_masks : provide masks for the photutils catalog
    - origin          : provide masks and sources for the origin catalog
input_catalogs :
    - photutils       : 0 rows
    - origin          : 0 rows
"""
    captured = capsys.readouterr()
    assert expected in captured.out


def test_dataset(mx):
    dataset = mx.datasets['origin']
    assert repr(dataset) == (
        '<DataSet(prefix=ORIG, detector=origin, linked_cat=origin, '
        '1 detector, 1 tables, 2 sources, 2 masks)>')

    srcfile = dataset.get_source_file(3)
    assert srcfile.endswith('origin_sources/source-00003.fits')

    assert dataset.get_source_refspec(3) == 'ORI_CORR_3_SKYSUB'


def test_ingest_photutils(mx):
    assert list(mx.input_catalogs.keys()) == ['photutils', 'origin']

    # Photutils catalog
    phot = mx.input_catalogs['photutils']
    assert phot.idname == 'id'
    assert phot.raname == 'ra'
    assert phot.decname == 'dec'

    phot.ingest_input_catalog(limit=3)
    assert len(phot.select()) == 3

    phot.ingest_input_catalog(upsert=True)
    assert len(phot.select()) == 13
    assert phot.meta['maxid'] == 13

    tbl = phot.select().as_table()
    assert tbl[phot.idname].max() == 13

    log = list(phot.get_log(3))
    assert log[0]['id'] == 3
    assert log[0]['catalog'] == 'photutils'
    assert log[0]['msg'] == 'inserted from input catalog'
    assert log[0]['author'] == 'John Doe'
    assert log[1]['catalog'] == 'photutils'
    assert log[1]['msg'] == 'updated from input catalog'
    assert '"eccentricity"' in log[1]['data']


def test_ingest_origin(mx):
    assert list(mx.input_catalogs.keys()) == ['photutils', 'origin']

    # Catalogue origin
    orig = mx.input_catalogs['origin']
    assert orig.idname == 'ID'
    assert orig.raname == 'ra'
    assert orig.decname == 'dec'

    orig.ingest_input_catalog()
    assert len(orig.select()) == 4
    assert orig.meta['status'] == 'inserted'

    # Test that ingesting again does not crash
    orig.ingest_input_catalog()

    orig.ingest_input_catalog(upsert=True)
    assert len(orig.select()) == 4
    assert orig.meta['maxid'] == 4
    assert orig.meta['status'] == 'updated'

    tbl = orig.select().as_table()
    assert tbl[orig.idname].max() == 4

    assert orig.meta['version_meta'] == 'CAT3_TS'
    assert orig.meta['CAT3_TS'] == "2019-04-12T18:05:59.435812"


def test_catalog_name(settings_file):
    mx = MuseX(settings_file=settings_file, show_banner=False, db=':memory:')
    with pytest.warns(UserWarning,
                      match='catalog name should contain only ascii letters '):
        Catalog('dont-use-dash', mx.db)


def test_catalog(mx):
    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()
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

    res = phot.select_id(1)
    assert res['id'] == 1
    assert res['area'] == 79
    phot.update_id(1, area=100)
    assert phot.select_id(1)['area'] == 100

    # Test selection with a wcs
    res = phot.select(wcs=mx.muse_dataset.white.wcs, columns=[phot.idname],
                      margin=2/0.2)
    assert len(res) == 8

    # Test selection with a WCS and a mask
    res = phot.select(wcs=mx.muse_dataset.white.wcs,
                      mask=np.ones(mx.muse_dataset.white.data.shape,
                                   dtype=bool))
    assert len(res) == 0


def test_resultset(mx):
    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()
    res = phot.select()

    # test get_cat_name
    assert get_cat_name('photutils') == 'photutils'
    assert get_cat_name(phot) == 'photutils'
    assert get_cat_name(res) == 'photutils'
    assert get_cat_name(res.as_table()) == 'photutils'

    # test get_result_table
    assert len(get_result_table(phot)) == 13
    assert len(get_result_table(res)) == 13
    assert len(get_result_table(res.as_table())) == 13


def test_user_catalog(mx):
    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()

    res = phot.select(phot.c[phot.idname] < 5)
    mx.new_catalog_from_resultset('my_cat', res)

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


def test_user_catalog_from_scratch(mx):
    """
    Test user catalogs: make sure that it works without ra/dec columns, and
    than bytes data are decoded.
    """
    # From file
    cat = mx.new_catalog('custom_cat', idname='id')
    catfile = os.path.join(DATADIR, 'cat_comments.fits')
    tbl = Table.read(catfile)
    cat.insert(tbl)
    assert 'custom_cat' in mx.catalogs
    assert cat.meta['type'] == 'user'
    assert cat.meta['maxid'] == 4
    rows = cat.select()
    assert rows[0]['id'] == 1
    assert rows[0]['comment'] == 'foo'
    assert rows[-1]['comment'] == 'a long comment'

    # From data
    cat2 = mx.new_catalog('custom_cat2', idname='id')
    cat2.insert([{'id': 2, 'foo': 'bar'}])
    cat2.insert([{'id': 4, 'baz': True}])
    assert len(cat2.table) == 2
    assert cat2.meta['raname'] is None
    assert cat2.meta['decname'] is None


def test_drop_user_catalog(mx):
    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()

    res = phot.select(phot.c[phot.idname] < 5)

    catname = 'my_cat_tmp'
    mx.new_catalog_from_resultset(catname, res)
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


def test_update_column(mx, caplog):
    # Test addition of a new column
    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()

    res = phot.select(phot.c[phot.idname] < 5)
    mycat = mx.new_catalog_from_resultset('my_cat', res)

    assert 'foobar' not in mycat.table.columns
    mycat.update_column('foobar', 1)
    assert 'foobar' in mycat.table.columns
    assert mycat.table.find_one()['foobar'] == 1

    mycat.update_column('foobar', range(len(mycat)))
    res = mycat.select_ids([1, 2, 3])
    tbl = res.as_table()
    assert_array_equal(tbl['foobar'], [0, 1, 2])

    # creation of new column with dash
    with pytest.raises(ValueError):
        mycat.insert([{'id': 402, 'foo-bar': 42}])

    caplog.clear()
    t = Table([[111], [42]], names=('id', 'with-dash'))
    mycat.insert(t)
    assert caplog.records[0].message == \
        'The column with-dash was renamed to with_dash.'


def test_update_rows(mx):
    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()

    res = phot.select(phot.c[phot.idname] < 5)
    mycat = mx.new_catalog_from_resultset('my_cat', res)

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

    # wrong keys in upsert
    with pytest.raises(KeyError):
        mycat.upsert(res, show_progress=False, keys=['foo'])
    with pytest.raises(KeyError):
        mycat.upsert(tbl, show_progress=False, keys=['foo'])


def test_id_mapping(mx):
    mx.create_id_mapping('mapping_name')
    meta = mx.db['catalogs'].find_one(name='mapping_name')
    assert meta['type'] == 'id'
    assert mx.id_mapping is not None

    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()

    res = phot.select(phot.c[phot.idname] > 8)
    mycat = mx.new_catalog_from_resultset('my_cat', res)

    mycat_ids = [x['id'] for x in mycat.select()]
    mx.id_mapping.add_ids(mycat_ids, mycat)
    assert mx.id_mapping.table.columns == ['ID', 'my_cat_id']
    assert mycat.table.count() == 5
    assert mx.id_mapping.table.count() == 5

    # By default insert row does not change id_mapping
    mycat.insert([{'ra': 1, 'dec': 2}])
    assert mycat.table.count() == 6
    assert mx.id_mapping.table.count() == 5

    # # It does update with the context manager
    # with mx.use_id_mapping(mycat):
    #     mycat.insert([{'id': 20, 'ra': 3, 'dec': 4}])
    #     assert mycat.table.count() == 7
    #     assert mx.id_mapping.table.count() == 6
    #     res = mx.id_mapping.select(limit=1, order_by=desc('ID')).results[0]
    #     assert res['ID'] == 6
    #     assert res['my_cat_id'] == 20

    # # And same for select and update
    # assert mycat.select_id(6) is None
    # assert mycat.select_id(20) is not None

    # mycat.update_id(10, area=200)
    # assert mycat.select_id(10)['area'] == 200

    # with mx.use_id_mapping(mycat):
    #     assert mycat.select_id(20) is None
    #     assert mycat.select_id(6)['id'] == 20

    #     mycat.update_id(6, area=100)
    #     assert mycat.select_id(6)['area'] == 100


def test_merge_sources(mx):
    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()

    res = phot.select_ids([9, 10, 11, 12, 13])
    mycat = mx.new_catalog_from_resultset('my_cat', res)

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


def test_export_marz(mx):
    # setup
    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()
    res = phot.select(phot.c[phot.idname] > 11)
    mycat = mx.new_catalog_from_resultset('my_cat', res)

    maskdir = os.path.join(mx.workdir, 'masks', 'hdfs')
    mx.create_masks_from_segmap(phot, maskdir, skip_existing=True, margin=10)

    refspec = ['MUSE_PSF_SKYSUB', 'MUSE_TOT_SKYSUB']
    mycat.update_column('refspec', refspec)

    mx.export_marz(mycat, export_sources=True, masks_dataset='photutils_masks')

    outdir = f'{mx.workdir}/export/hdfs/my_cat/marz'
    assert sorted(os.listdir(f'{outdir}')) == [
        'marz-my_cat-hdfs.fits', 'source-00012.fits', 'source-00013.fits']

    with fits.open(f'{outdir}/marz-my_cat-hdfs.fits') as hdul:
        assert [hdu.name for hdu in hdul] == [
            'PRIMARY', 'WAVE', 'DATA', 'STAT', 'SKY', 'DETAILS']
        for name in ['WAVE', 'DATA', 'STAT', 'SKY']:
            assert hdul[name].shape == (2, 200)
        assert hdul['DETAILS'].data.dtype.names == (
            'NAME', 'RA', 'DEC', 'Z', 'CONFID', 'TYPE', 'F775W', 'F125W',
            'REFSPEC')
        assert_array_equal(hdul['DETAILS'].data['REFSPEC'],
                           ['MUSE_PSF_SKYSUB', 'MUSE_TOT_SKYSUB'])


def test_import_marz(mx):
    # setup
    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()
    res = phot.select(phot.c[phot.idname] > 11)
    mycat = mx.new_catalog_from_resultset('my_cat', res)

    marzfile = os.path.join(DATADIR, 'marz-my-cat-hdfs_SCO.mz')
    with pytest.raises(ValueError):
        mx.import_marz(marzfile, 'foobar')

    mx.import_marz(marzfile, mycat)
    assert len(mx.marzcat) == 3

    res = mx.marzcat.select_ids(8)
    assert res[0]['ID'] == 8
    assert res[0]['Type'] == 6
    assert res[0]['catalog'] == 'my_cat'

    # MarZ one solution per row catalog creation
    marz_sol = mx.new_catalog_from_resultset(
        name="marzsol", resultset=mx.marzcat.select_flat(), primary_id="_id")
    assert len(marz_sol) == 15

    res = mx.marzcat.select_flat(limit_to_cat="my_cat", max_order=2)
    marz_sol = mx.new_catalog_from_resultset(
        name="marzsol", resultset=res, primary_id="_id", drop_if_exists=True)
    assert len(marz_sol) == 6


def test_export_marz_from_sources(mx):
    # setup
    orig = mx.input_catalogs['origin']
    orig.ingest_input_catalog()
    res = orig.select(orig.c[orig.idname] > 2)
    mycat = mx.new_catalog_from_resultset('my_oricat', res)

    # FIXME: By default we use REFSPEC from the origin sources but it
    # could be a good idea to test also that settings refspec works.
    # Currently it doesn't.
    # refspec = ['ORI_SPEC_1', 'ORI_SPEC_1']
    # mycat.update_column('refspec', refspec)

    mx.export_marz(mycat, sources_dataset='origin', skyspec='MUSE_SKY')

    outdir = f'{mx.workdir}/export/hdfs/my_oricat/marz'
    assert sorted(os.listdir(f'{outdir}')) == ['marz-my_oricat-hdfs.fits']

    with fits.open(f'{outdir}/marz-my_oricat-hdfs.fits') as hdul:
        assert [hdu.name for hdu in hdul] == [
            'PRIMARY', 'WAVE', 'DATA', 'STAT', 'SKY', 'DETAILS']
        for name in ['WAVE', 'DATA', 'STAT', 'SKY']:
            assert hdul[name].shape == (2, 200)
        assert hdul['DETAILS'].data.dtype.names == (
            'NAME', 'RA', 'DEC', 'Z', 'CONFID', 'TYPE', 'F775W', 'F125W',
            'REFSPEC')
        assert_array_equal(hdul['DETAILS'].data['REFSPEC'],
                           ['ORI_CORR_3_SKYSUB', 'ORI_CORR_4_SKYSUB'])


def test_export_sources(mx):
    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()

    res = phot.select((phot.c[phot.idname] > 7) & (phot.c[phot.idname] < 10))
    mycat = mx.new_catalog_from_resultset('my_cat', res)

    maskdir = os.path.join(mx.workdir, 'masks', 'hdfs')
    mx.create_masks_from_segmap(phot, maskdir, skip_existing=True, margin=10)

    outdir = f'{mx.workdir}/export'
    mx.export_sources(mycat, outdir=outdir, srcvers='0.1',
                      apertures=None, refspec='MUSE_PSF_SKYSUB', n_jobs=2,
                      verbose=True, masks_dataset='photutils_masks',
                      extra_header={'FOO': 'BAR'}, catalogs=[phot],
                      segmap=True)

    flist = os.listdir(outdir)
    assert sorted(flist) == ['source-00008.fits', 'source-00009.fits']

    src = Source.from_file(f'{outdir}/source-00008.fits')
    assert src.REFSPEC == 'MUSE_PSF_SKYSUB'
    assert src.FOO == 'BAR'

    assert list(src.tables.keys()) == ['PHU_CAT']
    assert_array_equal(src.tables['PHU_CAT']['id'], [8, 7])

    assert list(src.cubes.keys()) == ['MUSE_CUBE']
    assert src.cubes['MUSE_CUBE'].shape == (200, 25, 25)

    assert set(src.images.keys()) == {
        'MUSE_WHITE', 'MUSE_EXPMAP', 'TEST_FAKE', 'PHU_SEGMAP', 'MASK_SKY',
        'MASK_OBJ'}
    assert list(src.spectra.keys()) == [
        'MUSE_TOT', 'MUSE_WHITE', 'MUSE_PSF', 'MUSE_SKY', 'MUSE_TOT_SKYSUB',
        'MUSE_WHITE_SKYSUB', 'MUSE_PSF_SKYSUB']

    ref_header = """\
ID      =                    8 / object ID %d
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
NSKYMSK =                  467 / Size of MASK_SKY in spaxels
FSKYMSK =                74.72 / Relative Size of MASK_SKY in %
NOBJMSK =                   25 / Size of MASK_OBJ in spaxels
FOBJMSK =                  4.0 / Relative Size of MASK_OBJ in %
AUTHOR  = 'MPDAF   '           / Origin of the file
FORMAT  = '0.6     '           / Version of the Source format
"""

    cards = [fits.Card.fromstring(s) for s in ref_header.splitlines()]
    hdr = src.header
    for card in cards:
        assert hdr[card.keyword] == card.value
        assert hdr.comments[card.keyword] == card.comment


def test_export_sources_origin(mx):
    orig = mx.input_catalogs['origin']
    orig.ingest_input_catalog()
    res = orig.select(orig.c[orig.idname] > 2)
    mycat = mx.new_catalog_from_resultset('my_oricat', res)

    outdir = f'{mx.workdir}/export'
    mx.export_sources(mycat, outdir=outdir, srcvers='0.1', apertures=None,
                      verbose=True, masks_dataset='origin', catalogs=[orig])

    flist = os.listdir(outdir)
    assert sorted(flist) == ['source-00003.fits', 'source-00004.fits']

    src = Source.from_file(f'{outdir}/source-00003.fits')
    assert src.REFSPEC == 'MUSE_TOT_SKYSUB'

    # FIXME: why ORI_CAT and ORIG_CAT? ORI_CAT seems duplicate
    assert list(src.tables.keys()) == ['ORIG_ORI_CAT', 'ORIG_ORI_LINES',
                                       'ORIG_NB_PAR', 'ORIG_CAT']
    assert_array_equal(src.tables['ORIG_CAT']['ID'], [3, 1])

    assert list(src.cubes.keys()) == ['MUSE_CUBE', 'ORIG_ORI_CORREL']
    assert src.cubes['MUSE_CUBE'].shape == (200, 25, 25)

    assert set(src.images.keys()) == {
        'MASK_SKY', 'ORIG_ORI_SEGMAP_LABEL', 'MUSE_EXPMAP', 'TEST_FAKE',
        'ORIG_ORI_MASK_SKY', 'ORIG_ORI_MASK_OBJ', 'ORIG_ORI_CORR_3',
        'MASK_OBJ', 'ORIG_NB_LINE_3', 'ORIG_ORI_SEGMAP_MERGED', 'MUSE_WHITE',
        'ORIG_ORI_MAXMAP'
    }

    assert list(src.spectra.keys()) == [
        'ORIG_MUSE_SKY', 'ORIG_MUSE_TOT_SKYSUB', 'ORIG_MUSE_WHITE_SKYSUB',
        'ORIG_MUSE_TOT', 'ORIG_ORI_CORR', 'ORIG_MUSE_PSF_SKYSUB',
        'ORIG_MUSE_PSF', 'ORIG_ORI_SPEC_3', 'ORIG_ORI_CORR_3_SKYSUB',
        'ORIG_ORI_CORR_3', 'MUSE_TOT', 'MUSE_WHITE', 'MUSE_PSF', 'MUSE_SKY',
        'MUSE_TOT_SKYSUB', 'MUSE_WHITE_SKYSUB', 'MUSE_PSF_SKYSUB'
    ]

    ref_header = """\
ID      =                    3 / object ID %d
RA      =    338.2311010253501 / RA u.degree %.7f
DEC     =   -60.56595462398222 / DEC u.degree %.7f
FROM    = 'MuseX   '           / detection software
CUBE    = 'cube.fits'          / datacube
CUBE_V  = '1.24    '           / version of the datacube
SRC_V   = '0.1     '           / Source Version
SIZE    =                    5
EXPMEAN =                 52.0 / Mean value of EXPMAP
EXPMIN  =                   52 / Minimum value of EXPMAP
EXPMAX  =                   52 / Maximum value of EXPMAP
FSFMODE = 'MOFFAT1 '
FSF00BET=                  2.8
FSF00FWA=                  0.8
FSF00FWB=               -3E-05
NSKYMSK =                  370 / Size of MASK_SKY in spaxels
FSKYMSK =                 59.2 / Relative Size of MASK_SKY in %
NOBJMSK =                   64 / Size of MASK_OBJ in spaxels
FOBJMSK =                10.24 / Relative Size of MASK_OBJ in %
CATALOG = 'origin  '
REFSPEC = 'MUSE_TOT_SKYSUB'    / Name of reference spectra
REFCAT  = 'ORIG_CAT'
AUTHOR  = 'MPDAF   '           / Origin of the file
FORMAT  = '0.6     '           / Version of the Source format
"""

    cards = [fits.Card.fromstring(s) for s in ref_header.splitlines()]
    hdr = src.header
    for card in cards:
        assert hdr[card.keyword] == card.value
        assert hdr.comments[card.keyword] == card.comment


def test_join(mx):
    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()
    res = phot.select(phot.c[phot.idname] > 7)
    mycat = mx.new_catalog_from_resultset('my_cat', res)
    mycat.merge_sources([9, 10])
    mycat.merge_sources([11, 12, 13])

    marzfile = os.path.join(DATADIR, 'marz-my-cat-hdfs_SCO.mz')
    mx.import_marz(marzfile, mycat)

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
    mask_list = [
        fits.open(f"{DATADIR}/origin_masks/source-mask-00001.fits")[1],
        fits.open(f"{DATADIR}/origin_masks/source-mask-00002.fits")[1],
        fits.open(f"{DATADIR}/origin_masks/source-mask-00003.fits")[1]
    ]
    # mask = fits.open(f"{DATADIR}/origin_masks/combined_masks.fits")[1]
    # skymask = fits.open(f"{DATADIR}/origin_masks/combined_skymasks.fits")[1]

    ra, dec = 338.2302796, -60.5662872
    size = (60, 50)

    # assert (masks.merge_masks_on_area(ra, dec, size, mask_list).data ==
    #         mask.data).all()
    # assert (
    #     masks.merge_masks_on_area(ra, dec, size, mask_list, is_sky=True).data
    #     == skymask.data).all()

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
    # FIXME : don't know why but the new masks have a one pixel difference...
    assert _round(center[0]) == _round(size[1] / 2) - 1
    assert _round(center[1]) == _round(size[0] / 2) - 1


def test_matching(mx):
    orig = mx.input_catalogs['origin']
    orig.ingest_input_catalog()

    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()

    cross = mx.cross_match("cross_matching", orig, phot)
    assert len(cross) == 15
    assert cross.cat1 is orig
    assert cross.cat2 is phot
    assert len(cross.matched_table_with_more_than(0)) == 2


def test_export_match(mx):
    """Test export for a matched catalog, combining multiple datasets and
    catalogs.
    """
    orig = mx.input_catalogs['origin']
    orig.ingest_input_catalog()

    phot = mx.input_catalogs['photutils']
    phot.ingest_input_catalog()

    cross = mx.cross_match("cross_matching", orig, phot)
    res = cross.select(
        # whereclause='photutils_id NOT NULL AND origin_id NOT NULL',
        columns=['ID', 'photutils_id', 'origin_id']
    )
    mycat = mx.new_catalog('my_cat', idname='ID', raname='ra', decname='dec')
    mycat.insert(res)

    # get ra/dec from photutils
    tbl = phot.select(phot.c.id == mycat.c.photutils_id,
                      columns=['id', 'ra', 'dec', 'area']).as_table()
    tbl.rename_column('id', 'photutils_id')
    mycat.upsert(tbl, keys=['photutils_id'])

    # get ra/dec from origin
    tbl = orig.select(orig.c.ID == mycat.c.origin_id,
                      columns=['ID', 'ra', 'dec', 'purity']).as_table()
    tbl.rename_column('ID', 'origin_id')
    mycat.upsert(tbl, keys=['origin_id'])

    # add mask dataset
    maskds = ['origin' if row['origin_id'] else 'photutils_masks'
              for row in mycat.select()]
    mycat.update_column('mask_dataset', maskds)

    # we need to photutils id for the photutils_masks as well...
    tbl = mycat.select(columns=['photutils_id']).as_table()
    tbl['photutils_masks_id'] = tbl['photutils_id']
    mycat.upsert(tbl, keys=['photutils_id'])

    # and create the masks
    maskdir = os.path.join(mx.workdir, 'masks', 'hdfs')
    mx.create_masks_from_segmap(phot, maskdir, skip_existing=True, margin=10)

    outdir = f'{mx.workdir}/export'
    with mx.use_loglevel('DEBUG'):
        mx.export_sources(
            mycat.select_ids([1, 3, 6]),
            outdir=outdir,
            srcvers='0.1',
            verbose=True,
            datasets={'origin': ['ORI*', 'MUSE_CUBE', 'MUSE_WHITE']}
        )

    flist = os.listdir(outdir)
    assert sorted(flist) == ['source-00001.fits', 'source-00003.fits',
                             'source-00006.fits']

    # source detected with both
    src = Source.from_file(f'{outdir}/source-00001.fits')
    assert src.REFSPEC == 'MUSE_TOT_SKYSUB'
    assert src.PURITY == 0
    assert 'MUSE_CUBE' in src.cubes
    assert 'MUSE_WHITE' in src.images
    assert 'ORIG_NB_LINE_3' not in src.images   # 3 because ID=1 matches with
    assert 'ORIG_ORI_SPEC_3' in src.spectra     # origin source #3

    # origin only source
    src = Source.from_file(f'{outdir}/source-00003.fits')
    assert src.REFSPEC == 'MUSE_TOT_SKYSUB'
    assert src.PURITY == 0

    # photutils only source
    src = Source.from_file(f'{outdir}/source-00006.fits')
    assert src.REFSPEC == 'MUSE_TOT_SKYSUB'
    assert 'PURITY' not in src.header
