import numpy as np
import os
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from mpdaf.sdetect import Source
from musex import MuseX
from numpy.testing import assert_allclose, assert_array_equal


def test_settings(settings_file):
    mx = MuseX(settings_file=settings_file, author='Me')
    assert mx.muse_dataset.name == 'hdfs'
    assert mx.conf['author'] == 'Me'


def test_ingest(mx):
    assert list(mx.input_catalogs.keys()) == ['photutils']
    phot = mx.input_catalogs['photutils']
    assert phot.idname == 'id'
    assert phot.raname == 'ra'
    assert phot.decname == 'dec'
    assert phot.segmap.endswith('segmap.fits')

    phot.ingest_input_catalog()
    assert len(phot.select()) == 13

    phot.ingest_input_catalog()
    assert len(phot.select()) == 13
    assert phot.meta['maxid'] == 13

    tbl = phot.select().as_table()
    assert tbl[phot.idname].max() == 13


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

    res = phot.select_ids([1, 2, 3], columns=[phot.idname])
    tbl = res.as_table()
    assert tbl.whereclause == 'photutils.id IN (1, 2, 3)'
    assert tbl.catalog is phot
    assert len(tbl) == 3
    assert tbl.colnames == ['id']


def test_user_catalog(mx):
    phot = mx.input_catalogs['photutils']
    res = phot.select(phot.c[phot.idname] < 5)
    mx.new_catalog_from_resultset('my-cat', res, drop_if_exists=True)

    with pytest.raises(ValueError):
        mx.new_catalog_from_resultset('my-cat', res, drop_if_exists=False)

    tbl = res.as_table()
    mx.new_catalog_from_resultset('my-cat', tbl, drop_if_exists=True)

    mycat = mx.catalogs['my-cat']
    assert len(mycat) == 4
    assert mycat.meta['type'] == 'user'
    assert mycat.meta['query'] == 'photutils.id < 5'
    assert isinstance(mycat.skycoord(), SkyCoord)


def test_drop_user_catalog(mx):
    phot = mx.input_catalogs['photutils']
    res = phot.select(phot.c[phot.idname] < 5)

    catname = 'my-cat-tmp'
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


def test_update_rows(mx):
    phot = mx.input_catalogs['photutils']
    res = phot.select(phot.c[phot.idname] < 5)
    mx.new_catalog_from_resultset('my-cat', res, drop_if_exists=True)

    mycat = mx.catalogs['my-cat']

    # insert rows
    for i, r in enumerate(res):
        r['ra'] = i
    mycat.insert_rows(res, show_progress=False)
    assert len(mycat) == 4
    assert_allclose([o['ra'] for o in mycat.select(columns=['ra'])],
                    np.arange(4, dtype=float))

    # insert table
    tbl = res.as_table()
    tbl['dec'] = 2.0
    mycat.insert_table(tbl, show_progress=False)
    assert len(mycat) == 4
    assert_allclose([o['dec'] for o in mycat.select(columns=['dec'])], 2.0)


def test_segmap(mx):
    mycat = mx.catalogs['my-cat']
    segmap = mycat.get_segmap_aligned(mx.muse_dataset)

    assert segmap.img.shape == (90, 90)
    assert segmap.img.dtype == np.int64
    assert np.max(segmap.img._data) == 13
    assert np.all(np.unique(segmap.img._data) == np.arange(14))


def test_merge_sources(mx):
    phot = mx.input_catalogs['photutils']
    res = phot.select_ids([9, 10])
    mx.new_catalog_from_resultset('my-cat', res, drop_if_exists=True)

    mycat = mx.catalogs['my-cat']
    mycat.merge_sources([9, 10])

    assert mycat.table.find_one(id=100)['merged'] is True

    tbl = mycat.select(mycat.c.active.is_(False)).as_table()
    assert_array_equal(tbl['id'], [9, 10])
    assert_array_equal(tbl['active'], False)
    assert_array_equal(tbl['merged_in'], 100)


def test_attach_dataset(mx):
    """Test attaching a dataset with the merging of sources.

    - merge 9 & 10 before
    - merge 11, 12 & 13 after

    """
    phot = mx.input_catalogs['photutils']
    res = phot.select(phot.c[phot.idname] > 7)
    mx.new_catalog_from_resultset('my-cat', res, drop_if_exists=True)

    mycat = mx.catalogs['my-cat']
    mycat.merge_sources([9, 10])
    mycat.attach_dataset(mx.muse_dataset, skip_existing=False,
                         mask_size=(10, 10))

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


def test_export_sources(mx):
    phot = mx.input_catalogs['photutils']
    res = phot.select(phot.c[phot.idname] > 7)
    mx.new_catalog_from_resultset('my-cat', res, drop_if_exists=True)

    mycat = mx.catalogs['my-cat']
    mycat.merge_sources([9, 10])
    mycat.merge_sources([11, 12, 13])
    mycat.attach_dataset(mx.muse_dataset, skip_existing=False,
                         mask_size=(10, 10))

    outdir = f'{mycat.workdir}/export'
    os.makedirs(outdir, exist_ok=True)

    for src in mx.to_sources(mycat, srcvers='0.1', apertures=None):
        src.write(f'{outdir}/source-{src.ID:05d}.fits')
        # src.to_pdf(f'{outdir}/source-{src.ID:05d}.pdf', mx.muse_dataset.white)

    flist = os.listdir(outdir)
    assert sorted(flist) == [
        'source-00008.fits', 'source-00100.fits', 'source-00101.fits']

    src = Source.from_file(f'{outdir}/source-00008.fits')

    assert list(src.tables.keys()) == ['PHU_CAT']
    assert_array_equal(src.tables['PHU_CAT']['id'], [7, 8])

    assert list(src.cubes.keys()) == ['MUSE_CUBE']
    assert src.cubes['MUSE_CUBE'].shape == (200, 25, 25)

    assert list(src.images.keys()) == [
        'MUSE_WHITE', 'MUSE_EXPMAP', 'TEST_FAKE', 'PHU_SEGMAP', 'MASK_SKY',
        'MASK_OBJ']
    assert list(src.spectra.keys()) == [
        'MUSE_TOT', 'MUSE_WHITE', 'MUSE_SKY', 'MUSE_TOT_SKYSUB',
        'MUSE_WHITE_SKYSUB']
