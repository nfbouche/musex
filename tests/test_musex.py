import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from musex import MuseX


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
    assert repr(res) == "<ResultSet(photutils.id < 5)>, 4 results"
    assert len(res) == 4
    assert res[0][phot.idname] == 1

    res = phot.select_ids([1, 2, 3], columns=[phot.idname])
    tbl = res.as_table()
    assert str(tbl.whereclause) == 'photutils.id IN (:id_1, :id_2, :id_3)'
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
    __import__('pdb').set_trace()
    assert catname not in mx.db.tables
    assert catname not in mx.catalogs


def test_segmap(mx):
    mycat = mx.catalogs['my-cat']
    segmap = mycat.get_segmap_aligned(mx.muse_dataset)

    assert segmap.img.shape == (90, 90)
    assert segmap.img.dtype == np.int64
    assert np.max(segmap.img._data) == 13
    assert np.all(np.unique(segmap.img._data) == np.arange(14))
