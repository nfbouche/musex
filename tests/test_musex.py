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

    phot.ingest_input_catalog()
    assert len(phot.select()) == 13

    phot.ingest_input_catalog()
    assert len(phot.select()) == 13
    assert phot.meta['maxid'] == '13'

    tbl = phot.select().as_table()
    assert tbl[phot.idname].max() == 13
