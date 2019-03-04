import matplotlib
matplotlib.use('Agg')  # noqa

import os
import pytest

from musex import MuseX

CURDIR = os.path.abspath(os.path.dirname(__file__))
DATADIR = os.path.join(CURDIR, 'data')
WORKDIR = ''


@pytest.fixture(scope='session')
def workdir(tmpdir_factory):
    """Ensure that build directory does not exists before each test."""
    tmpdir = str(tmpdir_factory.mktemp('musex'))
    print('create tmpdir:', tmpdir)
    with open(os.path.join(DATADIR, 'settings.yaml'), 'r') as f:
        out = f.read().format(tmpdir=tmpdir, datadir=DATADIR, db=':memory:')

    with open(os.path.join(tmpdir, 'settings.yaml'), 'w') as f:
        f.write(out)

    # FIXME: keeping temp directory for now
    # yield tmpdir
    # shutil.rmtree(tmpdir)

    return tmpdir


@pytest.fixture
def settings_file(workdir):
    """Return the sample config file path."""
    return os.path.join(workdir, 'settings.yaml')


@pytest.fixture
def mx(settings_file):
    return MuseX(settings_file=settings_file, show_banner=False)


# def pytest_report_header(config):
#     return "project deps: Pillow-{}".format(PIL.PILLOW_VERSION)
