import matplotlib  # isort:skip

matplotlib.use("Agg")  # noqa isort:skip

import os
import shutil

import pytest

from musex import MuseX

CURDIR = os.path.abspath(os.path.dirname(__file__))
DATADIR = os.path.join(CURDIR, "data")
WORKDIR = ""


@pytest.fixture(scope="session")
def workdir(tmpdir_factory):
    """Ensure that build directory does not exists before each test."""
    tmpdir = str(tmpdir_factory.mktemp("musex"))
    print("create tmpdir:", tmpdir)
    with open(os.path.join(DATADIR, "settings.yaml"), "r") as f:
        out = f.read().format(tmpdir=tmpdir, datadir=DATADIR, db=":memory:")

    with open(os.path.join(tmpdir, "settings.yaml"), "w") as f:
        f.write(out)

    # FIXME: keeping temp directory for now
    # yield tmpdir
    # shutil.rmtree(tmpdir)

    return tmpdir


@pytest.fixture
def settings_file(workdir):
    """Return the sample config file path."""
    return os.path.join(workdir, "settings.yaml")


@pytest.fixture
def mx(settings_file):
    mx = MuseX(settings_file=settings_file, show_banner=False)
    yield mx
    shutil.rmtree(f"{mx.workdir}/export", ignore_errors=True)


# def pytest_report_header(config):
#     return "project deps: Pillow-{}".format(PIL.PILLOW_VERSION)
