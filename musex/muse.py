import logging
import os

from .extractor import Extractor
from .settings import db

DIRNAME = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)

__all__ = ['MuseExtractor']


class MuseExtractor(Extractor):
    pass
