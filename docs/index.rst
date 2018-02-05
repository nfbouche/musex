.. MuseX documentation master file, created by
   sphinx-quickstart on Mon Feb  5 09:38:29 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MuseX's documentation!
=================================

MuseX is a Python package which allows to manage catalogs of sources, with all
the steps required to extract spectra from MUSE datacubes:

- Use input catalogs from priors (e.g. HST) or detection softwares.
- Create extraction masks from segmentation maps.
- Extract spectra.
- Export to MarZ and import Marz results.
- TODO: Export to Platefit and import Platefit results.
- Export MPDAF sources with all the gathered information.

Contents
--------

.. toctree::
   :maxdepth: 2

   MuseX-Photutils


API
---

.. automodapi:: musex
   :no-inheritance-diagram:
