import datetime
import logging
import matplotlib.pylab as plt
import os

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from mpdaf.sdetect import Catalog

from .plots import (
    err_report, report_error,
    show_3dhst,
    show_comments,
    show_contspec,
    show_errors,
    show_field,
    show_fullspec,
    show_hstima,
    show_image,
    show_info,
    show_maxmap,
    show_nb,
    show_origspec,
    show_pfitline,
    show_pfitspec,
    show_title,
    show_white,
    show_zoomspec,
)

from .version import __version__

HST_TAGS = ['HST_F225W', 'HST_F336W', 'HST_F435W', 'HST_F606W', 'HST_F775W',
            'HST_F814W', 'HST_F850LP', 'HST_F105W', 'HST_F125W', 'HST_F140W',
            'HST_F160W']


def get_hstkeys(src, tags=HST_TAGS):
    return [tag for tag in tags if tag in src.images]


def create_pdf(src, outfile, mastercat=None, debug=False, white='MUSE_WHITE',
               ima2='HST_F775W'):
    logger = logging.getLogger(__name__)

    pdf_pages = PdfPages(outfile)

    white = src.images[white]

    if mastercat is not None:
        if isinstance(mastercat, str):
            cat = src.tables[mastercat]
            # cat = Catalog.read(mastercat)
        else:
            cat = Catalog(mastercat)
        # cat = cat[cat[args['colactive']]]  # filter out non-active src
        # cat.name = os.path.basename(mastercat)
        # cat.rename_column(args['colid'], 'id')
        # cat.rename_column(args['coltype'], 'type')
        # cat.rename_column(args['colhasz'], 'hasz')
        # cat.rename_column(args['colz'], 'z')
        # cat.rename_column(args['colconfid'], 'confid')
    else:
        cat = None

    ax = {}
    # first page
    pagenum = 1
    fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    # title box
    gs = GridSpec(1, 1, left=0.02, right=0.75, top=0.95, bottom=0.935)
    ax['TITLE'] = plt.subplot(gs[0])
    date = datetime.date.today().isoformat()
    show_title(ax['TITLE'], src, os.path.basename(outfile), date, pagenum)
    # info box
    gs = GridSpec(1, 1, left=0.02, right=0.75, top=0.935, bottom=0.85)
    ax['INFO'] = plt.subplot(gs[0])
    show_info(ax['INFO'], src)
    # field box
    gs = GridSpec(1, 1, left=0.75, right=0.98, top=0.98, bottom=0.85)
    ax['FIELD'] = plt.subplot(gs[0])
    show_field(ax['FIELD'], src, white)
    # white and HST images box
    gs = GridSpec(1, 5, left=0.02, right=0.98, top=0.84, bottom=0.64,
                  wspace=0.02)
    ax['WHITE1'] = plt.subplot(gs[0, 0])
    show_white(ax['WHITE1'], src, cat, zscale=True, showmask=True,
               showscale=True)
    ax['WHITE2'] = plt.subplot(gs[0, 1])
    show_white(ax['WHITE2'], src, cat, zscale=True, showmask=False, showz=True,
               showscale=False)
    ax['HST1'] = plt.subplot(gs[0, 2])
    show_hstima(ax['HST1'], src, key=ima2, showscale=True)
    ax['HST2'] = plt.subplot(gs[0, 3])
    show_hstima(ax['HST2'], src, key=ima2, showid=False, showmask=True,
                showscale=False)
    if 'ORIG_MXMAP' in src.images:
        ax['MXMAP'] = plt.subplot(gs[0, 4])
        show_maxmap(ax['MXMAP'], src, zscale=False, showscale=True)

    # all hst images
    if not debug:
        hstkeys = get_hstkeys(src)
        gs = GridSpec(1, len(hstkeys), left=0.05, right=0.98, top=0.64,
                      bottom=0.56)
        for k, key in enumerate(hstkeys):
            ax[key] = plt.subplot(gs[0, k])
            show_image(ax[key], src, key)

    # 3D HST
    if '3DHST_SPEC' in src.images:
        gs = GridSpec(1, 1, left=0.20, right=0.70, top=0.56, bottom=0.52)
        ax['3DHST'] = plt.subplot(gs[0])
        show_3dhst(ax['3DHST'], src)

    # full spectra
    sp2name = None
    if src.REFSPEC == 'MUSE_PSF_SKYSUB':
        if 'MUSE_WHITE_SKYSUB' in src.spectra:
            sp2name = 'MUSE_WHITE_SKYSUB'
        elif 'MUSE_TOT_SKYSUB' in src.spectra:
            sp2name = 'MUSE_TOT_SKYSUB'
    elif src.REFSPEC == 'MUSE_WHITE_SKYSUB':
        if 'MUSE_PSF_SKYSUB' in src.spectra:
            sp2name = 'MUSE_PSF_SKYSUB'
        elif 'MUSE_TOT_SKYSUB' in src.spectra:
            sp2name = 'MUSE_TOT_SKYSUB'

    logger.debug('Displaying Spectra %s %s', src.REFSPEC, sp2name)
    gs = GridSpec(2, 1, left=0.05, right=0.98, top=0.45, bottom=0.25,
                  hspace=0.03)
    ax['FULLSPEC1'] = plt.subplot(gs[0, 0])
    show_fullspec(ax['FULLSPEC1'], src, src.REFSPEC, sp2name=sp2name)
    ax['FULLSPEC2'] = plt.subplot(gs[1, 0], sharex=ax['FULLSPEC1'])
    show_contspec(ax['FULLSPEC2'], src, src.REFSPEC, sp2name=sp2name)
    plt.setp(ax['FULLSPEC1'].get_xticklabels(), visible=False)

    # find wavelengths of ORIGIN spectra if any
    origlbda = [int(key[5:]) for key in src.spectra
                if key[0:5] == 'ORIG_']

    # loop on NB and zoom on spectra
    if 'NB_PAR' not in src.tables or len(src.tables['NB_PAR']) == 0:
        msg = 'No emission lines (table NB_PAR not found or empty)'
        report_error(src.ID, msg, level='DEBUG')
        bottom = 0.25
    else:
        tab = src.tables['NB_PAR']
        top = 0.25
        h = 0.15
        dh = 0.06
        top = top - dh
        tab.sort('LBDA')
        for k, line in enumerate(tab):
            nb, l0, width, margin, fband = line
            logger.debug('Source {} {}/{} {} display'
                         .format(src.ID, k + 1, len(tab), nb))
            if width == 0:
                report_error(src.ID, f'bad NB_PAR parameters for NB {nb}')
                bottom = top
                continue
            bottom = top - h
            if bottom < 0:
                pagenum, top, fig = _newpage(pagenum, fig, src, outfile, date,
                                             pdf_pages, __version__)
                bottom = top - h
            gs = GridSpec(1, 4, left=0.05, right=0.98, top=top, bottom=top - h,
                          wspace=0.25)
            key = 'NB_{}'.format(k)
            ax[key] = plt.subplot(gs[0, 0])
            show_nb(ax[key], src, nb)
            key1 = 'ZOOM_{}'.format(k)
            ax[key1] = plt.subplot(gs[0, 1])
            lbda = show_zoomspec(ax[key1], src, src.REFSPEC, sp2name, l0,
                                 width, margin, fband)
            if lbda is None:
                continue
            else:
                l1, l2 = lbda

            if 'PFIT_REF_SPFIT' not in src.tables:
                report_error(src.ID, 'Missing PFIT_REF_SPFIT table in source')
                break
            if nb == 'NB_LYALPHA':
                key = 'PFIT_{}'.format(k)
                ax[key] = plt.subplot(gs[0, 2], sharey=ax[key1])
                show_pfitspec(ax[key], src, 'SPECFIT_COMPLEX', l1, l2)
                plt.setp(ax[key].get_yticklabels(), visible=False)
            else:
                key = 'PFIT_{}'.format(k)
                ax[key] = plt.subplot(gs[0, 2], sharey=ax[key1])
                show_pfitspec(ax[key], src, 'SPECFIT', l1, l2)
                plt.setp(ax[key].get_yticklabels(), visible=False)
            if nb == 'NB_LYALPHA':
                key = 'LINE_{}'.format(k)
                ax[key] = plt.subplot(gs[0, 3])
                show_pfitline(ax[key], src, 'NEBULAR_COMPLEX_ONLY', l1, l2)
            else:
                key = 'LINE_{}'.format(k)
                ax[key] = plt.subplot(gs[0, 3])
                show_pfitline(ax[key], src, 'NEBULAR_ALL', l1, l2)
            if len(origlbda) > 0:
                for lbda in origlbda:
                    if (lbda <= l1) or (lbda >= l2):
                        continue
                    key = 'LINE_{}'.format(k)
                    if key not in ax:
                        ax[key] = plt.subplot(gs[0, 3])
                    show_origspec(ax[key], src, lbda, l1, l2)
                    break
            top = bottom - dh

    # show errors if any
    if len(err_report) > 0 and src.ID in err_report:
        top = bottom - 0.02
        dt = 0.15
        bottom = top - dt
        if bottom < 0:
            pagenum, top, fig = _newpage(pagenum, fig, src, outfile, date,
                                         pdf_pages, __version__)
            bottom = top - dt
        gs = GridSpec(1, 1, left=0.05, right=0.98, top=top, bottom=bottom)
        ax['ERROR'] = plt.subplot(gs[0])
        show_errors(ax['ERROR'], src)
    # comments and history
    dt = 0.3
    top = bottom - 0.02
    bottom = top - dt
    if bottom < 0:
        pagenum, top, fig = _newpage(pagenum, fig, src, outfile, date,
                                     pdf_pages, __version__)
        bottom = top - dt
    gs = GridSpec(1, 1, left=0.05, right=0.98, top=top, bottom=bottom)
    ax['COM'] = plt.subplot(gs[0])
    show_comments(ax['COM'], src)

    # close figure
    pdf_pages.savefig(fig)
    plt.close(fig)
    pdf_pages.close()


def _newpage(pagenum, fig, src, outfile, date, pdf_pages, version):
    pagenum += 1
    # logger = logging.getLogger(__name__)
    # logger.debug('Source {} New PDF page {}'.format(src.ID, pagenum))
    pdf_pages.savefig(fig)
    plt.close(fig)
    fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    # info box
    gs = GridSpec(1, 1, left=0.02, right=0.75, top=0.95, bottom=0.935)
    ax = plt.subplot(gs[0])
    show_title(ax, src, outfile, date, pagenum, version=version)
    top = 0.90
    return pagenum, top, fig
