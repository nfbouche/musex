# module for UDF source and catalog display
# Taken from muse_analysis/udf/display.py

import astropy.units as u
import lineid_plot
import logging
import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import warnings

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from itertools import cycle

from astropy.convolution import convolve, Box1DKernel
from astropy.io.votable import exceptions
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.units import UnitsWarning
from lineid_plot import plot_line_ids
from matplotlib.patches import Ellipse
from matplotlib.ticker import AutoMinorLocator
from mpdaf.sdetect import get_emlines

warnings.filterwarnings('ignore', category=UnitsWarning)
warnings.filterwarnings('ignore', category=exceptions.VOWarning)

__all__ = ['show_title', 'show_comments', 'show_errors',
           'show_info', 'show_field', 'show_image', 'show_mask', 'show_white',
           'show_3dhst', 'show_maxmap', 'show_hstima', 'show_catalog',
           'show_fullspec', 'show_nb', 'show_zoomspec', 'show_pfitspec',
           'show_pfitline', 'show_origspec']

SPECTRA_TYPES = {0: 'Star', 1: 'Ha emi.', 2: 'OII emi.', 3: 'Abs. gal.',
                 4: 'CIII emi.', 5: 'OIII emi.', 6: 'Lya emi.', 7: 'Other'}

# doublets = [['OII3727', 'OII_3726', 'OII_3729'],
#             ['CIII1909', 'CIII_1907', 'CIII_1909'],
#             ['OIII', 'OIII_4959', 'OIII_5007'],
#             ['MGII', 'MGII_2796', 'MGII_2803'],
#             ['SII', 'SII_6717', 'SII_6731'],
#             ['NII', 'NII_6548', 'NII_6584']]

CONFID_COLORS = ['red', 'orange', 'green', 'blue']

err_report = defaultdict(list)


def report_error(source_id, msg, log=True, level='ERROR'):
    if log:
        logger = logging.getLogger(__name__)
        if level == 'ERROR':
            logger.error(msg)
        elif level == 'WARNING':
            logger.warning(msg)

    err_report[source_id].append(f'{level}: {msg}')


@contextmanager
def no_axis(ax, hide_frame=True):
    yield
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.axis('off')


def add_text_with_fontsize(ax, fontsize=12):
    def _add(x, y, text, **kwargs):
        ax.text(x, y, text, fontsize=fontsize, ha='left', **kwargs)
    return _add


def _check_source_input(kind, key=None):
    def func_wrapper(func):
        @wraps(func)
        def returned_wrapper(*args, **kwargs):
            # args must be ax, src, key
            src = args[1]
            if key is None:
                try:
                    tag = args[2]
                except IndexError:
                    tag = kwargs.get('key')
            else:
                tag = key

            if kind == 'image':
                attr = src.images
            elif kind == 'spectrum':
                attr = src.spectra
            elif kind == 'table':
                attr = src.tables
            if tag not in attr:
                report_error(src.ID, '{} {} not found in source'
                             .format(kind.capitalize(), tag))
                return
            return func(*args, **kwargs)
        return returned_wrapper
    return func_wrapper


def show_title(ax, src, outfile, date, numpage, srcfont=12, textfont=9,
               version=None):
    with no_axis(ax):
        text = 'ID: {:04d} version: {}'.format(src.ID, src.SRC_V)
        bcol = CONFID_COLORS[src.header.get('CONFID', 0)]
        ax.text(0.02, 0.8, text, fontsize=srcfont,
                bbox={'facecolor': bcol, 'alpha': 0.5, 'pad': 3}, ha='left')
        text = 'PDF: {} Date: {} Page: {}'.format(outfile, date, numpage)
        if version is not None:
            text += ' UDF_show_source.py v' + version
        ax.text(0.35, 0.8, text, fontsize=textfont, ha='left')


def _show_text_block(ax, header_text, lines, textfont=8, h=0.02, v=0.95,
                     facecolor='lightgray'):
    dv = 0.05
    bbox = {'facecolor': facecolor, 'alpha': 0.3, 'pad': 2}
    with no_axis(ax):
        ax.text(h, v, header_text, fontsize=textfont + 1, ha='left', bbox=bbox)
        for text in lines:
            v = v - dv
            if v < 0.02:
                break
            ax.text(h, v, text, fontsize=textfont, ha='left')
    return v


def show_comments(ax, src, textfont=8):
    h, v = (0.02, 0.95)
    dv = 0.05
    if 'COMMENT' in src.header:
        _show_text_block(ax, 'Comments', src.header['COMMENT'],
                         textfont=textfont, h=h, v=v)
    if 'HISTORY' in src.header:
        v = v - dv
        _show_text_block(ax, 'History', src.header['HISTORY'],
                         textfont=textfont, h=h, v=v)


def show_errors(ax, src, textfont=8):
    _show_text_block(ax, 'Processing Errors', err_report.get(src.ID, []),
                     textfont=textfont, facecolor='red')


def show_info(ax, src, textfont=9):
    logger = logging.getLogger(__name__)

    add_text = add_text_with_fontsize(ax, fontsize=textfont)
    text = f'Cube: {src.CUBE} v {src.CUBE_V} Finder: {src.FROM} v {src.FROM_V}'
    add_text(0.02, 0.9, text)

    if hasattr(src, 'CONFID'):
        bcol = 'green'
        text = f"MUSE Class Type: {SPECTRA_TYPES.get(src.TYPE, 'Unknown')}"

        if src.z and 'MUSE' in src.z['Z_DESC']:
            zmuse = src.z[src.z['Z_DESC'] == 'MUSE']['Z'][0]
            text += ' Redshift {:.3f}'.format(zmuse)
        else:
            text += ' Redshift ?'
            bcol = 'red'

        text += (
            f" Confid: {src.CONFID}"
            f" Defect: {src.header.get('DEFECT', '?')}"
            f" Blend: {src.header.get('BLEND', '?')}"
            f" Revisit: {src.header.get('REVISIT', '?')}"
        )

        if src.header.get('DEFECT') != 0:
            bcol = 'magenta'
        if src.header.get('BLEND') != 0:
            bcol = 'orange'
        if src.header.get('REVISIT') != 0:
            bcol = 'yellow'

        add_text(0.02, 0.7, text,
                 bbox={'facecolor': bcol, 'alpha': 0.3, 'pad': 2})
    else:
        logger.debug('No MUSE Z info')
        add_text(0.02, 0.7, 'No MUSE z')

    if hasattr(src, 'z') and (src.z is not None):
        t = [f'{ztype} {z:.3f} '
             for ztype, z in zip(src.z['Z_DESC'], src.z['Z'])
             if ztype != 'MUSE']
        add_text(0.02, 0.5, 'Published redshifts: ' + ''.join(t))
    else:
        logger.debug('No Published z')
        add_text(0.02, 0.5, 'Published redshifts: None')

    srcmag = getattr(src, 'mag')
    if srcmag is not None:
        maxlen = 7
        t = []
        for mag, err, band in zip(srcmag['MAG'], srcmag['MAG_ERR'],
                                  srcmag['BAND']):
            if band[0:3] != 'HST' or mag is np.ma.masked:
                continue
            t.append('{}{}{:.1f} '
                     .format(band[5:], '>' if err < 0 else ':', mag))

        add_text(0.02, 0.3, 'HST Mag: ' + ''.join(t[:maxlen]))

        t = ['{}:{:.1f} '.format(band[6:], mag)
             for mag, band in zip(srcmag['MAG'], srcmag['BAND'])
             if band[0:4] == 'MUSE']
        add_text(0.02, 0.1, 'MUSE Mag: ' + ''.join(t[:maxlen]))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis('off')


def show_field(ax, src, white):
    white.plot(ax=ax, vmin=0, vmax=10, scale='arcsinh', cmap='gray_r',
               show_xlabel=False, show_ylabel=False)
    y, x = white.wcs.sky2pix((src.DEC, src.RA))[0]
    with no_axis(ax, hide_frame=False):
        ax.scatter(x, y, s=50, marker='s', c='green', alpha=0.5)


@_check_source_input('image')
def show_image(ax, src, key, cuts=(None, None), zscale=False, scale='arcsinh',
               showcenter=True, cmap='gray_r', fwhm=0):
    ima = src.images[key]
    title = key
    if fwhm > 0:
        logger = logging.getLogger(__name__)
        logger.debug('Displaying Image %s after gaussian filter of %s '
                     'arcsec FWHM', key, fwhm)
        ima = ima.fftconvolve_gauss(fwhm=fwhm)
        title = '{} [{:.1f}]'.format(key, fwhm)

    with no_axis(ax, hide_frame=False):
        ima.plot(ax=ax, vmin=cuts[0], vmax=cuts[1], zscale=zscale, scale=scale,
                 cmap=cmap, show_xlabel=False, show_ylabel=False)
        ax.set_title(title, fontsize=8)
        if showcenter:
            p, q = ima.wcs.sky2pix((src.DEC, src.RA))[0]
            ax.axvline(q, color='r', alpha=0.2)
            ax.axhline(p, color='r', alpha=0.2)


@_check_source_input('image')
def show_mask(ax, src, key, col='r', levels=[0], alpha=0.4, surface=False):
    im = src.images[key]._data.astype(float)
    if surface:
        ax.contourf(im, levels=levels, origin='lower', colors=col, alpha=alpha,
                    extent=[-0.5, im.shape[0] - 0.5, -0.5, im.shape[1] - 0.5])
    else:
        ax.contour(im, levels=levels, origin='lower', colors=col, alpha=alpha)


def show_white(ax, src, cat=None, showmask=False, showid=True, showz=False,
               idcol=None, zcol='z', showscale=True, cuts=(None, None),
               zscale=False, scale='arcsinh', showcenter=True, cmap='gray_r',
               fwhm=0):
    show_image(ax, src, 'MUSE_WHITE', cuts=cuts, zscale=zscale, scale=scale,
               showcenter=showcenter, cmap=cmap, fwhm=fwhm)
    if showscale:
        show_scale(ax, src.images['MUSE_WHITE'].wcs)
    if showmask:
        show_mask(ax, src, 'MASK_OBJ', col='magenta', surface=True,
                  levels=[0.99, 1.01], alpha=0.2)
        show_mask(ax, src, 'MASK_SKY', col='green', surface=True,
                  levels=[0.99, 1.01], alpha=0.2)
    if cat is not None:
        if not hasattr(cat, 'name'):
            cat.name = 'MUSE catalog'
        if showz:
            cat.name = 'MUSE redshifts'
        if 'CONFID' not in cat.colnames:
            cat['COLOR'] = 'k'
        else:
            cat['COLOR'] = 'yellow'
            cat['COLOR'][cat['CONFID'] == 1] = 'blue'
            cat['COLOR'][cat['CONFID'] == 2] = 'cyan'
            cat['COLOR'][cat['CONFID'] == 3] = 'lime'

        if idcol is None:
            idcol = cat.meta.get('idname', 'id')
        ksrc = (cat[idcol] == src.ID)
        cat['COLOR'][ksrc] = 'r'
        if showz and 'hasz' in cat.colnames:
            mask = cat['hasz']
            cat[zcol][mask] = ['%.3f' % z for z in cat[zcol][mask]]
            show_catalog(ax, src, 'MUSE_WHITE', cat, iden=zcol, symb=0.2,
                         fontsize=8)
        else:
            show_catalog(ax, src, 'MUSE_WHITE', cat, iden=idcol, symb=0.2,
                         fontsize=8, showid=showid)


@_check_source_input('image', key='3DHST_SPEC')
def show_3dhst(ax, src):
    logger = logging.getLogger(__name__)
    logger.debug('Display of 3DHST data')
    show_image(ax, src, '3DHST_SPEC', scale='linear', showcenter=False,
               cmap='jet')
    ax.set_title('')
    ax2 = ax.twiny()
    xlim = (src.images['3DHST_SPEC'].get_start()[1] * 1.e6,
            src.images['3DHST_SPEC'].get_end()[1] * 1.e6)
    ax2.set_xlim(xlim)
    ax2.xaxis.tick_bottom()
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.text(-0.05, 0.5, '3DHST', {'color': 'k', 'fontsize': 8},
             transform=ax2.transAxes, ha='center', va='center', rotation=90)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    if '3DHST_LINES' in src.tables:
        logger.debug('Display of {} lines from 3DHST line table'
                     .format(len(src.tables['3DHST_LINES'])))
        for line in src.tables['3DHST_LINES']:
            xpos = line['LBDA_OBS'] * 1.e-4  # wavelength in microns
            iden = line['LINE']
            y1, y2 = ax2.get_ylim()
            ax2.axvline(xpos, ymin=0.8, ymax=0.9, color='black', alpha=1.0)
            ax2.text(xpos, y2 - 2, iden,
                     {'color': 'k', 'fontsize': 8},
                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 1},
                     ha='center', va='bottom', rotation=90)
    else:
        logger.debug('No 3DHST line table found in source')


@_check_source_input('image', key='ORIG_MXMAP')
def show_maxmap(ax, src, zscale=False, showcat=True, showid=False,
                showscale=True):
    show_image(ax, src, 'ORIG_MXMAP', zscale=zscale)
    if showscale:
        show_scale(ax, src.images['ORIG_MXMAP'].wcs)
    if hasattr(src, 'ORIG_RA') and hasattr(src, 'ORIG_DEC'):
        cen = src.images['ORIG_MXMAP'].wcs.sky2pix((src.ORIG_DEC, src.ORIG_RA),
                                                   unit=u.deg)[0]
        _plot_ellipse(ax, cen, 2.0, alpha=0.5, col='cyan', fill=True)
        if hasattr(src, 'ORIG_ID'):
            ax.text(cen[1] + 2, cen[0] + 2, str(src.ORIG_ID), ha='center',
                    color='cyan', fontsize=8)
    if showcat:
        cat = src.refcat
        if cat is None:
            return
        # cat.name = 'Rafelski Catalog'
        cat['COLOR'] = 'w' if zscale else 'b'
        if hasattr(src, 'RAF_ID'):
            rafids = [int(iden) for iden in src.RAF_ID.split(',')]
            if len(rafids) > 0:
                idcol = cat.meta.get('idname', 'ID')
                cat['COLOR'][np.in1d(cat[idcol], rafids)] = 'r'

        # logger = logging.getLogger(__name__)
        # logger.debug('%d sources to display from catalog', len(cat))
        show_catalog(ax, src, 'ORIG_MXMAP', cat, symb=0.2, showid=showid)


@_check_source_input('image')
def show_hstima(ax, src, key='HST_F775W', zscale=False, showcat=True,
                showid=True, showmask=False, showscale=True):
    show_image(ax, src, key, zscale=zscale)
    if showscale:
        show_scale(ax, src.images[key].wcs)

    cat = src.refcat
    if cat is None:
        return
    # cat.name = 'Rafelski Catalog'
    idcol = cat.meta.get('idname', 'ID')

    if showcat:
        cat['COLOR'] = 'w' if zscale else 'b'
        if hasattr(src, 'RAF_ID'):
            rafids = [int(iden) for iden in src.RAF_ID.split(',')]
            if len(rafids) > 0:
                cat['COLOR'][np.in1d(cat[idcol], rafids)] = 'r'
        # logger = logging.getLogger(__name__)
        # logger.debug('%d sources to display from catalog', len(cat))
        show_catalog(ax, src, key, cat, symb=0.2, showid=showid)

    if showmask:
        cycol = cycle('bgrcmy')
        segm = src.images[src.SEGMAP].align_with_image(src.images[key])
        data = segm.data.filled().astype(float)
        extent = [-0.5, data.shape[0] - 0.5, -0.5, data.shape[1] - 0.5]
        for iden in cat[idcol]:
            ax.contourf(data, levels=[iden - 0.1, iden + 0.1], origin='lower',
                        colors=next(cycol), alpha=0.5, extent=extent)


def show_scale(ax, wcs, right=0.95, y=0.96, linewidth=1, color='b'):
    scale = 1.0
    dx = scale / (wcs.naxis1 * 3600 * wcs.get_step()[0])
    ax.axhline(y * wcs.naxis1, right - dx, right, linewidth=linewidth,
               color=color)
    ax.text(right - 0.5 * dx, y + 0.06, '1"', horizontalalignment='center',
            fontsize=8, transform=ax.transAxes, color=color)


@_check_source_input('image')
def show_catalog(ax, src, key, cat, showid=True, iden=None, ra=None, dec=None,
                 col='COLOR', symb=0.5, fontsize=8, alpha=0.5, legend=True):
    if iden is None:
        iden = cat.meta.get('idname', 'ID')
    if ra is None:
        ra = cat.meta.get('raname', 'RA')
    if dec is None:
        dec = cat.meta.get('decname', 'DEC')

    wcs = src.images[key].wcs
    arr = np.vstack([cat[dec].data, cat[ra].data]).T
    arr = wcs.sky2pix(arr, unit=u.deg)
    size = symb / wcs.get_step(unit=u.arcsec)[0]
    # margin = 2 * size

    if showid:
        texts = []
        for src, cen in zip(cat, arr):
            yy, xx = cen
            if (xx < 0) or (yy < 0) or (xx > wcs.naxis1) or (yy > wcs.naxis2):
                continue
            if not np.ma.is_masked(src[iden]):
                texts.append((ax.text(xx, yy, src[iden], ha='center',
                                      color=src[col], fontsize=fontsize),
                              cen[1], cen[0]))
            _plot_ellipse(ax, cen, size, alpha=alpha, col=src[col])
        if len(texts) > 0:
            from adjustText import adjust_text
            text, x, y = zip(*texts)
            adjust_text(text, x=x, y=y, ax=ax, only_move={text: 'xy'})
    else:
        for src, cen in zip(cat, arr):
            yy, xx = cen
            if (xx < 0) or (yy < 0) or (xx > wcs.naxis1) or (yy > wcs.naxis2):
                continue
            _plot_ellipse(ax, cen, size, alpha=alpha, col=src[col])
    if legend and hasattr(cat, 'name'):
        ax.text(0.5, -0.07, cat.name, ha='center', transform=ax.transAxes,
                fontsize=8)


@_check_source_input('spectrum')
def show_fullspec(ax, src, sp1name, sp2name=None, ymin=-20, ymax=None,
                  wfilter=5, fontsize=7, showvar=True, varlim=None,
                  legend=True, showlines=True, expectedlines=True):
    logger = logging.getLogger(__name__)
    if sp2name is not None and sp2name not in src.spectra:
        report_error(src.ID, 'Spectra {} not found in source'.format(sp2name))
        return
    # plot (filtered) refspectra in blue
    sp = src.spectra[sp1name]
    if wfilter > 0:
        sp1 = sp.copy()
        sp1._data = convolve(sp._data, Box1DKernel(wfilter))
    else:
        sp1 = sp
    sp1.plot(ax=ax, alpha=0.8, color='b')
    # plot (filtered) alternate spectra in black
    if sp2name is not None:
        sp = src.spectra[sp2name]
        if wfilter > 0:
            sp2 = sp.copy()
            sp2._data = convolve(sp._data, Box1DKernel(wfilter))
        else:
            sp2 = sp
        sp2.plot(ax=ax, alpha=0.2, color='k')
    l1, l2, y1, y2 = ax.axis()
    ax.axhline(0, color='k', alpha=0.5)
    if showvar:
        ax2 = ax.twinx()
        err = np.sqrt(sp._var)
        ax2.plot(sp.wave.coord(), err, alpha=0.2, color='m')
        if varlim is not None:
            ax2.set_ylim(varlim)
        ax2.invert_yaxis()
        ax2.axis('off')
    ax.set_xlim(4750, 9350)
    if ymax is None:
        ax.set_ylim(ymin, y2)
    else:
        ax.set_ylim(ymin, ymax)
    if showlines and src.z is not None:
        if src.lines is None or len(src.lines) == 0:
            logger.debug('No identified emission lines in source')
            # add lines
            z = src.get_zmuse()
            if z is None:
                return
            olines = get_emlines(z=z, vac=False, lbrange=[4750, 9350])
            if len(olines) == 0:
                return
            lines = Table(data=[olines['id'], olines['c']],
                          names=['LINE', 'LBDA_OBS'])
            lines['COLOR'] = 'k'
            lines['FONTSIZE'] = fontsize
        else:
            lines = src.lines['LINE', 'LBDA_OBS']
            lines['COLOR'] = 'b'
            lines['FONTSIZE'] = fontsize
            nblines = src.nb_to_lines()
            for line in lines:
                if line['LINE'] in nblines:
                    line['COLOR'] = 'r'
            # additional lines
            if expectedlines:
                z = src.get_zmuse()
                olines = get_emlines(z=z, vac=False, lbrange=[4750, 9350])
                if len(olines) > 0:
                    for line in olines:
                        # skip the abs line if already in the emission line
                        # list
                        if line['id'] in lines['LINE']:
                            continue
                        lines.add_row([line['id'], line['c'], 'k', fontsize])
        logger.debug('%d lines displayed on the full spectrum', len(lines))
        lines.sort('LBDA_OBS')
        plot_line_ids(sp1.wave.coord(), sp1._data, lines['LBDA_OBS'],
                      lines['LINE'], lines['FONTSIZE'], arrow_tip=0.9 * y2,
                      box_axes_space=0.04, ax=ax)
        color_text_boxes(ax, lines['LINE'], lines['COLOR'])
        color_lines(ax, lines['LINE'], lines['COLOR'])

    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if legend:
        txt = '{} [{}]'.format(src.REFSPEC, wfilter)
        ax.text(0.02, 0.8, txt, ha='left', transform=ax.transAxes,
                fontsize=7, alpha=0.2, color='b')
        if sp2name is not None:
            txt = '{} [{}]'.format(sp2name, wfilter)
            ax.text(0.02, 0.7, txt, ha='left', transform=ax.transAxes,
                    fontsize=7, alpha=0.2, color='k')
    yminloc = AutoMinorLocator(5)
    ax.xaxis.set_minor_locator(yminloc)


@_check_source_input('spectrum')
def show_contspec(ax, src, sp1name, sp2name=None, ymin=-20, dymin=30, nsig=5,
                  wfilter=5, fontsize=7, showvar=False):
    if sp2name is not None and sp2name not in src.spectra:
        report_error(src.ID, f'Cannot read spectra {sp2name} in source')
        return
    # plot (filtered) refspectra in blue
    sp = src.spectra[sp1name]
    if wfilter > 0:
        sp1 = sp.copy()
        sp1._data = convolve(sp._data, Box1DKernel(wfilter))
    else:
        sp1 = sp
    sp1.plot(ax=ax, alpha=0.8, color='b')
    # plot (filtered) alternate spectra in blue
    if sp2name is not None:
        sp = src.spectra[sp2name]
        if wfilter > 0:
            sp2 = sp.copy()
            sp2._data = convolve(sp._data, Box1DKernel(wfilter))
        else:
            sp2 = sp
        sp2.plot(ax=ax, alpha=0.2, color='k')
    # estimate y1,y2 around continuum
    sp_mean, sp_med, sp_std = sigma_clipped_stats(sp1._data, sigma=2.0)
    y1 = np.round(max(sp_mean - nsig * sp_std, ymin), 0)
    y2 = np.round(max(sp_mean + nsig * sp_std, ymin + dymin), 0)

    logger = logging.getLogger(__name__)
    logger.debug('contspec mean=%.1f med=%.1f std=%.1f yaxis=(%.1f,%.1f)',
                 sp_mean, sp_med, sp_std, y1, y2)

    if showvar:
        ax2 = ax.twinx()
        err = np.sqrt(sp._var)
        ax2.plot(sp.wave.coord(), err, alpha=0.2, color='m')
        ax2.invert_yaxis()
        ax2.axis('off')

    # ax.set_xlim(4750, 9350)
    ax.set_ylim(y1, y2)
    yminloc = AutoMinorLocator(5)
    ax.xaxis.set_minor_locator(yminloc)

    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')


@_check_source_input('image')
def show_nb(ax, src, nb, zscale=False, scale='linear', fwhm=0.6, showcat=True):
    show_image(ax, src, nb, cmap='coolwarm', scale=scale, zscale=zscale,
               fwhm=fwhm)
    if showcat:
        cat = src.refcat
        if cat is None:
            return
        # cat.name = 'Rafelski Catalog'
        cat['COLOR'] = 'w'
        if hasattr(src, 'RAF_ID'):
            rafids = [int(iden) for iden in src.RAF_ID.split(',')]
            if len(rafids) > 0:
                cat['COLOR'][np.in1d(cat['ID'], rafids)] = 'k'

        # logger = logging.getLogger(__name__)
        # logger.debug('%d sources to display from catalog', len(cat))
        show_catalog(ax, src, nb, cat, symb=0.2, showid=False, legend=False)

    lines = src.nb_to_lines([nb])
    snr = []
    flux = []
    if 'SNR_REF' in src.lines and 'FLUX_REF' in src.lines:
        for line in lines:
            if line not in src.lines['LINE']:
                continue
            row = src.lines.loc[line]
            snr.append(row['SNR_REF'])
            flux.append(row['FLUX_REF'])
        if len(line) == 2:
            flux = flux[0] + flux[1]
            err = np.sqrt((flux[0] / snr[0])**2 + (flux[1] / snr[1])**2)
            snr = flux / err
        else:
            flux = flux[0]
            snr = snr[0]
        text = 'F: {:.1f} SNR: {:.1f}'.format(flux, snr)
        ax.text(0.5, -0.09, text, ha='center', transform=ax.transAxes,
                fontsize=8)


def show_zoomspec(ax, src, sp1name, sp2name=None, l0=None, width=50, margin=0,
                  fband=0, waverange=None, zero=True, showlines=True,
                  name=None, showvar=True, varlim=None, ymin=None, ymax=None):
    if name is not None:
        if name not in src.lines['LINE']:
            report_error(src.ID, '%s not found in src.lines'.format(name))
            return
        l0 = src.lines.loc[name]['LBDA_OBS']
    if waverange is not None:
        l1, l2 = waverange
    else:
        if l0 is None:
            report_error(src.ID,
                         'show_zoomspec with waverange=None and l0=None')
            return
        l1 = l0 - width / 2 - margin - fband - 10
        l2 = l0 + width / 2 + margin + fband + 10
    if sp1name not in src.spectra:
        report_error(src.ID, f'Cannot find spectrum {sp1name} in source')
        return
    sp0 = src.spectra[sp1name]
    sp = sp0.subspec(lmin=l1, lmax=l2)
    sp.plot(ax=ax, alpha=0.8, color='b')
    ax.set_xlim(l1, l2)
    x1, x2, y1, y2 = ax.axis()
    if zero:
        ax.axhline(0, color='k', alpha=0.5)
    if sp2name is not None:
        sp2 = src.spectra[sp2name]
        sp2 = sp2.subspec(lmin=l1, lmax=l2)
        sp2.plot(ax=ax, alpha=0.2, color='k')
    if showvar:
        ax2 = ax.twinx()
        err = np.sqrt(sp._var)
        ax2.plot(sp.wave.coord(), err, alpha=0.2, color='m')
        if varlim is None:
            ferr = np.sqrt(sp0._var)
            ax2.set_ylim(ferr.min(), ferr.max())
        else:
            ax2.set_ylim(varlim)
        ax2.invert_yaxis()
        ax2.axis('off')
    if ymax is None:
        ymax = y2
    if ymin is None:
        ymin = y1
    ax.set_ylim(ymin, ymax)
    if hasattr(src, 'lines') and showlines:
        lines = src.lines
        lines = lines[(lines['LBDA_OBS'] > l1) & (lines['LBDA_OBS'] < l2)]
        if len(lines) == 0:
            report_error(src.ID, 'Line table is empty for the given '
                         'wavelength window {}-{}'.format(l1, l2))
            return
        lines['COLOR'] = 'r'
        lines['FONTSIZE'] = 7
        plot_line_ids(sp.wave.coord(), sp._data, lines['LBDA_OBS'],
                      lines['LINE'], lines['FONTSIZE'], box_axes_space=0.02,
                      ax=ax)
        color_text_boxes(ax, lines['LINE'], lines['COLOR'])
        color_lines(ax, lines['LINE'], lines['COLOR'])

    if (margin > 0) and (fband > 0):
        ax.axvspan(l0 - width / 2, l0 + width / 2, color='g', alpha=0.2)
        ax.axvspan(l0 - width / 2 - margin - fband, l0 - width / 2 - margin,
                   color='b', alpha=0.2)
        ax.axvspan(l0 + width / 2 + margin, l0 + width / 2 + margin + fband,
                   color='b', alpha=0.2)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    yloc = plt.MaxNLocator(4)
    yminloc = plt.MultipleLocator(5)
    ax.xaxis.set_major_locator(yloc)
    ax.xaxis.set_minor_locator(yminloc)
    ax.set_xlabel('')
    ax.set_ylabel('')
    return l1, l2


def show_pfitspec(ax, src, key, l1, l2):
    logger = logging.getLogger(__name__)
    if key not in src.tables['PFIT_REF_SPFIT']:
        report_error(src.ID, 'Cannot read table {} in source'.format(key))
        return
    tab = src.tables['PFIT_REF_SPFIT']
    lbda = tab['WAVELENGTH']
    stab = tab[(lbda >= l1) & (lbda <= l2)]
    ax.plot(stab['WAVELENGTH'], stab['FLUX'], ls='steps', alpha=0.5, color='k')
    logger.debug('Display {} spectrum from table'.format(key))
    ax.plot(stab['WAVELENGTH'], stab[key], ls='steps', alpha=1, color='r')
    ax.set_xlim(l1, l2)
    ax.axhline(0, color='k', alpha=0.5)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.text(0.1, 0.9, key, ha='left', transform=ax.transAxes,
            fontsize=7, alpha=0.2, color='r')
    yloc = plt.MaxNLocator(4)
    yminloc = plt.MultipleLocator(5)
    ax.xaxis.set_major_locator(yloc)
    ax.xaxis.set_minor_locator(yminloc)


def show_pfitline(ax, src, key, l1, l2):
    logger = logging.getLogger(__name__)
    if key not in src.tables['PFIT_REF_SPFIT']:
        report_error(src.ID, 'Cannot find column {} in table PFIT_REF_SPFIT'
                     .format(key))
        return
    tab = src.tables['PFIT_REF_SPFIT']
    lbda = tab['WAVELENGTH']
    stab = tab[(lbda >= l1) & (lbda <= l2)]
    logger.debug('Display {} spectrum from table'.format(key))
    ax.plot(stab['WAVELENGTH'], stab[key], ls='steps', alpha=1, color='r')
    ax.set_xlim(l1, l2)
    ax.axhline(0, color='k', alpha=0.5)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.text(0.1, 0.9, key, ha='left', transform=ax.transAxes,
            fontsize=7, alpha=0.2, color='r')
    yloc = plt.MaxNLocator(4)
    yminloc = plt.MultipleLocator(5)
    ax.xaxis.set_major_locator(yloc)
    ax.xaxis.set_minor_locator(yminloc)


def show_origspec(ax, src, l0, l1, l2):
    logger = logging.getLogger(__name__)
    spname = 'ORIG_{}'.format(l0)
    if spname not in src.spectra:
        report_error(src.ID, 'Cannot read spectra {} in source'.format(spname))
        return
    logger.debug('Display {} spectrum'.format(spname))
    sp = src.spectra[spname]
    sp.plot(ax=ax, alpha=1, color='b')
    ax.set_xlim(l1, l2)
    ax.axhline(0, color='k', alpha=0.5)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.text(0.1, 0.80, spname, ha='left', transform=ax.transAxes,
            fontsize=7, alpha=0.2, color='b')
    yloc = plt.MaxNLocator(4)
    yminloc = plt.MultipleLocator(5)
    ax.xaxis.set_major_locator(yloc)
    ax.xaxis.set_minor_locator(yminloc)


def _plot_ellipse(ax, cen, size, alpha=0.5, col='k', fill=False):
    ell = Ellipse((cen[1], cen[0]), size, size, 0, fill=fill, alpha=alpha)
    ax.add_artist(ell)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(alpha)
    ell.set_edgecolor(col)
    ell.set_facecolor(col)


# Taken from muse_analysis/lineid_utils.py
# 2015/10/12 H. Inami
#    Taken from http://gist.github.com/phn/2326396


def color_text_boxes(ax, labels, colors, color_arrow=True):
    assert len(labels) == len(colors), \
        "Equal no. of colors and lables must be given"
    boxes = ax.findobj(mpl.text.Annotation)
    box_labels = lineid_plot.unique_labels(labels)
    for box in boxes:
        label = box.get_label()
        try:
            loc = box_labels.index(label)
        except ValueError:
            continue  # No changes for this box
        box.set_color(colors[loc])
        if color_arrow:
            box.arrow_patch.set_color(colors[loc])

    ax.figure.canvas.draw()


def color_lines(ax, labels, colors):
    assert len(labels) == len(colors), \
        "Equal no. of colors and lables must be given"
    lines = ax.findobj(mpl.lines.Line2D)
    line_labels = [i + "_line" for i in lineid_plot.unique_labels(labels)]
    for line in lines:
        label = line.get_label()
        try:
            loc = line_labels.index(label)
        except ValueError:
            continue  # No changes for this line
        line.set_color(colors[loc])

    ax.figure.canvas.draw()
