import astropy.units as u
import logging
import numpy as np


def skymask_from_hst(coord, hstsegmap, refimage, size=5.0, fsf=0.6,
                     thres=(0.01, 0.1)):
    """
    coord: (ra,dec)
    hstsegmap: hst segmentation map
    fsf: fwhm of fsf in arcsec
    thres: threshold value for masking
    """
    logger = logging.getLogger(__name__)
    logger.debug('Sky mask computation at Coord %s FSF %.2f thres %s',
                 coord, fsf, thres)
    invcoord = (coord[1], coord[0])  # MPDAF use (dec,ra)
    logger.debug('Extract Refimage %.1f', size)
    unit_size = u.arcsec
    minsize = size
    subref = refimage.subimage(invcoord, size, minsize=minsize,
                               unit_center=u.deg, unit_size=unit_size)
    shape = subref.shape
    step = subref.wcs.get_step(unit=u.arcsec)
    start = subref.wcs.get_start(unit=u.deg)
    pa_ref = subref.get_rot()
    pa = hstsegmap.get_rot()
    # resample, rotate hstsegmap and select all sources
    if np.abs(pa_ref - pa) > 1.e-3:
        logger.debug('Rotate %.1f and Extract SegMap %.1f', pa_ref - pa, size)
        subima = hstsegmap.subimage(invcoord, size * 2,
                                    minsize=minsize, unit_center=u.deg,
                                    unit_size=unit_size)
        mask = subima.clone(data_init=np.zeros)
        mask.data[subima.data.data != 0] = 1
        mask = mask.rotate(pa_ref - pa, order=0)
        mask = mask.subimage(invcoord, size * 1.5,
                             minsize=minsize, unit_center=u.deg,
                             unit_size=unit_size)
    else:
        logger.debug('Extract SegMap Size %.1f', size)
        subima = hstsegmap.subimage(invcoord, size * 1.5,
                                    minsize=minsize, unit_center=u.deg,
                                    unit_size=unit_size)
        mask = subima.clone(data_init=np.zeros)
        mask.data[subima.data.data != 0] = 1
    # convolve with FSF
    logger.debug('Convolve with Gaussian FSF %.1f', fsf)
    conv = mask.fftconvolve_gauss(fwhm=(fsf, fsf))
    # mask to 1/0 using first threshold value
    logger.debug('Mask to 1/0 using first threshold value %.3f', thres[0])
    tconv = conv.clone(data_init=np.zeros)
    peakobj = np.max(conv.data.data)
    tconv.data[conv.data / peakobj >= thres[0]] = 1
    # rebin
    logger.debug('Resample to step %.2f', step[0])
    bconv = tconv.resample(shape, start, step, order=1,
                           unit_start=u.deg, unit_step=u.arcsec)
    # mask to 1/0 using second threshold value
    logger.debug('Mask to 1/0 using second threshold value %.3f', thres[1])
    seg_sky = bconv.clone(data_init=np.zeros)
    peakobj = np.max(bconv.data.data)
    seg_sky.data[bconv.data / peakobj >= thres[1]] = 1

    return seg_sky


def objmask_from_hst(coord, idlist, hstsegmap, refimage, size=5.0, fsf=0.6,
                     thres=(0.01, 0.1)):
    """
    coord: (ra,dec)
    hstsegmap: hst segmentation map
    fsf: fwhm of fsf in arcsec
    thres: threshold value for masking
    """
    logger = logging.getLogger(__name__)
    if type(idlist) is not list:
        idlist = [idlist]
    logger.debug('Object mask computation at Coord %s HST ID %s FSF %.2f '
                 'thres %s', coord, idlist, fsf, thres)
    invcoord = (coord[1], coord[0])  # MPDAF use (dec,ra)
    logger.debug('Extract Refimage %.1f', size)
    unit_size = u.arcsec
    minsize = size
    subref = refimage.subimage(invcoord, size, minsize=minsize,
                               unit_center=u.deg, unit_size=unit_size)
    shape = subref.shape
    step = subref.wcs.get_step(unit=u.arcsec)
    start = subref.wcs.get_start(unit=u.deg)
    pa_ref = subref.get_rot()
    pa = hstsegmap.get_rot()
    # resample, rotate hstsegmap and select the ID segment and mask all
    # others
    if np.abs(pa_ref - pa) > 1.e-3:
        logger.debug('Rotate %.1f and Extract SegMap %.1f', pa_ref - pa, size)
        subima = hstsegmap.subimage(invcoord, size * 2,
                                    minsize=minsize, unit_center=u.deg,
                                    unit_size=unit_size)
        mask = subima.clone(data_init=np.zeros)
        logger.debug('Select Object(s) %s in SegMap', idlist)
        for iden in idlist:
            mask.data[subima.data.data == iden] = 1
        if mask.data.max() == 0:
            logger.error('HST ID not found in segmentation map within source '
                         'location')
            return None
        mask = mask.rotate(pa_ref - pa, order=0)
        mask = mask.subimage(invcoord, size * 1.5,
                             minsize=minsize, unit_center=u.deg,
                             unit_size=unit_size)
    else:
        logger.debug('Extract SegMap Size %.1f', size)
        subima = hstsegmap.subimage(invcoord, size * 1.5,
                                    minsize=minsize, unit_center=u.deg,
                                    unit_size=unit_size)
        mask = subima.clone(data_init=np.zeros)
        logger.debug('Select Object(s) %s in SegMap', idlist)
        for iden in idlist:
            mask.data[subima.data.data == iden] = 1
        if mask.data.max() == 0:
            logger.error('HST ID not found in segmentation map within source '
                         'location')
            return None
    # convolve with FSF
    logger.debug('Convolve with Gaussian FSF %.1f', fsf)
    conv = mask.fftconvolve_gauss(fwhm=(fsf, fsf))
    # mask to 1/0 using first threshold value
    logger.debug('Mask to 1/0 using first threshold value %.3f', thres[0])
    tconv = conv.clone(data_init=np.zeros)
    peakobj = np.max(conv.data.data)
    tconv.data[conv.data / peakobj >= thres[0]] = 1
    # rebin
    logger.debug('Resample to step %.2f', step[0])
    bconv = tconv.resample(shape, start, step, order=1, unit_start=u.deg,
                           unit_step=u.arcsec)
    # mask to 1/0 using second threshold value
    logger.debug('Mask to 1/0 using second threshold value %.3f', thres[1])
    seg_obj = bconv.clone(data_init=np.zeros)
    peakobj = np.max(seg_obj.data.data)
    seg_obj.data[bconv.data / peakobj >= thres[1]] = 1

    return seg_obj
