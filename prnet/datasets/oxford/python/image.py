################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
###############################################################################

import re
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
import numpy as np

BAYER_STEREO = 'gbrg'
BAYER_MONO = 'rggb'


def load_image(image_path, model=None, debayer=True):
    """Loads and rectifies an image from file.

    Args:
        image_path (str): path to an image from the dataset.
        model (camera_model.CameraModel): if supplied, model will be used to undistort image.

    Returns:
        numpy.ndarray: demosaiced and optionally undistorted image

    """
    if model:
        camera = model.camera
    else:
        camera = re.search('(stereo|mono_(left|right|rear))', image_path).group(0)

    if camera == 'stereo':
        pattern = BAYER_STEREO
    else:
        pattern = BAYER_MONO

    img = Image.open(image_path)
    if debayer:
        img = demosaic(img, pattern)
    demosaic_im = Image.fromarray(img.astype('uint8')).convert('RGB')
    demosaic_im = demosaic_im.save("/mnt/workspace/code/demosaic.jpg")

    if model:
        img = model.undistort(img)
        undistort_im = Image.fromarray(img.astype('uint8')).convert('RGB')
        undistort_im = undistort_im.save("/mnt/workspace/code/undistort.jpg")

    return np.array(img).astype(np.uint8)

