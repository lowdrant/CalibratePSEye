#!/usr/bin/env python3
"""
This software is copyrighted material (C) Marion Anderson 2020

The library is provided under the Apache 2.0 License,
as found at https://www.apache.org/licenses/LICENSE-2.0

    OVERVIEW:

    Investigate the alpha parameter in `cv2.getOptimalNewCameraMatrix`. Requires
    `CalibratePSEye.py` in PYTHONPATH.

    CONCLUSION:

    alpha=1 creates a black ring around the image, and alpha=0 expands the
    image to fill the frame, and newCameraMatrix is unnecessary

    USAGE:

        alpha.py [-h] [--cam CAM] [--width W] [--height H] [--calibs CALIBS]

        optional arguments:
          -h, --help       show this help message and exit
          --cam CAM        camera device to access (default: /dev/psEye)
          --width W        width [px] of image plane (default: 320)
          --height H       height [px] of image plane (default: 240)
          --calibs CALIBS  Calibration CSV to load. See CalibratePSEye.py for details.
                           (default: tests/calibs_00000000-000000/camera_params.csv)
"""

import logging
from argparse import ArgumentParser
from os.path import realpath

from cv2 import (CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, createTrackbar,
                 destroyAllWindows, getOptimalNewCameraMatrix, namedWindow,
                 undistort, waitKey)
from cv2 import imshow as cv_imshow
from cv2 import putText as cv_putText
from numpy import concatenate, zeros

from CalibratePSEye import CalibratePSEye, open_camera


parser = ArgumentParser(
    description=(
        'Investigate `getOptimalNewCameraMatrix` alpha parameter Requires'
        + ' `CalibratePSEye.py` in PYTHONPATH.')
)
parser.add_argument(
    '--cam', default='/dev/psEye',
    help='camera device to access (default: /dev/psEye)'
)
parser.add_argument(
    '--width', dest='w', default=320,
    help='width [px] of image plane (default: 320)'
)
parser.add_argument(
    '--height', dest='h', default=240,
    help='height [px] of image plane (default: 240)'
)
rootdir = realpath(__file__+'/../..')
fn_test = rootdir + '/tests/calibs_00000000-000000/camera_params.csv'
parser.add_argument(
    '--calibs', default=fn_test,
    help=('Calibration CSV to load. See CalibratePSEye.py for details.'
          + ' (default: tests/calibs_00000000-000000/camera_params.csv)')
)
args = parser.parse_args()
calib = CalibratePSEye()
calib.load_calibrations(fn_test)

alpha_val = 1


def getAlpha(val):
    """
    Normalize alpha from trackbar for `getOptimalNewCameraMatrix`
    """
    global alpha_val
    alpha_val = val / 100.0


cap = open_camera(cam=args.cam, w=args.w, h=args.h)
try:
    namedWindow('win')
    createTrackbar('alpha', 'win', 100, 100, getAlpha)
    W = int(cap.get(CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(CAP_PROP_FRAME_HEIGHT))
    u = zeros((H*2,W,3), dtype='uint8')  # pre-initialize display
    while 1:
        # Interface
        cv_imshow('win', u)
        press = waitKey(20)
        if press in (27, 81, 113):
            break

        # Update Camera Matrix
        ncm, roi = getOptimalNewCameraMatrix(
            calib.cameraMatrix, calib.distCoeffs, (W, H), alpha_val
        )
        x, y, w, h = roi

        # Read frames and correct
        r, f = cap.read()
        if not r:
            logging.warning('Unable to read frame')
            continue
        _u1 = undistort(
            f, calib.cameraMatrix, calib.distCoeffs, None, newCameraMatrix=ncm
        )
        u1 = zeros((H,W,3), dtype='uint8')
        u1[y:y+h,x:x+w] = _u1[y:y+h,x:x+w]
        u2 = calib.correct(f)

        cv_putText(u1, 'newCamMat', (5,15), 1,1, (0,0,255), thickness=2)
        cv_putText(u2, 'CamMat', (5,15), 1,1, (0,0,255), thickness=2)
        u = concatenate((u1,u2),axis=0)
except Exception as e:
    raise e
finally:
    cap.release()
    destroyAllWindows()
