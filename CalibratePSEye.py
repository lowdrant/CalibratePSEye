#!/usr/bin/env python3
"""
This software is copyrighted material (C) Marion Anderson 2020

The library is provided under the Apache 2.0 License,
as found at https://www.apache.org/licenses/LICENSE-2.0

    OVERVIEW:

    This file contains a robust (some might say ornery) interface to chessboard
    calibrate a camera, undistort JPEG images, and most importantly (to me)
    store the calibration images and found parameters. It currently only works
    on UNIX systems, due to my laziness about filepath separators.

    Its primary feature is keeping track of all the calibration data for you.
    I made a point of having it aggressively save/duplicate/timestamp
    calibration data, parameters, and metadata so that the calibrations can
    be easily reconstructed at a moment's notice any time in the future.
    To this end, most functions will accept Python data structures or
    filename strings equivalently.

    The name comes from the PlayStation Eye, a lovely USB camera capable of
    100 FPS at 320x240px, with which this library was developed.

    GOTCHAS:

        1. Currently only works on UNIX systems (2020-06-22)

        2. Requires the specification of a directory to store calibration directories

        3. Only pays attention to JPEGs to match UofM archival standards

        4. Uses CHESSBOARD calibration

    MAIN CLASSES:

        CalibratePSEye -- The workhorse of this library. Stores & handles all data

    MAIN FUNCTIONS:

        open_camera -- open a camera, with options to set framesize and FPS

        calibrate -- calibrate using a photobooth-like interface

        correct_and_save -- undistort jpg images with reference to calibration data

        livetest -- view undistorted images from a live camera video feed

    EXAMPLES:

        1. Calibration

            bs = (4,3)
            processing_params = {
                'boardsize': bs,
                'winsize': (11, 11),
                'zerozone': (-1, -1),
                'criteria': (TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER, 30, 0.001)
            }
            # physical locations of chessboard corners
            sqdim = 40  # square length in [units]
            objp = zeros((boardsize[0]*boardsize[1],3), float32)
            objp[:,:2] = mgrid[0:bs[0],0:bs[1]].T.reshape(-1,2) * sqdim
            # see docstring for additional optional arguments
            calibrate(processing_params, objp, cam=SSS)

        2. Undistortion

            rootpath = dirname(realpath(__file__ + '/..))
            imgpath = rootpath + '/basepath/psEye_20200519-142131/rawframes'
            calibfn = rootpath + '/basepath/calibration_20200507-180600/camera_params.csv'
            undistorted_imgs = correct_and_save(calibfn, imgpath)

    TECHNICAL DETAILS:

    At the time of this writing (2020-06-22), my computer used OpenCV 3.2.0.

    This library builds heavily on work already done by the folks at OpenCV, so
    any camera calibration tutorials or textbooks should provide plenty of
    insight into the computer vision workings of this library. Two resouces I
    found particularly helpful were:

        https://docs.opencv.org/3.2.0/dc/dbb/tutorial_py_calibration.html

        "Robotics, Vision, and Control," 2nd Ed by Peter Corke
         ISBN-13: 978-3319544120

    Data storage follows the University of Michigan's Deep Blue Preservation
    and Format Support Policy for Level 1 preservation support, as found at
        https://deepblue.lib.umich.edu/static/about/deepbluepreservation.html

    TODO:
    
    identify decimal precision of calibration parameters
    switch from CSV to YAML
"""

import logging  # for debugging
from os import listdir, mkdir
from os import remove as os_remove
from os.path import basename, dirname, isdir, realpath
from os.path import exists as os_exists
from shutil import copyfile, rmtree
from time import localtime, strftime
from time import time as now

from cv2 import (CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
                 COLOR_GRAY2RGB, COLOR_RGB2GRAY, IMWRITE_JPEG_QUALITY,
                 TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, VideoCapture,
                 calibrateCamera, cvtColor, destroyAllWindows,
                 drawChessboardCorners, findChessboardCorners, undistort,
                 waitKey)
from cv2 import imread as cv_imread
from cv2 import imshow as cv_imshow
from cv2 import imwrite as cv_imwrite
from cv2 import putText as cv_putText
from numpy import (array_equal, asarray, concatenate, float32, floor, log10,
                   mgrid, ndarray, newaxis, reshape, round_, uint8, zeros)


__all__ = ['CalibratePSEye', 'open_camera', 'calibrate', 'correct_and_save', 'livetest']


class CalibratePSEye():

    def __init__( self, basepath=None ):
        """
        INPUTS
            (optional)
            basepath -- Directory to store calibrations; None
                        if basepath is None, `basepath=dirname(__file__)`

        CALIBRATION STORAGE
            Calibration paramaters, matrices, images, etc. will be stored in
            `basepath/calibration_YYYYmmdd-HHMMSS`. There are a bunch of
            different parameters associated with camera calibration, including
            setup, calibration images, and found camera parameters. This class
            saves all of that together so that the calibration process can be
            identically recomputed years in the future.

        ATTRIBUTES
            basepath      -- directory where calibration directories get placed
            calibpath     -- directory to save images/calibrations (typically) timestamped

            (see tutorial or `init_chessboard` for details)
            objp          -- array denoting chessboard corners in desired units
            boardsize     -- (x,y) number of chessboard corners. x*y>=6
            self.winsize  -- (x,y) corner-finding parameter (px)
                             working example: (11,11)
            self.zerozone -- (x,y) corner-finding parameter
                             working example: (-1,-1)
            self.criteria -- (x,y,z) numerical method corner-finding parameters
                             working example: (3,30,0.001)

            (calibration data)
            img_arr       -- array of calibration images
            objpoints     -- array of objp
            corners_arr   -- array of chessboard corners locations in image plane (px)
            w             -- width (px) of images
            h             -- height (px) of images

            (undistortion parameters)
            cameraMatrix  -- camera matrix. standard computer vision object
            distCoeffs    -- distortion coefficients. Taylor Series

        METHODS
            clear                -- reset all class data
            init_chessboard      -- initialize chessboard calibration parameters
            get_timestamp        -- get calibration timestamp
            load_calib_imgs      -- load all jpgs from a directory of calibration images
            record_calib_imgs    -- photobooth interface to capture calibration images
            clean_calib_imgs     -- interface to remove bad calibration images from a directory
            compute_calibrations -- compute calibrations after load/record calib_imgs
            save_calibrations    -- save calibrations
            load_calibrations    -- load calibrations from textfile
            correct              -- undistort (correct) image (uint8 array)
            correct_and_save     -- undistort (correct) images in directory, and save
            inspect              -- view calibration images and their undistorted counterparts
            removepath           -- remove autogenerated calibration directory
        """
        self.clear()
        if basepath is None:
            self.basepath = dirname(realpath(__file__))
        else:
            self.basepath = basepath

    def clear( self ):
            """
            Clear collected/computed calibration data.

            Called by init_chessboard()
            """
            self.calibpath = None     # paths

            self.img_arr = None       # pre-calibration
            self.objpoints = []
            self.corners_arr = []
            self.w = None
            self.h = None

            self.boardsize = None     # chessboard parameters
            self.winsize = None
            self.zerozone = None
            self.criteria = None
            self.objp = None

            self.cameraMatrix = None  # calibration
            self.distCoeffs = None

    def init_chessboard( self, processing_params, objp ):
        """
        Initialize chessboard calibration data. Will create & save params to
        new directory. Already called if class initialized with not-None args.

        INPUTS
            params -- dict/str -- Processing parameters, dict or CSV filename
            objp -- N x 3 -- 3D locations of chessboard corners in desired units

        CSV FORMAT
            - delimiter: ', '
            - no floats, round and convert to int using known precision
            - strings in double quotes

        EXAMPLE PARAMS
            {'boardsize': (7,6),    # corner count of chessboard
             'winsize': (11, 11),   # search param for finding board corners
             'zerozone': (-1, -1),  # search param for finding board corners
             'critera': (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
            }  # criteria -> sub-pixel corner finding search param

        EXAMPLE OBJP
            objp = np.zeros((6*7,3), np.float32)
            objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        """
        # asserts handled in the individual loaders
        self.clear()
        self._load_processing_params(processing_params)
        self._load_objp(objp)
        self._verify_calib_params()
        self._create_calib_path()
        self._save_calib_params()

    def _load_processing_params( self, params ):
        """
        Load in corner finding parameters for OpenCV chessboard functions.

        INPUTS
            params -- dict/str -- Processing parameters, dict or CSV filename

        CSV FORMAT
            delimiter: ', '
            no floats, round and convert to int using known precision
            strings in double quotes

        EXAMPLE PARAMS DICT
            {'boardsize': (3,4),    # corner count of chessboard
             'winsize': (11, 11),   # search param for finding board corners
             'zerozone': (-1, -1),  # search param for finding board corners
             'critera': (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
            }  # criteria -> sub-pixel corner finding search param

        EXAMPLE OBJP NDARRAY
            objp = np.zeros((boardsize[0]*boardsize[1],3), dtype='float32')
            objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        """
        if type(params) not in (dict, str):
            raise TypeError('params must be dict or str')

        # read from csv
        if type(params) == str:
            with open(params, 'r') as f:
                lines = f.read().splitlines()
            param_text = [line.split(', ') for line in lines]
            params = dict()  # save as dict for compatability w/ non-dict case
            for p in param_text:
                # sanitize
                name = p[0].replace('\"', '')
                if name not in ('boardsize', 'zerozone', 'winsize', 'criteria'):
                    logging.warning('name: \'%s\' not valid calib name' % name)
                    continue
                # interpret
                if name == 'boardsize':
                    params[name] = (int(p[1]), int(p[2]))
                elif name == 'zerozone':
                    params[name]= (int(p[1]), int(p[2]))
                elif name == 'winsize':
                    params[name] = (int(p[1]), int(p[2]))
                elif name == 'criteria':
                    # first value should be single-digit number
                    #   use that to determine decimal place shift
                    powshift = -int(log10(int(p[1])))
                    params[name] = (
                        int(int(p[1]) * 10**powshift),  # mode (int)
                        int(int(p[2]) * 10**powshift),  # something about pixels (int)
                        int(p[3]) * 10**powshift        # EPS termination (float)
                    )

        # assign data
        self.boardsize = params['boardsize']
        self.winsize = params['winsize']
        self.zerozone = params['zerozone']
        self.criteria = params['criteria']  # recursion termination criteria

    def _load_objp( self, objp ):
        """
        Load in 3D chessboard corner positions for OpenCV chessboard functions.

        INPUTS
            objp -- N x 3 -- 3D locations of chessboard corners in desired units

        CSV FORMAT
            delimiter: ', '
            no floats, round and convert to int using known precision
            strings in double quotes

        EXAMPLE PARAMS DICT
            {'boardsize': (3,4),    # corner count of chessboard
             'winsize': (11, 11),   # search param for finding board corners
             'zerozone': (-1, -1),  # search param for finding board corners
             'critera': (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
            }  # criteria -> sub-pixel corner finding search param

        EXAMPLE OBJP NDARRAY
            objp = np.zeros((boardsize[0]*boardsize[1],3), dtype='float32')
            objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        """
        if type(objp) not in (ndarray, str):
            raise TypeError('\'objp\' must be ndarray or str')
        if type(objp) == ndarray:
            if len(objp.shape) != 2:
                raise TypeError('\'objp\' must be 2-dimensional array')
            if objp.shape[-1] != 3:
                raise TypeError('\'objp\' must be Nx3 array')
        if type(objp) == str:
            with open(objp, 'r') as f:
                text = f.read()
                data = text.split(', ')
                shape = [int(v) for v in data[0].replace('\"', '').split('x')]
                objp = reshape([int(v) for v in data[1:]], shape).astype('float32')
        self.objp = objp.copy()

    def _create_calib_path( self ):
        """
        Create timestamped directory for storing new calibration data.
        """
        # time_str not member var b/c load_calibrations might have weird interaction
        time_str = strftime('%Y%m%d-%H%M%S', localtime())
        file_path = dirname(dirname(realpath(__file__)))
        self.calibpath = self.basepath + '/calibration_' + time_str
        if not isdir(self.calibpath):
            if os_exists(self.calibpath):
                os_remove(self.calibpath)
            mkdir(self.calibpath)
        logging.info('set up calibration directory at \'%s\'' % self.calibpath)

    def get_timestamp( self ):
        """
        Internal helper to get timestamp from saved calibration directory.

        OUTPUTS
            'YYYYMMDD-HHmmss' string from calibration path
        """
        if self.calibpath is None:
            raise RuntimeError('\'calibpath\' is None')
        time_ndx = self.calibpath.rfind('-') - 8  # '-' from 'YYYYMMDD-HHmmss'
        timestr = self.calibpath[time_ndx:time_ndx+15]
        return timestr

    def _verify_calib_params( self ):
        """
        Verifies calibration parameters are "correct enough."

        this boils down to checking shape/size/type
        """
        try:
            assert len(self.boardsize) == 2
            assert floor(self.boardsize[0]) == self.boardsize[0]
            assert floor(self.boardsize[1]) == self.boardsize[1]
        except AssertionError:
            raise TypeError('\'boardsize\' must be length-2 iterable of ints')

        try:
            assert len(self.winsize) == 2
            assert floor(self.winsize[0]) == self.winsize[0]
            assert floor(self.winsize[1]) == self.winsize[1]
        except AssertionError:
            raise TypeError('\'winsize\' must be length-2 iterable of ints')

        try:
            assert len(self.zerozone) == 2
            assert floor(self.zerozone[0]) == self.zerozone[0]
            assert floor(self.zerozone[1]) == self.zerozone[1]
        except AssertionError:
            raise TypeError('\'zerozone\' must be length-2 iterable of ints')

        if len(self.criteria) != 3:
            raise TypeError('\'criteria\' must be length-3 iterable')

        try:
            assert self.objp.dtype == float32
            assert self.objp.shape[0] == self.boardsize[0] * self.boardsize[1]
            assert self.objp.shape[1] == 3
        except AssertionError:
            raise TypeError('\'objp\' must be Nx3 float32 ndarray')

    def _save_calib_params( self ):
        """
        Save calibration parameters as csvs to a timestamped directory.

        CSV FORMAT
            - delimiter: ', '
            - no floats, round and convert to int using known precision
            - strings in double quotes

        NOTE
            Chessboard corner positions are converted directly to integers. Be
            sure to scale units appropriately prior to this function.
        """
        # Params
        fn_params = self.calibpath + '/processing_params.csv'
        with open(fn_params, 'w') as f:
            # boardsize, winsize, zerozone all length-2 tuples of ints
            f.write('\"boardsize\", ' + ', '.join(str(v) for v in self.boardsize) + '\n')
            f.write('\"winsize\", ' + ', '.join(str(v) for v in self.winsize) + '\n')
            f.write('\"zerozone\", ' + ', '.join(str(v) for v in self.zerozone) + '\n')
            # compensate for decimal in criteria
            #  also, know for a fact that criteria[0] is single digit number
            criteria = [int(10**-int(log10(self.criteria[-1])) * v) for v in self.criteria]
            f.write('\"criteria\", ' + ', '.join(str(v) for v in criteria))
            f.flush()
        logging.debug('saved processing params to \'%s\'' % fn_params)

        # Object point
        #   precision is basically integer - more than 1mm precision is absurd
        fn_objpoints = self.calibpath + '/objp.csv'
        with open(fn_objpoints, 'w') as f:
            f.write(
                '\"' + 'x'.join(str(v) for v in self.objp.shape) + '\", '  # size
                + ', '.join(str(int(v)) for v in self.objp.flatten()))     # points
            f.flush()
        logging.debug('saved objpoints at \'%s\'' % fn_objpoints)

    def _find_chessboard( self, gray ):
        """
        Internal helper to find chessboard of a single grayscale image. This is
        re-used in every calib_imgs function.

        INPUTS
            gray -- uint8 W x H or W x H x 1 -- grayscale image

        OUTPUTS
            Detected corners, None if no corners found
        """
        if gray.dtype != uint8:
            raise TypeError('gray must be uint8')
        if len(gray.shape) not in (2, 3):
            raise TypeError('gray must be dimension 2 or 3 array')
        if len(gray.shape) == 3:
            if gray.shape[-1] != 1:
                raise TypeError('gray must have 1 color dimension')
        ret, corners = findChessboardCorners(gray, self.boardsize, None)
        if not ret:
            corners = None
        return corners

    def load_calib_imgs( self, img_path, clean=False ):
        """
        Load calibration JPGs from directory & get chessboard. Be sure to
        call init_chessboard() first.

        INPUTS
            img_path -- str -- path to calibration jpegs
            (optional)
            clean -- bool -- Whether or not to go through sanitization; False

        EXCEPTIONS
            raises RuntimeError: when chessboard hasn't been properly
                initialized by constructor or `init_chessboard`.

        ALGORITHM
            Finds all files ending in '.jpg' and loads them. Objp should have
            been handled in init_chessboard()
        """
        if type(img_path) != str:
            raise TypeError('img_path must be str')
        if not (type(clean) == bool or clean in (0, 1)):
            raise TypeError('clean must be bool')
        if self.img_arr is not None or self.calibpath is None:
            raise RuntimeError('Did you call init_chessboard() first?')

        # Find calibration images and process
        potential_files = listdir(img_path)
        fn_imgs = [
            img_path+'/'+f for f in potential_files if f[-4:].lower() == '.jpg'
        ]
        imageshape = cv_imread(fn_imgs[0]).shape
        self.h = imageshape[0]
        self.w = imageshape[1]

        # Save images in calib path
        rawpath = self.calibpath + '/raw'   # copy to current calibration dir
        if rawpath != img_path:
            if not isdir(rawpath):
                if os_exists(rawpath):
                    os_remove(rawpath)
                mkdir(rawpath)
            for f in fn_imgs:
                copyfile(f, rawpath + '/' + basename(f))
        # corners frames for debugging
        cpath = self.calibpath + '/corners'  # save drawn corners for debug
        mkdir(cpath)
        logging.info('saving raw frames to \'%s\'' % rawpath)
        logging.info('saving corners frames to \'%s\'' % cpath)

        # Load images
        self.img_arr = zeros((self.h, self.w, 1, len(fn_imgs)), uint8)
        for i in range(len(fn_imgs)):
            f = fn_imgs[i]
            if imageshape[-1] == 3:
                self.img_arr[...,i] = cvtColor(cv_imread(f), COLOR_RGB2GRAY)[...,newaxis]
            else:
                self.img_arr[...,i] = cv_imread(f)

        # Chessboard computations
        logging.debug('finding chessboards...')
        for i in range(self.img_arr.shape[-1]):
            gray = self.img_arr[...,i].copy()
            corners = self._find_chessboard(gray)
            if corners is None:
                logging.error(
                    'Failed to find chessboard at frame \'%s\'' % str(i+1).zfill(5)
                )
                continue
            self.corners_arr.append(corners)
            self.objpoints.append(self.objp)  # 3d position (same for all?)

            # save chessboard images for debugging
            #   cvt to rgb for color chessboard
            fn_c = cpath + ('/f%s' % str(i+1).zfill(5)) + '.jpg'
            gray_color = cvtColor(gray, COLOR_GRAY2RGB)
            img_corners = drawChessboardCorners(gray_color, self.boardsize, corners, 1)
            cv_imwrite(fn_c, img_corners, (IMWRITE_JPEG_QUALITY, 100))

        # Go through chessboards to make sure okay
        if clean:
            basepath = dirname(cpath)
            self.clean_calib_imgs(basepath=basepath)
        logging.debug('load_calib_imgs() done!')

    def record_calib_imgs( self, **kwargs ):
        """
        Provides photobooth-esque countdown interface. Saves frames to calib
        path in subdirectories `raw/` and `corners/`. Be sure to
        initialize the chessboard first.

        INPUTS
            (optional)
            cam       -- str/int -- camera descriptor for VideoCapture; '/dev/psEye'
            nframes   -- int     -- number of frames to record for calibration; 15
            w         -- int     -- width (px) to set camera frame; 320
            h         -- int     -- height (px) to set camera frame; 240
            fps       -- int     -- frames per second to set camera; 100
            countdown -- int     -- seconds to countdown before recording frame; 3

        EXCEPTIONS
            raises RuntimeError when chessboard hasn't been properly initialized
                by constructor or `init_chessboard`.
        """
        cam = kwargs.get('cam', '/dev/psEye')
        nframes = kwargs.get('nframes', 15)
        w = kwargs.get('w', 320)
        h = kwargs.get('h', 240)
        countdown = kwargs.get('countdown', 3)
        if type(nframes) != int:
            raise TypeError('nframes must be integer')
        if type(countdown) != int:
            raise TypeError('countdown must be integer')
        if self.img_arr is not None or self.calibpath is None:
            raise RuntimeError('Did you call init_chessboard() first?')
        cap = open_camera(cam, w, h, 100)  # this handles asserts
        self.w = int(cap.get(CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(CAP_PROP_FRAME_HEIGHT))
        self.img_arr = zeros((self.h,self.w,1,nframes), dtype=uint8)  # raw frames
        clist = []                                                    # corners frames
        self.corners_arr = []
        self.objpoints = []

        # Recording
        sc = 0  # "sample count"
        timer_ref = now()
        timer = lambda: 1+int(countdown + timer_ref - now())  # countDOWN 3,2,1
        try:  # try/except to make sure camera device gets released
            img = zeros(self.img_arr.shape[:-1])  # for immediate cv_imshow
            while sc < nframes:
                # Display at top so can always exit
                cv_imshow('capture', img)
                press = waitKey(20)
                if press in (113, 81, 27):  # q, Q, esc:
                    logging.debug('quitting record')
                    break

                # Find chessboard, if possible
                ret, raw = cap.read()
                if not ret:
                    logging.error('Failed to access frame')
                    timer_ref = now()  # reset timer when things go wrong
                    continue
                gray = cvtColor(raw, COLOR_RGB2GRAY)
                corners = self._find_chessboard(gray)

                # Compute visual feedback
                if corners is None:  # alert to unfindable chessboard
                    img = raw.copy()
                    cv_putText(img,'NO CHESSBOARD',(5,15),1,1,(0,0,255),thickness=2)
                    timer_ref = now()  # reset timer when chessboard isn't viable
                else:  # show countdown and progess
                    board1 = drawChessboardCorners(raw, self.boardsize, corners, ret)
                    img = board1.copy()
                    cv_putText(img,'T-%ds' % timer(),(5,15),1,1,(0,0,255),thickness=2)
                cv_putText(img,'%d/%d' % (sc+1,nframes),(5,30),1,1,(0,0,255),thickness=2)

                # Capture image
                if timer() <= 0:
                    # image saving
                    self.img_arr[...,sc] = gray.copy()[...,newaxis]
                    clist.append(board1)

                    # for camera calibration
                    self.corners_arr.append(corners)
                    self.objpoints.append(self.objp)

                    # program progess/display
                    img = zeros(raw.shape, dtype=uint8) + 255  # "flash" camera
                    sc += 1
                    timer_ref = now()

            # Save images to file
            if self.calibpath is None:
                self._create_calib_path()
            # Create save directories
            rawpath = self.calibpath + '/raw'
            cpath = self.calibpath + '/corners'
            for p in (rawpath, cpath):
                if not isdir(p):
                    if os_exists(p):
                        logging.warning(
                            '\'%s\' exists, but is not directory. Overwriting.' % p
                        )
                        os_remove(p)
                    mkdir(p)
            for i in range(nframes):
                fn_raw = rawpath + ('/f%s' % str(i+1).zfill(5)) + '.jpg'
                fn_c = cpath + ('/f%s' % str(i+1).zfill(5)) + '.jpg'
                cv_imwrite(fn_raw, self.img_arr[...,i], (IMWRITE_JPEG_QUALITY, 100))
                cv_imwrite(fn_c, clist[i], (IMWRITE_JPEG_QUALITY, 100))

        # Close Capture
        except Exception as e:
            logging.error(e)
        finally:
            cap.release()
            destroyAllWindows()
            logging.debug('released \'%s\'' % cam)

    def clean_calib_imgs( self, basepath=None, rawpath=None, cpath=None ):
        """
        Provides interface to sanitize calibration images using a
        tutorial-like imshow() interface. Deletes frames, and removes elements
        from self.corners_arr, if they are not None.

        INPUTS
            (optional)
            basepath -- str -- Base path for calib frames; None
            rawpath  -- str -- Path to frames for calibration; None
            cpath    -- str -- Path to frames with chessboard; None

        NOTE
            The frames in each path must have the same name, e.g. 'f00001.jpg'

        USAGE
            SPECIFIED ONLY BASEPATH
                When all other paths the frame paths will
                be defined as:
                    rawpath = basepath + '/raw'
                    cpath = basepath + '/corners'

            PATHS TO ALL FRAMES
                If rawpath and cpath are all explicitly defined, they will be
                used as is. basepath will be ignored.
        """
        # Parse paths
        if basepath is not None:
            basepath = realpath(basepath)
            if basepath[basepath.rfind('/')+1:] not in ('raw', 'corners', 'corners2'):
                logging.warning(
                    'Assuming \'%s\' is base dir for all unspecified frames' % basepath
                )
                basepath = basepath
            else:
                basepath = dirname(basepath)
        if rawpath is None:
            rawpath = basepath + '/raw'
        if cpath is None:
            cpath = basepath + '/corners'

        # Select images to save
        fn_imgs = [f for f in listdir(cpath) if f[-4:].lower() == '.jpg']
        fn_imgs.sort()
        i = 0
        while i < len(fn_imgs):
            f = fn_imgs[i]
            img = cv_imread(cpath+'/'+f)
            if img is None:  # catch deleted image
                fn_imgs.pop(i)
                if self.corners_arr is not None:
                    self.corners_arr.pop(i)
                i += 1
                continue
            cv_putText(img, f, (5,15), 1, 1,
                       (0,0,255), thickness=2)
            cv_putText(img, '\'r\' to remove', (5,30), 1, 1,
                       (0,0,255), thickness=2)
            cv_putText(img, 'left bracket to go back', (5,45), 1, 1,
                       (0,0,255), thickness=2)
            cv_imshow('image', img)

            # interface
            press = waitKey(-1)
            if press == ord('r'):
                fn = fn_imgs[i][fn_imgs[i].rfind('f'):]
                os_remove(rawpath + '/' + fn)
                os_remove(cpath + '/' + fn)
            elif press in (27, 81, 113):  # esc, q, Q
                break
            elif press == 91:  # left bracket
                i -= 2
            i += 1
        destroyAllWindows()

    def compute_calibrations( self ):
        """
        Generate calibration matrices from member data.

        EXCEPTIONS
            Raises RuntimeError when member data needed for calibration is unset

        OUTPUTS
            the overall RMS re-projection error, per `cv2.calibrateCamera`

        NOTE
            computation requires self.corners, self.objpoints, self.w, self.h
        """
        if (len(self.corners_arr) == 0 or len(self.objpoints) == 0
                or self.w is None or self.h is None):
            raise RuntimeError('Did you assign calibration member data?')
        logging.debug('computing calibrations...')
        # Intial undistortion
        self.rms, self.cameraMatrix, self.distCoeffs, _, _ = calibrateCamera(
            self.objpoints, self.corners_arr, (self.w, self.h), None, None
        )
        logging.debug('calibrations computed')
        return self.rms

    def save_calibrations( self ):
        """
        Save calibration results & detected corners as CSVs in timestamped directory.

        CSV FORMAT
            - delimiter: ', '
            - no floats, round and convert to int using known precision
            - strings in double quotes

            e.g. "varname", var[0], var[1], ...\\n

        DECISION
            Matrices are rounded to the 4th decimal place. For the cases of
            cameraMatrix and distCoeffs, they differ in the 5th decimal place
            between my desktop computer and my laptop. I still need to check
            pixel accuracy, but so far this is a good start.

        NOTE
            corners_arr might be None, so it is only saved if it exists as
            meaningful data.
        """
        if self.calibpath is None:
            self._create_calib_path()
        # convert matrices to appropriate strings
        f64_to_csv = lambda arr: ', '.join([
            str(int(round(v*10**4))) for v in asarray(arr).flatten()
        ])
        fn_calibration = self.calibpath + '/camera_params.csv'
        with open(fn_calibration, 'w') as f:
            # writing
            #   var name
            #   shape
            #   flattened data
            # f.write('\"camSize\", \"%dx%d\"\n' % (self.w, self.h))
            f.write('\"w\", %d\n' % self.w)
            f.write('\"h\", %d\n' % self.h)
            f.write('\"cameraMatrix\", \"'
                    + 'x'.join(str(v) for v in self.cameraMatrix.shape)
                    + '\", ' + f64_to_csv(self.cameraMatrix) + '\n')
            f.write('\"distCoeffs\", \"'
                    + 'x'.join(str(v) for v in self.distCoeffs.shape)
                    + '\", ' + f64_to_csv(self.distCoeffs) + '\n')
            # f.write('')
            f.flush()

    def load_calibrations( self, calibs ):
        """
        Load calibration matrices.

        INPUTS
            calibs -- dict/str -- Calibration data struct or CSV filename

        WARNING
            If calibs is a filename, it must be a CSV file with fields:
            \"w\", \"h\", \"cameraMatrix\", \"distCoeffs\"

        CSV FORMAT
            - delimiter: ', '
            - no floats, round and convert to int using known precision
            - strings in double quotes

            e.g. "varname", var[0], var[1], ...\\n
        """
        if type(calibs) not in (dict, str):
            raise TypeError('calibs must be dict or str')

        if type(calibs) == dict:
            self.w = calibs['w']
            self.h = calibs['h']
            self.cameraMatrix = calibs['cameraMatrix']
            self.distCoeffs = calibs['distCoeffs']
        else:  # parse csv
            self.calibpath = dirname(realpath(calibs))
            logging.debug('loading calibrations from \'%s\'' % calibs)
            with open(calibs, 'r') as f:
                lines = f.read().splitlines()

            entries = [line.split(', ') for line in lines]
            for c in entries:
                name = c[0].replace('\"', '')
                if name not in ('cameraMatrix', 'distCoeffs', 'w', 'h'):
                    logging.warning(
                        'variable name: \'%s\' not valid calib name' % name
                    )
                    continue

                # Scale decimal numbers
                if name not in ('w', 'h'):
                    shape = [int(v) for v in c[1].replace('\"', '').split('x')]
                    data = asarray([int(v) for v in c[2:]]) / 10**4  # already rounded
                    data = data.reshape(shape)
                if name in ('cameraMatrix', 'distCoeffs'):
                    setattr(self, name, data.astype('float64'))
                if name in ('w', 'h'):
                    setattr(self, name, int(c[1]))

    def correct( self, frames ):
        """
        Undistort a frame or frames.

        INPUTS
            frames -- uint8 N x H x W x BYTES -- Frames to undistort

        OUTPUTS
            undistorted_imgs -- uint8 N x H x W x BYTES
        """
        # Preprocess to ensure iteration code works
        if type(frames) == list and len(frames) > 0:
            frames = asarray(frames, dtype='uint8')
        if frames.dtype != uint8:
            raise TypeError('frames must be type uint8')
        if len(frames.shape) not in (3,4):
            raise TypeError('frames must be 3 or 4 dimensional array')
        imgs = frames.copy()
        if len(imgs.shape) != 4:  # temporarily resize for iteration case
            imgs = frames[newaxis,...].copy()

        # Correct all frames
        undistorted_imgs = zeros(imgs.shape, dtype=uint8)
        for i in range(imgs.shape[0]):
            img = imgs[i,...].copy()
            ud = undistort(img, self.cameraMatrix, self.distCoeffs, None)
            undistorted_imgs[i,...] = ud.copy()
        # Post process to match user input
        #   simplifies 1-image-passed case for user
        if len(frames.shape) == 3:
            undistorted_imgs = undistorted_imgs.squeeze()
        return undistorted_imgs

    def correct_and_save( self, imgpath ):
        """
        Correct images and then saves them with the calibration timestamp.

        INPUTS
            imgpath -- str -- path to images to correct

        OUTPUTS
            undistorted_imgs -- uint8 N x H x W x BYTES

        NOTE
            Only corrects filenames ending in '.jpg'

        ALGORITHM
            Saves corrected images at same level as the 'imgpath' directory, but
            adds a timestamp which references the calibration dataset/params
            used. `undistorted_YYMMDD-HHmmss/` and `remapped_YYMMDD-HHmmss/`

        EXCEPTIONS
            Raises RuntimeError If calibration path isn't set. Likely results
                from failing to load/compute calibration matrices.
        """
        if type(imgpath) != str:
            raise TypeError('imgpath must be str')
        if self.calibpath is None:
            raise RuntimeError(
                'Be sure to set self.calibpath (did you compute/load calibrations?)'
            )

        # Find images to correct
        potential_imgs = listdir(imgpath)
        fn_imgs = [f for f in potential_imgs if f[-4:].lower() == '.jpg']  # select jpegs
        img_list = [cv_imread(imgpath+'/'+fn) for fn in fn_imgs]           # read all jpegs

        # Make Save Directories
        savedir = realpath(imgpath + '/..')
        timestamp = self.get_timestamp()
        copyfile(self.calibpath + '/camera_params.csv',
                        savedir + '/' + timestamp + '_camera_params.csv')
        ud_path = savedir + '/' + timestamp + '_undistorted'
        if not isdir(ud_path):
            if os_exists(ud_path):
                os_remove(ud_path)
            mkdir(ud_path)

        # Correct & Save Frames
        ud = self.correct(img_list)
        for i in range(len(img_list)):
            fnud = ud_path + ('/f%s' % str(i+1).zfill(5)) + '.jpg'
            cv_imwrite(fnud, ud[i,...], (IMWRITE_JPEG_QUALITY, 100))

        # Return in case we want to use them later
        return ud

    def inspect( self ):
        """
        Run through calibration images, showing original and undistorted.
        Intended to be run after saving calibration images.
        """
        # access calibration images
        if self.calibpath is None:
            raise RuntimeError('calibpath is unset')
        imgpath = self.calibpath + '/corners2'
        fns = [imgpath+'/'+f for f in listdir(imgpath) if f[-4:].lower() == '.jpg']
        fns.sort()
        imgs = [cv_imread(f) for f in fns]
        shape = list(imgs[0].shape)
        shape.append(len(imgs))

        # undistort
        #   go through 1-by-1 b/c asarray(imgs) gets shaped wrong & reshape messes with things
        ud = zeros(shape, dtype=uint8)
        for i in range(len(imgs)):
            ud[...,i] = self.correct(imgs[i].squeeze())

        # display side-by-side
        for i in range(len(imgs)):
            img = imgs[i].copy()
            u = ud[...,i].copy()
            cv_putText(img,fns[i][-10:],(5,15),1,1,(0,0,255),thickness=2)
            cv_putText(img,'Press space for next image',(5,30),1,1,(0,0,255),thickness=2)
            cv_putText(u,'Undistort',(5,15),1,1,(0,0,255),thickness=2)
            frame = concatenate((img, u), axis=0)
            cv_imshow('frame', frame)
            press = waitKey(-1)
            if press == ord('q'):
                break
        destroyAllWindows()

    def removepath( self ):
        """
        Remove calibration files from testing calibration computation.
        """
        if self.calibpath is not None:
            rmtree(self.calibpath)


def open_camera( cam='/dev/psEye', w=320, h=240, fps=100 ):
    """
    Opens camera and configures height, width, and fps settings.

    INPUTS
        (optional)
        cam -- str/int -- camera descriptor for VideoCapture; '/dev/psEye'
        w   -- int     -- width (px) to set camera frame; 320
        h   -- int     -- height (px) to set camera frame; 240
        fps -- int     -- frames per second to set camera; 100

    OUTPUTS
        cv2.VideoCapture object

    EXCEPTIONS
        Raises RuntimeError when unable to open camera
    """
    if type(cam) not in (str, int):
        raise TypeError('\'cam\' must be int or str')
    if type(w) != int:
        raise TypeError('\'w\' must be int')
    if type(h) != int:
        raise TypeError('\'h\' must be int')
    if type(fps) != int:
        raise TypeError('\'fps\' must be int')
    cap = VideoCapture(cam)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('failed to open camera \'%s\'' % cam)
    cap.set(CAP_PROP_FRAME_HEIGHT, h)
    cap.set(CAP_PROP_FRAME_WIDTH, w)
    cap.set(CAP_PROP_FPS, fps)
    return cap


def calibrate( params, objp, basepath=None, **kwargs):
    """
    Record calibration images using a photobooth-like interface, and then
    save calibration matrices to a timestamped directory under `basepath`.

    INPUTS
        params -- dict/str -- Processing parameters, dict or CSV filename
        objp -- N x 3 -- 3D locations of chessboard corners in desired units
        (optional)
        basepath  -- str     -- directory to store calibrations; None
        cam       -- str/int -- camera descriptor for VideoCapture; '/dev/psEye'
        nframes   -- int     -- number of frames to record for calibration; 15
        w         -- int     -- width (px) to set camera frame; 320
        h         -- int     -- height (px) to set camera frame; 240
        fps       -- int     -- frames per second to set camera; 100
        countdown -- int     -- seconds to countdown before recording frame; 3

    OUTPUTS
        CalibratePSEye object with populated calibration member data.

    CSV FORMAT
        delimiter: ', '
        no floats, round and convert to int using known precision
        strings in double quotes

    EXAMPLE PARAMS DICT
        {'boardsize': (3,4),    # corner count of chessboard
         'winsize': (11, 11),   # search param for finding board corners
         'zerozone': (-1, -1),  # search param for finding board corners
         'critera': (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
        }  # criteria -> sub-pixel corner finding search param

    EXAMPLE OBJP NDARRAY
        objp = np.zeros((boardsize[0]*boardsize[1],3), dtype='float32')
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    """
    calibs = CalibratePSEye(basepath=basepath)
    calibs.record_calib_imgs(**kwargs)
       # cam=kwargs.get('cam', '/dev/psEye'), nframes=kwargs.get('nframes', 15),
       # w=kwargs.get('w', 320), h=kwargs.get('h', 240), fps=kwargs.get('fps', 100),
       # countdown=kwargs.get('countdown', 3)
    #)
    calibs.compute_calibrations()
    calibs.save_calibrations()
    return calibs


def correct_and_save( calibs, imgdir ):
    """
    Correct all `.jpg` images in imgdir and then save corrected images
    in a timestamped subdirectory of imgdir. The timestamp references the
    calibration data used.

    INPUTS
        calibs  -- str/CalibratePSEye -- CSV calibration fn or CalibratePSEye
                                         object with loaded calibrations
        imgdir  -- str                -- path to directory of images

    OUTPUTS
        undistorted_imgs
    """
    if type(calibs) == str:
        fn = calibs
        calibs = CalibratePSEye()
        calibs.load_calibrations(fn)
    if type(calibs) != CalibratePSEye:
        raise RuntimeError('\'calibs\' is not CalibratePSEye')
    return calibs.correct_and_save(imgdir)


def livetest( calibs, cam='/dev/psEye', w=320, h=240, fps=100 ):
    """
    Visually inspect calibrations on images actively streaming in from cam.

    INPUTS
        calibs -- str/CalibratePSEye -- fn to CSV calibrations or object with
                                        calibrations already loaded
        (optional)
        cam    -- str/int -- camera descriptor for VideoCapture; '/dev/psEye'
        w      -- int -- width (px) to set camera frame; 320
        h      -- int -- height (px) to set camera frame; 240
        fps    -- int -- frames per second to set camera; 100

    CSV FORMAT
        - delimiter: ', '
        - no floats, round and convert to int using known precision
        - strings in double quotes

    INTERFACE
        'space' to pause stream, 'q' to quit stream.
    """
    # Load calibrations if necessary
    if type(calibs) == str:
        fn = calibs  # resave to overwrite `calibs`
        calibs = CalibratePSEye()
        calibs.load_calibrations(fn)
    if type(calibs) != CalibratePSEye:
        raise RuntimeError('\'calibs\' is not type \'CalibratePSEye\'')

    # open_camera handles type asserts for camera params
    cap = open_camera(cam, w, h, fps)

    # Stream undistortion
    # try/finally to ensure cap.release() is called
    try:
        while 1:
            ret, raw = cap.read()
            if not ret:
                logging.warning('failed to read frame from \'%s\'' % cam)
            ud = calibs.correct(raw)

            # label data for user
            cv_putText(raw, 'Space to pause, q to quit', (5,15), 1, 1,
                       (0,0,255), thickness=2)
            cv_putText(ud, 'undistort', (5,15), 1, 1, (0,0,255), thickness=2)
            total = concatenate((raw, ud), axis=0)  # 1 frame for cleanliness
            cv_imshow('total', total)

            press = waitKey(30)
            if press == ord('q'):
                break
            elif press == ord(' '):
                if waitKey(-1) == ord('q'):
                    break
    finally:
        cap.release()
        destroyAllWindows()

if __name__ == '__main__':
    print(
        'You have successfully called a module meant to be imported!'
    )
