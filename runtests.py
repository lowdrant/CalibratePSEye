#!/usr/bin/env python3
"""
This software is copyrighted material (C) Marion Anderson 2020

The library is provided under the Apache 2.0 License,
as found at https://www.apache.org/licenses/LICENSE-2.0

    OVERVIEW:
    
    This script provides unit testing for CalibratePSEye.py
        
    USAGE:
    
        runtests.py [-h] [--loglvl{CRITICAL,ERROR,WARNING,INFO,DEBUG}]
        
    TODO:
    
        1. check file permissions and existence at get-go
"""
from argparse import ArgumentParser
from logging import CRITICAL as log_CRITICAL
from logging import DEBUG as log_DEBUG
from logging import ERROR as log_ERROR
from logging import INFO as log_INFO
from logging import WARNING as log_WARNING
from logging import basicConfig, critical, debug, error, info, warning
from os import listdir, mkdir
from os import remove as os_remove
from os.path import dirname
from os.path import exists as os_exists
from os.path import isdir, realpath
from shutil import rmtree
from time import localtime, strftime

from cv2 import (COLOR_GRAY2RGB, COLOR_RGB2GRAY, IMWRITE_JPEG_QUALITY,
                 TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, cvtColor,
                 destroyAllWindows)
from cv2 import imread as cv_imread
from cv2 import imshow as cv_imshow
from cv2 import imwrite as cv_imwrite
from cv2 import undistort, waitKey
from numpy import (around, array_equal, argwhere, concatenate, asarray, float32, log10, mgrid,
                   newaxis, reshape, uint8, zeros)

from CalibratePSEye import *

parser = ArgumentParser(description='Run CalibratePSEye unittests')
parser.add_argument('--loglvl',
                    choices=['CRITICAL','ERROR','WARNING','INFO','DEBUG'],
                    default='CRITICAL', help='set unittest log level')
if __name__ == '__main__':
    args = parser.parse_args()
    level=log_CRITICAL
    if args.loglvl == 'ERROR':
        level=log_ERROR
    elif args.loglvl == 'WARNING':
        level=log_WARNING
    elif args.loglvl == 'INFO':
        level=log_INFO
    elif args.loglvl == 'DEBUG':
        level=log_DEBUG
    basicConfig(level=level, format='\t%(levelname)s:%(funcName)s:%(message)s')
    fdir = realpath(dirname(__file__))
    testdir = realpath(fdir + '/tests')
    calibsdir = testdir + '/calibs_00000000-000000'
    if not os_exists(fdir + '/data'):
        mkdir(fdir + '/data')
        warning('data/ directory did not exists; created it')
    # params for chessboard on marand's HP monitor
    #   last updated: 2020-05-22
    #  points of corners on chessboard
    boardsize = (3, 4)
    sq_len = 63  # mm
    o = zeros((boardsize[0]*boardsize[1],3), float32)
    o[:,:2] = mgrid[0:boardsize[0], 0:boardsize[1]].T.reshape(-1,2) * sq_len
    # chessboard detection rules
    p = {
        'boardsize': boardsize,
        'winsize': (11, 11),
        'zerozone': (-1, -1),
        'criteria': (
            TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER, 30, 0.001
        )
    }
    cpdict = {}  # for testing calibration loading later

    def test_report( func=None ):
        """
        Use as decorator for a unit test to run it & prepare a report. Call
        without any arguments to generate an overall test report.

        :param func: Unit test, passed in like to a decorator, defaults to None
        :type func: function, optional

        When func is None, reports how many tests passed and how many tests
        failed.
        """
        # set static vars
        if not hasattr(test_report, 'p'):
            test_report.p = 0
        if not hasattr(test_report, 'f'):
            test_report.f = 0

        # color-code messages for ease of reading
        PASS_C = '\033[92m'
        FAIL_C = '\033[91m'
        REPORT_C = '\033[93m'
        END_C = '\033[0m'

        # overall unit testing report
        if func is None:
            msg = REPORT_C + '\nREPORT: %d/%d tests passed\n' % (
                test_report.p, test_report.p+test_report.f
            ) + END_C

        # running unit test
        else:
            try:
                func()
            except Exception as e:
                # 46 - longest testname and pass label
                # 4  - tabsize
                max_msg_len = 46 + 4
                msg = FAIL_C + 'FAIL:\t' + func.__name__ + ':' + END_C
                spacer = max_msg_len - len(msg)
                msg = msg + ''.ljust(spacer)
                msg = msg + type(e).__name__
                if len(e.args) != 0:
                    msg = msg + ': ' + e.args[0]

                msg = msg
                test_report.f += 1
            else:
                msg = PASS_C + 'PASS:\t' + func.__name__ + END_C
                test_report.p += 1
            finally:
                # reset parameters regardless
                global p, o, boardsize
                boardsize = (3, 4)
                sq_len = 63  # mm
                o = zeros((boardsize[0]*boardsize[1],3), float32)
                o[:,:2] = mgrid[0:boardsize[0], 0:boardsize[1]].T.reshape(-1,2) * sq_len
                p = {
                    'boardsize': boardsize,
                    'winsize': (11, 11),
                    'zerozone': (-1, -1),
                    'criteria': (
                        TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER,30,0.001
                    )
                }
        print(msg)

    def test_clear(calib):
        """
        Asserts a bunch of things that should have been cleared on
        CalibratePSEye instantiation. This code is re-used so much it gets its
        own function.
        """
        for k in ('calibpath', 'img_arr', 'objp',
                  'cameraMatrix', 'distCoeffs'):
            if getattr(calib, k) is not None:
                raise RuntimeError('\'calib.%s\' was not reset' % k)
            debug('\'calib.%s\' was reset' % k)

        for k in ('objpoints', 'corners_arr'):
            if len(getattr(calib, k)) != 0:
                raise RuntimeError('\'calib.%s\' was not reset' % k)
            debug('\'calib.%s\' was reset' % k)
# ============================================================================
# Intialization Methods
#   these also test `clear`
# ============================================================================

    @test_report
    def test_init_empty():
        """
        Test empty constructor.
        """
        calib = CalibratePSEye()
        test_clear(calib)

    @test_report
    def test_init_chessboard_obj():
        """
        Test chessboard initialization with Python objects
        """
        calib = CalibratePSEye()
        calib.init_chessboard(p, o)
        try:
            if calib.calibpath is None:
                raise RuntimeError('_calib_path wasn\'t created')
            for k in p.keys():
                if not getattr(calib, k) == p[k]:
                    raise RuntimeError('\'%s\' param was loaded incorrectly' % k)
                debug('\'%s\' param matched' % k)
            if not array_equal(calib.objp, o):
                raise RuntimeError('Failed to load objp correctly')
        finally:
            calib.removepath()

    @test_report
    def test_init_chessboard_str():
        """
        Test initializing based off of filepaths.
        """
        pstr = calibsdir + '/processing_params.csv'
        ostr = calibsdir + '/objp.csv'
        if not os_exists(pstr) or not os_exists(ostr):
            raise RuntimeError('Bad test files \'%s\' and \'%s\'' % (pstr, ostr))
        # Read String Files
        with open(ostr, 'r') as f:
            text = f.read()
            data = text.split(', ')
            shape = [int(v) for v in data[0].replace('\"', '').split('x')]
            objp = reshape([int(v) for v in data[1:]], shape).astype('float32')
        with open(pstr, 'r') as f:
            lines = f.read().splitlines()
            param_text = [line.split(', ') for line in lines]
            params = dict()
        for p in param_text:
            name = p[0].replace('\"', '')
            if name not in ('boardsize', 'zerozone', 'winsize', 'criteria'):
                continue
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
        # Test
        calib = CalibratePSEye()
        calib.init_chessboard(pstr, ostr)
        try:
            if calib.calibpath is None:
                raise RuntimeError('_calib_path wasn\'t created')
            for k in params.keys():
                if not getattr(calib, k) == params[k]:
                    raise RuntimeError('\'%s\' param was loaded incorrectly' % k)
                debug('\'%s\' param matched' % k)
            if not array_equal(calib.objp, objp):
                raise RuntimeError('Failed to load objp correctly')
        finally:
            calib.removepath()
# ============================================================================
# Private Methods
# ============================================================================

    @test_report
    def test_create_calib_path():
        """
        Test creation of calibration storage path.

        EXCEPTIONS
            RuntimeError if failed to remove calibration directory
        """
        calib = CalibratePSEye()
        calib.init_chessboard(p, o)
        if not isdir(calib.calibpath):  # ensure creation
            raise RuntimeError('Failed to create _calib_path')
        time_str = strftime('%Y%m%d-%H%M%S', localtime())  # test timestamp
        if not time_str[:-2] == calib.calibpath[-15:-2]:
            raise RuntimeError('_calib_path timestamp doesn\'t match to current minute')
        calib.removepath()

    @test_report
    def test_get_timestamp():
        """
        Test retrieval of calib path timestamp

        EXCEPTIONS
            RuntimeError if failed to remove calibration directory
        """
        calib = CalibratePSEye()
        calib.init_chessboard(p, o)
        time_str = strftime('%Y%m%d-%H%M%S', localtime())  # test timestamp
        if not calib.calibpath[-15:] == calib.get_timestamp():
            raise RuntimeError('Error getting timestamp')
        if not time_str[:-2] == calib.get_timestamp()[:-2]:
            raise RuntimeError(
                'Retrieved timestamp doesn\'t match to minute - check `test_create_calib_path`'
            )
        calib.removepath()

    @test_report
    def test_verify_calib_params():
        """
        Test verify_calib_params asserts

        For int 2-tuples (boardsize, winsize, zerozone):
            Tests too many and then too few elements to ensure equality check
            Exact fit case was tested in the init testers above
            Tests catching of floats

        objp
            `assert self.objp.shape[0] == self.boardsize[0] * self.boardsize[1]`
            `assert self.objp.shape[1] == 3`
            Same testing rules as above
            Also checks for non-float32 types
        """
        global p, o
        # Test 2-tuples
        for k in ('boardsize', 'winsize', 'zerozone'):
            p[k] = (-1,-1,-1)
            try:
                calib = CalibratePSEye()
                calib.init_chessboard(p, o)
            except TypeError:
                pass
            else:
                raise RuntimeError('Failed to catch len(%s) too large' % k)
            finally:
                del calib

            p[k] = (-1,)
            try:
                calib = CalibratePSEye()
                calib.init_chessboard(p, o)
            except TypeError:
                pass
            else:
                raise RuntimeError('Failed to catch len(%s) too large' % k)
            finally:
                del calib

            p[k] = (-1.1,-1)
            try:
                calib = CalibratePSEye()
                calib.init_chessboard(p, o)
            except TypeError:
                pass
            else:
                raise RuntimeError('Failed to catch %s[0] non-int' % k)
            finally:
                del calib

            p[k] = (-1,-1.1)
            try:
                calib = CalibratePSEye()
                calib.init_chessboard(p, o)
            except TypeError:
                pass
            else:
                raise RuntimeError('Failed to catch %s[1] non-int' % k)
            finally:
                del calib

            p = {
                'boardsize': boardsize,
                'winsize': (11, 11),
                'zerozone': (-1, -1),
                'criteria': (
                    TERM_CRITERIA_EPS+TERM_CRITERIA_MAX_ITER,30,0.001
                )
            }

        p['criteria'] = (-1.1, -1, 5, 18)
        try:
            calib = CalibratePSEye()
            calib.init_chessboard(p, o)
        except TypeError:
            pass
        else:
            raise RuntimeError('Failed to catch len(criteria) too large')

        p['criteria'] = (-1,)
        try:
            calib = CalibratePSEye()
            calib.init_chessboard(p, o)
        except TypeError:
            pass
        else:
            raise RuntimeError('Failed to catch len(criteria) too small')

        # check float32 case (should pass)
        p['criteria'] = (3, 30, 0.001)
        calib = CalibratePSEye()
        calib.init_chessboard(p, o)
        calib.removepath()

        # check non float32 case (should fail)
        for t in ('uint8', 'uint16', 'uint32', 'uint64', 'int16', 'int32',
                  'int64', 'float64', 'object'):
            objp = o.astype(t)
            try:
                CalibratePSEye(p, objp)
            except TypeError:
                pass
            else:
                raise RuntimeError('Failed to catch objp non-float32')

        objp = o.copy()[...,newaxis]
        try:
            CalibratePSEye(p,objp)
        except TypeError:
            pass
        else:
            raise RuntimeError('Failed to catch objp too many dimensions')

        objp = zeros((boardsize[0]*boardsize[1]+1, 3))
        try:
            CalibratePSEye(p,objp)
        except TypeError:
            pass
        else:
            raise RuntimeError('Failed to catach objp[0] bad size')

        objp = o[:, :2].copy()
        try:
            CalibratePSEye(p, objp)
        except TypeError:
            pass
        else:
            raise RuntimeError('Failed to catach objp[1] too small')

        objp = zeros((o.shape[0], o.shape[1]+1))
        try:
            CalibratePSEye(p, objp)
        except TypeError:
            pass
        else:
            raise RuntimeError('Failed to catach objp[1] too large')

    @test_report
    def test_save_calib_params():
        """
        Test that calibration params are being saved correctly, by opening
        file and comparing results.

        Loading params is tested in `test_init_str`.
        """
        calib = CalibratePSEye()
        calib.init_chessboard(p, o)
        try:
            pstr = calib.calibpath + '/processing_params.csv'
            ostr = calib.calibpath + '/objp.csv'
            calib._load_processing_params(pstr)
            calib._load_objp(ostr)
            for k in p.keys():
                if getattr(calib, k) != p[k]:
                    raise RuntimeError('\'%s\' does not match' % k)
            debug('\'%s\' matched' % k)
            if not array_equal(o, calib.objp):
                raise RuntimeError('objp does not match')
        finally:
            calib.removepath()
# ============================================================================
# Calibration Image Recording/Loading Methods
# ============================================================================

    @test_report
    def test_load_calib_imgs_asserts():
        """
        Test load_calibs_imgs asserts.
        Assert testing
            Tests elements that are supposed to pass b/c as long as
            `init_chessboard` hasn't been called, a Runtime (not Assertion)
            error will be raised.
        """
        calib = CalibratePSEye()  # do NOT init_chessboard yet
        for t in (int, float, complex, list, tuple, range, dict, set,
                  frozenset, bool, bytes, bytearray, memoryview):
            try:
                calib.load_calib_imgs(t)
            except TypeError:
                debug('\'%s\' calibrations' % t.__name__)
            else:
                raise RuntimeError('Failed to catch %s calibrations' % t.__name__)

        try:
            calib.load_calib_imgs('asdf', 1.5)
        except TypeError:
            pass
        else:
            raise RuntimeError('Failed to catch bad clean param')

        # check for elements that do pass
        #   this works b/c it fails another test w/RuntimeError (not Assertion)
        #   in the next line
        for elem in (True, False, 0, 1):
            try:
                calib.load_calib_imgs('asdf', elem)
            except RuntimeError:
                pass
            else:
                raise RuntimeError('Failed to catch bad img_path')

    @test_report
    def test_load_calib_imgs_paths():
        """
        Test load_calib_imgs path creation/checking.

        All `calib_imgs_paths` tests are basically the same
        """
        global p, o
        if not os_exists(testdir+'/raw'):
            raise RuntimeError('test \'raw\' directory could not be found')
        # Setup
        calib = CalibratePSEye()
        calib.init_chessboard(p, o)
        try:
            calib.load_calib_imgs(testdir+'/raw')

            # Make sure everything was created properly
            for p in ('/raw', '/corners'):
                if not isdir(calib.calibpath + p):
                    raise RuntimeError('path \'%s\' wasn\'t created')

            # Make sure raw images were copied correctly
            for fn in listdir(testdir + '/raw'):
                f1 = calib.calibpath + '/raw/' + fn
                f2 = testdir + '/raw/' + fn
                g1 = cv_imread(f1)
                i2 = cv_imread(f2)
                g2 = cvtColor(cvtColor(i2, COLOR_RGB2GRAY), COLOR_GRAY2RGB)
                if not array_equal(g1, g2):
                    raise RuntimeError('frame \'%s\' did not match' % fn)
                debug('\'%s\' matched' % fn)
        finally:
            calib.removepath()

    @test_report
    def test_record_calib_imgs_asserts():
        """
        Test record_calibs_imgs asserts. Basically just for initialization
        """
        calib = CalibratePSEye()

        # calib.calibpath is None
        try:
            calib.record_calib_imgs()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Failed to catch absence of _calib_path')

        # calib.img_arr is not None
        calib.img_arr = (1,2,3,4)
        try:
            calib.record_calib_imgs()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Failed to catch not-None img_arr')

        calib.clear()
        for t in (str, float, complex, list, tuple, range, dict, set,
                  frozenset, bool, bytes, bytearray, memoryview):
            try:
                calib.record_calib_imgs(nframes=t, countdown=15)
            except TypeError:
                pass
            else:
                raise RuntimeError('Failed to catch %s nframes' % t.__name__)

        calib.clear()
        for t in (str, float, complex, list, tuple, range, dict, set,
                  frozenset, bool, bytes, bytearray, memoryview):
            try:
                calib.record_calib_imgs(nframes=15, countdown=t)
            except TypeError:
                pass
            else:
                raise RuntimeError('Failed to catch %s countdown' % t.__name__)

    @test_report
    def test_record_calib_imgs_member_data():
        """
        Test member data assignment inside of `record_calib_imgs`

        `record_calib_imgs` is called with a negative countdown to force
        immediate chessboard logging. The test calibration images provided
        should all have valid chessboards for the provided params.
        """
        # Setup
        h, w, _ = cv_imread(testdir+'/raw/f00001.jpg').shape
        nf = len([
            f for f in listdir(testdir+'/raw') if f[-4:].lower() == '.jpg'
        ])

        # Tests
        calib = CalibratePSEye()
        calib.init_chessboard(p, o)
        try:
            calib.record_calib_imgs(
                cam=testdir+'/raw/f%05d.jpg', nframes=nf, countdown=-1
            )
            if calib.w != w:
                raise RuntimeError('\'w\' wasn\'t set properly')
            if calib.h != h:
                raise RuntimeError('\'h\' wasn\'t set properly')
            if calib.img_arr.shape != (h,w,1,nf):
                raise RuntimeError('\'img_arr.shape\' wasn\'t set properly')
            if calib.img_arr.dtype != uint8:
                raise RuntimeError('\'img_arr.dtype\' wasn\'t set properly')
        finally:
            calib.removepath()

    @test_report
    def test_record_calib_imgs_paths():
        """
        Test record_calib_imgs path creation/checking.

        `record_calib_imgs` is called with a negative countdown to force
        immediate chessboard logging. The test calibration images provided
        should all have valid chessboards for the provided params.
        """
        global p, o
        nf = len([
            f for f in listdir(testdir+'/raw') if f[-4:].lower() == '.jpg'
        ])
        calib = CalibratePSEye()
        calib.init_chessboard(p, o)
        try:
            calib.record_calib_imgs(
                cam=testdir+'/raw/f%05d.jpg', nframes=nf, countdown=-1
            )

            # Make sure everything was created properly
            for p in ('/raw', '/corners'):
                if not isdir(calib.calibpath + p):
                    raise RuntimeError('\'%s\' wasn\'t created')

            # Make sure raw images were handled correctly
            #   never actually checks for frame equality - keep running into
            #   jpgs quality loss issues that honestly weren't worth the time
            #   I spent trying to fix them
            fns1 = sorted(listdir(testdir + '/raw'))
            fns2 = sorted(listdir(calib.calibpath + '/raw'))
            if len(fns1) != len(fns2):
                raise RuntimeError('Failed to save all valid calibration images')

        finally:
            calib.removepath()
# ============================================================================
# Calibration Methods
# ============================================================================

    @test_report
    def test_load_calibrations_asserts():
        """
        Test load calibrations assertions. This is separate from loading with
        str or dict input to avoid duplicate code and unclear asymmetry
        """
        calib = CalibratePSEye()
        for t in (int, float, complex, list, tuple, range, dict, set,
                  frozenset, bool, bytes, bytearray, memoryview):
            try:
                calib.load_calibrations(t)
            except TypeError:
                debug('Caught \'%s\' calibrations' % t.__name__)
            else:
                raise RuntimeError('Failed to catch %s calibrations' % t.__name__)

    @test_report
    def test_load_calibrations_csv():
        """
        Test load_calibrations parsing CSV.

        WARNING
            Call `test_load_calibrations_str` before
            `test_load_calibrations_dict` and before
            `test_compute_calibrations`so that the dict can be loaded
            rather than manually inputted.

        CSV FORMAT
            - delimiter: ', '
            - no floats, round and convert to int using known precision
            - strings in double quotes

            e.g. "varname", var[0], var[1], ...\\n
        """
        testcalibpath = calibsdir + '/camera_params.csv'
        if not os_exists(testcalibpath):
            raise RuntimeError('Can\'t find \'%s\'' % testcalibpath)
        global cpdict  # global for use in other functions
        cpdict = {}

        # Load params from test
        with open(testcalibpath, 'r') as f:
            lines = f.read().splitlines()
        entries = [line.split(', ') for line in lines]
        for c in entries:
            name = c[0].replace('\"', '')
            if name not in ('cameraMatrix', 'distCoeffs', 'w', 'h'):
                warning(
                    'variable name: \'%s\' not valid calib name' % name
                )
                continue
            if name in ('w', 'h'):
                cpdict[name] = int(c[1])
                continue

            shape = [int(v) for v in c[1].replace('\"', '').split('x')]
            data = asarray([int(v) for v in c[2:]]) / 10**4
            if name == 'cameraMatrix':
                cpdict[name] = reshape(data, shape).astype('float64')
            elif name == 'distCoeffs':
                cpdict[name] = reshape(data, shape).astype('float64')
            else:
                raise RuntimeError('Unreachable state!?')

        # Test calib loading
        calib = CalibratePSEye()
        calib.load_calibrations(calibsdir + '/camera_params.csv')
        for k in cpdict.keys():
            if not array_equal(getattr(calib, k), cpdict[k]):
                raise RuntimeError('\'%s\' did not match' % k)
            debug('\'%s\' matched' % k)

    @test_report
    def test_load_calibrations_dict():
        """
        Test the loading of a dict of calibration matrices

        CSV FORMAT
            - delimiter: ', '
            - no floats, round and convert to int using known precision
            - strings in double quotes

            e.g. "varname", var[0], var[1], ...\\n
        """
        # raise NotImplementedError
        global cpdict
        calib = CalibratePSEye()
        calib.load_calibrations(cpdict)
        for k in cpdict.keys():
            if not array_equal(getattr(calib, k), cpdict[k]):
                raise RuntimeError('\'%s\' did not match' % k)
            debug('\'%s\' matched' % k)

    @test_report
    def test_save_calibrations():
        """
        Test save_calibrations.

        Loading calibs was already tested above, so can rely on it.

        CSV FORMAT
            - delimiter: ', '
            - no floats, round and convert to int using known precision
            - strings in double quotes

            e.g. "varname", var[0], var[1], ...\\n
        """
        fn_cp = calibsdir + '/camera_params.csv'
        if not os_exists(fn_cp):
            raise RuntimeError('tests/camera_params.csv could not be found')
        # set not to tests/ (`load_calibrations` will do that)
        #   don't want to overwrite test data
        calib1 = CalibratePSEye()
        calib1.init_chessboard(p, o)
        try:
            # Load known calibs and then save
            cp = calib1.calibpath
            debug('%s' % calib1.calibpath)
            calib1.load_calibrations(fn_cp)
            calib1.calibpath = cp
            calib1.save_calibrations()
            if not os_exists(calib1.calibpath):
                raise RuntimeError('Failed to create calib path \'%s\''
                                   % calib1.calib_path)
            # Compare saving
            with open(fn_cp, 'r') as f:
                f1 = f.read()
            with open(cp+'/camera_params.csv', 'r') as f:
                f2 = f.read()
            if f1 != f2:
                raise RuntimeError('Camera parameter csvs did not match')

            # Compare loading
            calib2 = CalibratePSEye()
            calib2.load_calibrations(calib1.calibpath+'/camera_params.csv')
            paramlist = ('cameraMatrix', 'distCoeffs')
            for k in paramlist:
                k1 = getattr(calib1, k)
                k2 = getattr(calib2, k)
                if not array_equal(k1, k2):
                    raise RuntimeError(
                        'param \'%s\' does not match between calib1 and calib2' % k
                    )
                debug('\'%s\' matched' % k)
        finally:
            calib1.removepath()

    @test_report
    def test_compute_calibrations():
        """
        Test computation to the 4th decimal place on known calibration data.
        """
        # Test Asserts
        calib = CalibratePSEye()
        calib.corners_arr = [1, 2]
        calib.objpoints = [1, 2]
        calib.w = 320
        calib.h = 240

        calib.corners_arr = []
        try:
            calib.compute_calibrations()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Failed to catch len(corners_arr)==0')

        calib.corners_arr = [1, 2]
        calib.objpoints = []
        try:
            calib.compute_calibrations()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Failed to catch len(objpoints)==0')

        calib.objpoints = [1, 2]
        calib.w = None
        try:
            calib.compute_calibrations()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Failed to catch _w is None')

        calib.h = None
        calib.w = 320
        try:
            calib.compute_calibrations()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Failed to catch h is None')

        # Test Math
        global cpdict
        imgpath = testdir + '/raw'
        if not os_exists(imgpath):
            raise RuntimeError('Could not find imgpath')
        calib = CalibratePSEye()
        calib.init_chessboard(p, o)
        calib.load_calib_imgs(imgpath)
        try:
            calib.compute_calibrations()
            calib.save_calibrations()
            for k in cpdict.keys():
                k1 = cpdict[k]  # already rounded b/c loaded from file
                k2 = around(getattr(calib, k), decimals=4)
                if not array_equal(k1, k2):
                    raise RuntimeError('\'%s\' did not match' % k)
                debug('\'%s\' matched' % k)
                # print(getattr(calib, k))
        finally:
            calib.removepath()
# ============================================================================
# Correction Methods
# ============================================================================

    @test_report
    def test_internal_correct():
        """
        Test internal correction method.

        Discovered this song while debugging this test case on 2020-06-19
        https://open.spotify.com/track/5XgYWQKEqSqA5vXJmwZa6n?si=HvcZD32-T2KRspTPcb4uGQ
        """
        calib = CalibratePSEye()
        calib.load_calibrations(calibsdir + '/camera_params.csv')

        # test datatype check
        try:
            for t in ('uint16', 'uint32', 'uint64', 'int16', 'int32', 'int64',
                      'float32', 'float64', 'object'):
                frames = zeros((240, 320, 3), dtype=t)
                calib.correct(frames)
        except TypeError:
            pass
        else:
            raise RuntimeError('Failed to catch not-uint8 dtype')

        # test size check
        try:
            frames = zeros((240,320), dtype='uint8')
            calib.correct(frames)
        except TypeError:
            pass
        else:
            raise RuntimeError('Failed to catch frames too few dimensions')
        try:
            frames = zeros((240,320,1,1,1), dtype='uint8')
            calib.correct(frames)
        except TypeError:
            pass
        else:
            raise RuntimeError('Failed to catch frames too many dimensions')

        # setup test
        frames = []
        for f in listdir(testdir+'/raw'):
            if f[-4:].lower() == '.jpg':
                frames.append(cv_imread(testdir + '/raw/' + f))
        fshape = [len(frames)] + list(frames[0].shape)
        u1 = zeros(fshape, dtype='uint8')
        for i in range(len(frames)):
            f = frames[i].copy()
            u1[i,...] = undistort(f, calib.cameraMatrix, calib.distCoeffs, None)

        # test single frame
        u2 = calib.correct(frames[0])
        if not array_equal(u1[0,...], u2):
            raise RuntimeError('Single-frame undistort/remap is incorrect')

        # test several frames
        u2 = calib.correct(asarray(frames, dtype='uint8'))
        if not array_equal(u1, u2):
            raise RuntimeError('Multi-frame ndarray undistort/remap is incorrect')

        u2 = calib.correct(frames)
        if not array_equal(u1, u2):
            raise RuntimeError('Multi-frame list undistort/remap is incorrect')

    @test_report
    def test_internal_correct_and_save():
        """
        Test internal correction saving method.
        """
        calib = CalibratePSEye()
        fn_c = calibsdir + '/camera_params.csv'
        # Asserts
        for t in (int, float, complex, list, tuple, range, dict, set,
                  frozenset, bool, bytes, bytearray, memoryview):
                try:
                    calib.correct_and_save(t)
                except TypeError:
                    pass
                else:
                    raise RuntimeError('Failed to catch %s imgpath' % t.__name__)
        calib.load_calibrations(fn_c)
        cp = calib.calibpath
        calib.calibpath = None
        try:
            calib.correct_and_save('file-that-does-not-exist')
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Failed to catch _calib_path is None')

        # Saving
        calib.calibpath = cp
        imgpath = testdir + '/raw'
        storeddir = testdir + '/00000000-000000_undistorted'
        storedcp = testdir + '/00000000-000000_camera_params.csv'
        if os_exists(storeddir):
            rmtree(storeddir)
        if os_exists(storedcp):
            os_remove(storedcp)
        ud1 = calib.correct_and_save(imgpath)
        try:
            # Proper saving
            if not os_exists(storeddir) or not os_exists(storedcp):
                raise RuntimeError('Error creating corrected directories')
            imgcount1 = len([f for f in listdir(imgpath) if f[-4:].lower() == '.jpg'])
            imgcount2 = len([f for f in listdir(storeddir) if f[-4:].lower() == '.jpg'])
            if imgcount1 != imgcount2:
                raise RuntimeError('Not all images were saved')

            # Correct calibration
            #   Check pre-save equality
            imglist = [f for f in listdir(imgpath) if f[-4:].lower() == '.jpg']
            rawimg = [cv_imread(imgpath + '/' + f) for f in imglist]
            ud2 = calib.correct(rawimg)  # will know if `correct` works
            if not array_equal(ud1, ud2):
                raise RuntimeError('Failed pre-save equality check')

            #   Check post-save equality
            for i in range(len(imglist)):
                fnud = storeddir + ('/_f%s' % str(i+1).zfill(5)) + '.jpg'
                cv_imwrite(fnud, ud2[i,...], (IMWRITE_JPEG_QUALITY, 100))
            ud1list = [cv_imread(storeddir + '/' + f) for f in imglist]
            ud2list = [cv_imread(storeddir + '/_' + f) for f in imglist]

            ud1reload = asarray(ud1list, dtype='uint8')
            ud2reload = asarray(ud2list, dtype='uint8')
            if not array_equal(ud1reload, ud2reload):
                raise RuntimeError('Failed reload equality check')
        finally:
            os_remove(storedcp)
            rmtree(storeddir)
            try:
                if os_exists(storedcp):
                    raise RuntimeError('failed to deleted cameraParams csv')
                if os_exists(storeddir):
                    raise RuntimeError('failed to remove undistored img dir')
            except AssertionError:
                raise RuntimeError('Exception during test cleanup')
# ============================================================================
# Interface Methods (Non-Class-Methods)
# ============================================================================

    @test_report
    def test_open_camera():
        """
        Tests `open_camera` datatype checks and accessing non-existent file.
        """
        for t in (float, complex, list, tuple, range, dict, set,
                  frozenset, bool, bytes, bytearray, memoryview):
            try:
                open_camera(cam=t)
            except TypeError:
                debug('Caught %s camera input' % t.__name__)
            else:
                raise RuntimeError('Failed to catch %s camera input' % t.__name__)
        try:
            open_camera(w=1.1)
        except TypeError:
            pass
        else:
            raise RuntimeError('Failed to catch non-integer pixel frame width')
        try:
            open_camera(h=1.1)
        except TypeError:
            pass
        else:
            raise RuntimeError('Failed to catch non-integer pixel frame height')
        try:
            open_camera(fps=1.1)
        except TypeError:
            pass
        else:
            raise RuntimeError('Failed to catch non-integer FPS')

        try:
            open_camera(cam='test/file_that_does_not_exist')
        except RuntimeError:
            pass
        else:
            raise RuntimeError('Failed to error on non-existent camera.')

    @test_report
    def test_livetest_asserts():
        """
        Test livetest asserts. This is an interface, so it's not easy to mock
        keyboard strokes.
        """
        for t in (int, float, complex, list, tuple, range, dict, set,
                  frozenset, bool, bytes, bytearray, memoryview):
                try:
                    livetest(t)
                except RuntimeError:
                    debug('Caught %s calibrations' % t.__name__)
                else:
                    raise RuntimeError('Failed to catch %s calibrations' % t.__name__)
# ============================================================================
# All the way at the end for an overall report
# ============================================================================
    test_report()
