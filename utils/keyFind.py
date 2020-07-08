#!/usr/bin/env python3
"""
This software is copyrighted material (C) Marion Anderson 2020

The library is provided under the Apache 2.0 License,
as found at https://www.apache.org/licenses/LICENSE-2.0

    OVERVIEW:

    Determine how OpenCV's `waitKey` interprets keypresses. Inspired
    by trying to read weird keys like 'left arrow'.

    INTERFACE:

    Run the file as is. The last key pressed will be displayed in an OpenCV
    display frame.
"""

from argparse import ArgumentParser
from cv2 import destroyAllWindows, waitKey
from cv2 import imshow as cv_imshow
from cv2 import putText as cv_putText
from numpy import zeros

parser = ArgumentParser(
    description='Determine how OpenCV\'s `waitKey` interprets keypresses'
)
args = parser.parse_args()

u = zeros((240, 320, 3))  # reference frame
u1 = u.copy()             # prealloc mem
lpress = 0                # last press (`waitKey` timeout returns 255)
try:
    while 1:
        u1 = u.copy()  # avoid constant text
        cv_putText(u1, 'q to quit', (5,30), 1,1,(0,0,255), thickness=2)
        cv_putText(u1, 'Last press: %d' % lpress, (5,15), 1,1,(0,0,255), thickness=2)

        cv_imshow('win', u1)
        press = waitKey(20)
        if press == ord('q'):
            break
        if press != 255:  # `waitKey` returns 255 on timeout
            lpress = press
except Exception as e:
    raise e
finally:
    destroyAllWindows()
