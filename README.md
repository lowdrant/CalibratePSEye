# CalibratePSEye
Robust (some might say ornery) interface for camera chessboard calibration, JPEG image undistortion, and calibration image and parameter saving. So named because it was developed with a [Playstation Eye](https://en.wikipedia.org/wiki/PlayStation_Eye) camera. Currently only works on UNIX systems, due to my laziness about filepath separators.

## Requirements
* Be running Linux (for filepath navigation)
* Python3
* OpenCV
* NumPy

## Using this Repository

### Setup
[linux-setup.sh](linux-setup.sh) handles installing all dependencies and adding this module to PYTHONPATH.

#### Explanation of setup.sh
1. Installs the udev rule [99-psEye.rules](99-psEye.rules). This maps a PSEye camera to a consistent file, `/dev/psEye`
2. Installs Python library dependencies
3. Adds CalibratePSEye.py to PYTHONPATH via `~/.profile`
4. Creates `data/` directory in this repository

### Camera Calibration
[CalibratePSEye/CalibratePSEye.py](CalibratePSEye/CalibratePSEye.py) provides calibration utilities to calibrate using a chessboard pattern per the [OpenCV tutorial](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html). The class is deliberately ornery about overwriting data, so every time calibration process is started (computing of parameters, loading of calibration images), it creates a new time-stamped directory with all calibration data saved into it. It provides support for recording/loading/sanitizing calibration images, computing/loading/saving camera calibration parameters, and undistoring/remapping images. Calibration images are automatically saved every time to a directory of the form `basepath/calibration_yyyymmdd-HHMMSS`.

`linux-setup.sh` adds the module to your PYTHONPATH. You can ensure you have a working interface by running [CalibratePSEye/runtests.py](CalibratePSEye/runtests.py)

## Some Design Notes

1. This library saves, and only pays attention to, JPG images. This is to match UofM data archival standards. If you find that you want to change that to '.png', or some other, decidedly better image format, go right on ahead. I wish I could.

2. The `CalibratePSEye` class requires the specification of a basepath to store calibration directories. If you do not specify `basepath` on class initialization, it will create `data/` in this repository.

3. This is an interface for *chessboard* calibration.


## Data Storage
This library is deliberately ornery about timestamping data and storing it, so that the calibration process can be identically recomputed years in the future. As there are many parameters associated with camera calibration, they are grouped by timestamped directory inside of a larger base directory, with the form `basepath/calibration_yyyymmdd-HHMMSS/`. See [CalibratePSEye.py](CalibratePSEye.py) for additional details.

Data storage follows the University of Michigan's [Deep Blue Preservation and Format Support Policy](https://deepblue.lib.umich.edu/static/about/deepbluepreservation.html) for Level 1 preservation support. Specifically, that means video recordings are stored as JPEG frames, and all calibration data is stored in CSV files.

CSV files follow this format:
* Delimiter: comma followed by a space (', ')
* Strings in double quotes: '\\"hello!\\", '
* No floating point numbers. All floats should be scaled by the appropriate power of 10 precision and then truncated.

This ensures the .csvgz storage format has the roughly same size as the binary file equivalent.

## A Recommendation
If you are running a Linux distro, it might be worth setting up a udev rule for your camera. I have included [99-psEye.rules](99-psEye.rules), which I use to interface with my PlayStation Eye camera. A useful tutorial can be found [here](https://www.thegeekdiary.com/beginners-guide-to-udev-in-linux/).

## Useful Resources
* [this](https://docs.opencv.org/3.2.0/dc/dbb/tutorial_py_calibration.html) camera calibration tutorial
* "Robotics, Vision, and Control," 2nd Ed by Peter Corke, ISBN-13: 978-3319544120

## Acknowledgments
I developed this in the course of my Ph.D. research in the [Biologically Inspired Robotics and Dynamical Systems (BIRDS) Lab](https://birds.eecs.umich.edu/index.html) at the University of Michigan, advised by Dr. Shai Revzen, and with funding from ARO MURI W911NF-17-1-0306.

## License
This software is copyrighted material (C) [Marion Anderson](https://github.com/lowdrant) 2020

This library is provided under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
[NOTICE](NOTICE) must be included with all distributions of this library, with the singular exception of only taking the udev rule, which is far too simple to rightfully claim as any sort of intellectual property.
