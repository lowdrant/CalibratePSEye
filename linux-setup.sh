#!/usr/bin/env bash
#
#  Downloads needed programs and installs them. Copies udev rules and makes
#  sure PATH and directories are setup correctly.
#
# @author lowdrant

set -o nounset pipefail errexit

# copy udev rules
sudo cp 99-psEye.rules /etc/udev/rules.d
sudo udevadm control --reload-rules && sudo udevadm trigger

# python libs
sudo apt install -y python3-opencv python3-numpy python3-yaml

# add pseyepath
fdir="$(dirname $(realpath $0))"
tgtpath="$fdir/CalibratePSEye"
grep "$tgtpath" <<< "$PYTHONPATH"
if [ ! $? -eq 0 ]
then
    echo "INFO: CalibratePSEye is not in your PYTHONPATH. Adding to .profile..."
    echo "PSEYEPATH=$tgtpath" >> "$HOME/.profile"
    echo 'export PYTHONPATH=$PYTHONPATH:$PSEYEPATH' >> "$HOME/.profile" # want '$'
fi

# add data directory
if [ ! -e "$fdir/data" ]
then
    mkdir "$fdir/data"
fi
