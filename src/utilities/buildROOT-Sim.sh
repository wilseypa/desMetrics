#!/bin/bash

# we want this script to fail on any error
set -e

export CUR_LOC=`pwd`
echo "Building ROOT-sim and  models at: " $CUR_LOC

#
# ROSS
#

echo "....ROOT-sim"
date

git clone http://github.com/HPDCS/ROOT-Sim
cd ROOT-Sim
./autogen.sh

export INSTALL_DIR=`pwd`/root-sim/local
mkdir -p $INSTALL_DIR
./configure --prefix=$INSTALL_DIR
make -j 8 install
export ROOTSIM_CC=$INSTALL_DIR/bin/rootsim-cc
date
# return to original subdirectory
cd $CUR_LOC

#
# now on to building the models
#

echo "....ROOT-Sim models"
date
cd ROOT-Sim/models
for model in `ls`
do
    echo "building" $model
    cd $model
    $ROOTSIM_CC *.c -o $model
    cd ..
done
	     
date
# return to original subdirectory
cd $CUR_LOC
