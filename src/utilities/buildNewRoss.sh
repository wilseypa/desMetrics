#!/bin/bash

# we want this script to fail on any error
set -e

export CUR_LOC=`pwd`
echo "Building New ROSS and CODES models at: " $CUR_LOC

#
# ROSS
#

echo "....ROSS"
date

git clone http://github.com/carothersc/ROSS.git
cd ROSS
mkdir build
cd build
export ROSS_BUILD=`pwd`
ARCH=x86_64 CC=mpicc CXX=mpicxx cmake -DCMAKE_INSTALL_PREFIX=../install ../
export ROSS_LIB=`pwd`/../install/lib/pkgconfig/
make -j 8
make install
date
# return to original subdirectory
cd $CUR_LOC

#
# CODES
#

# ok, codes won't build with 'set -e' so we'll turn it off
set +e

echo "....CODES"
date

git clone https://github.com/codes-org/codes
cd codes/
./prepare.sh 
mkdir build
cd build
../configure --prefix=$ROSS_BUILD CC=mpicc CXX=mpicxx PKG_CONFIG_PATH=$ROSS_LIB
#../configure --prefix=/home/paw/wk/new-ross/ROSS/build/codes/build CC=mpicc CXX=mpicxx PKG_CONFIG_PATH=/home/paw/wk/new-ross/ROSS/install/lib/pkgconfig/
make -j 8
make install
date
# return to original subdirectory
cd $CUR_LOC
