#!/bin/bash

# we want this script to fail on any error
set -e

export CUR_LOC=`pwd`
echo "Building warped2 and warped2-models at: " $CUR_LOC

#
# warped2
#

echo "....warped2"
date

git clone https://github.com/wilseypa/warped2

cd warped2
autoreconf -i

#
# configure with openmpi for google-proftools 
#

# let's try clang....built successfully....
#./configure --with-mpi-includedir=/usr/lib/x86_64-linux-gnu/openmpi/include --prefix=$CUR_LOC/warped2/local CXXFLAGS='-g -O3 -Wno-inconsistent-missing-override -Wno-reserved-id-macro -Wno-keyword-macro -Wno-redundant-move -Wno-pessimizing-move -Wno-infinite-recursion' LDFLAGS='-lprofiler' CXX=clang++
./configure --with-mpi-includedir=/usr/lib/x86_64-linux-gnu/openmpi/include --prefix=$CUR_LOC/warped2/local CXXFLAGS='-g -O3' LDFLAGS='-lprofiler' CXX=clang++

make -j 8 install
date
# return to original subdirectory
cd $CUR_LOC

#
# warped2-models
#

echo "....warped2-models"
date

git clone https://github.com/wilseypa/warped2-models

cd warped2-models
autoreconf -i

#
# configure with openmpi for google-proftools
#

#### for some reason, the -Wno-unused-private-field switch appears to be ignored by the -Werror flag
#### placed after it in the epidemic makefile anyway i just jump into the epidemic subdirectory and
#### remove the -Werror switch for the CXXFLAGS variable, go back to the warped2-models root, type
#### make and everything builds (ok, epidemic still throws a warning, but it builds....

# tell mpicxx to use clang
export OMPI_CXX=clang++
#./configure --with-warped=$CUR_LOC/warped2/local CXXFLAGS='-g -O3 -std=c++11 -Wno-inconsistent-missing-override -Wno-unused-private-field' LDFLAGS='-lprofiler' CXX=mpicxx
./configure --with-warped=$CUR_LOC/warped2/local CXXFLAGS='-g -O3 -std=c++11' LDFLAGS='-lprofiler' CXX=mpicxx

make -j 8
date
# return to original subdirectory
cd $CUR_LOC

