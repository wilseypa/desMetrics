#!/bin/sh
arg1=$1
if [ $arg1 -eq 1 ]; then
	awk '{h[$1]++}; END { for(k in h) print k, h[k]}' streamedData/eventInfo-unsorted.csv > streamedData/lpInfo.csv
	sort -t, -n -k4 streamedData/eventInfo-unsorted.csv > streamedData/eventInfo-sorted.csv
	rm -f streamedData/eventInfo-unsorted.csv
fi
