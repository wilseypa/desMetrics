#!/bin/sh
arg1=$1
# sorts eventInfo by receive time, and makes a file on containing information about each LP
#LP, num of received events
if [ $arg1 -eq 1 ]; then
	> streamedData/printLog
	awk '{h[$3]++}; END { for(k in h) print k, h[k]}' streamedData/eventInfo-unsorted.csv > streamedData/lpInfo.csv
	awk 'FNR==NR {h[$1];next}; !($1 in h) {print $1, "0"}'  streamedData/lpInfo.csv streamedData/eventInfo-unsorted.csv >> streamedData/lpInfo.csv
	sort -t, -n -k4 streamedData/eventInfo-unsorted.csv > streamedData/eventInfo-sorted.csv
	#rm -f streamedData/eventInfo-unsorted.csv
	echo "Verifying that all LPs received at least one event." >> "streamedData/printLog"
	awk '$2 == "0" {{gsub(","," ",$1)} print "WARNING: LP " $1 "received zero messages." >> "streamedData/printLog"}' streamedData/lpInfo.csv
	sed -i '$ d' analysisData/modelSummary.json
	awk 'BEGIN{max=0}{if ($2>max) max=$2} END{print max}' streamedData/lpInfo.csv > streamedData/temp
fi
