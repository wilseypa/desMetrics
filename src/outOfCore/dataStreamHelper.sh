#!/bin/sh
arg=$1
# sorts eventInfo by receive time, and makes a file on containing information about each LP
#LP, num of received events, num of sent events
if [ $arg -eq 1 ]; then
	> streamedData/printLog
	awk '{h[$3]++}; END {for(k in h) print k, h[k]}' streamedData/eventInfo-unsorted.csv > streamedData/lpInfo-received.csv
	awk '{h[$1]++}; END {for(k in h) print k, h[k]}' streamedData/eventInfo-unsorted.csv > streamedData/lpInfo-sent.csv
	
	awk 'FNR==NR {h[$1];next}; !($1 in h) {print $1, 0}'  streamedData/lpInfo-received.csv streamedData/eventInfo-unsorted.csv >> streamedData/lpInfo-received.csv
	awk 'FNR==NR {h[$1];next}; !($3 in h) {print $3, 0}'  streamedData/lpInfo-sent.csv streamedData/eventInfo-unsorted.csv >> streamedData/lpInfo-sent.csv
	
	awk 'FNR==NR{a[$1]=$2 FS $3;next}{ print $0, a[$1]}' streamedData/lpInfo-sent.csv streamedData/lpInfo-received.csv > streamedData/lpInfo.csv
	sort -t, -n -k4 streamedData/eventInfo-unsorted.csv > streamedData/eventInfo-sorted.csv
	echo "Verifying that all LPs received at least one event." >> "streamedData/printLog"
	awk '$2 == "0" {{gsub(","," ",$1)} print "WARNING: LP " $1 "received zero messages." >> "streamedData/printLog"}' streamedData/lpInfo-received.csv
	awk 'BEGIN{max=0}{if ($2>max) max=$2} END{print max}' streamedData/lpInfo-received.csv > streamedData/temp
	#rm -f streamedData/eventInfo-unsorted.csv
	#rm -f streamedData/lpInfo-sent.csv 
	#rm -f streamedData/lpInfo-received.csv 
	
# adds number of local and remote events sent to LP file	
# LP, num of received events, num of sent events, num of local, num of remote
elif [ $arg -eq 2 ]; then	
	> streamedData/printLog
	awk '{if($5==0)h[$1]++}; END {for(k in h) print k, h[k]}' streamedData/eventInfo-sorted.csv > streamedData/lpInfo-local.csv
	awk '{if($5==1)h[$1]++}; END {for(k in h) print k, h[k]}' streamedData/eventInfo-sorted.csv > streamedData/lpInfo-remote.csv
	
	awk 'FNR==NR {h[$1];next}; !($1 in h) {print $1, 0} !($3 in h) {print $3, 0}'  streamedData/lpInfo-local.csv streamedData/eventInfo-sorted.csv >> streamedData/lpInfo-local.csv
	awk 'FNR==NR {h[$1];next}; !($1 in h) {print $1, 0} !($3 in h) {print $3, 0}'  streamedData/lpInfo-remote.csv streamedData/eventInfo-sorted.csv >> streamedData/lpInfo-remote.csv
	
	awk 'FNR==NR{a[$1]=$2 FS $3;next}{ print $0 a[$1]}' streamedData/lpInfo-local.csv streamedData/lpInfo.csv > streamedData/temp.csv
	awk 'FNR==NR{a[$1]=$2 FS $3;next}{ print $0 a[$1]}' streamedData/lpInfo-remote.csv streamedData/temp.csv > streamedData/lpInfo.csv
	tr ' ' ',' < streamedData/lpInfo.csv | sed 's/.$//' > streamedData/lpInfo-comma.csv

	#rm -f streamedData/lpInfo-local.csv 
	#rm -f streamedData/lpInfo-remote.csv
	#rm -f streamedData/lpInfo-temp.csv
fi
