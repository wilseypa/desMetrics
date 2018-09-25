## Building/using desAnalysis.go

desAnalysis will analyze trace files (or sample trace files) from simulation models.  This analysis
will kick out a bunch of .csv files in the directory ./analysisData from where the tool is run.
These .csv files are then generally processed by the desGraphics.py python program to kick out a
bunch of plots.

Compile:
  go build -o desAnalysis desAnalysis.go

Run:
  ./desAnalysis <file.json>

## desSampler.go

desSampler will extract samples from a simulation trace file.  The output samples will be written to
subdirectories ./sampleDir/startEvent-stopEvent/ (where startEvent-stopEvent will be the line offset
in the original .csv file where the extracted events begin/end).  Each sample will be written as two
desAnalysis processable files (modelSummary.json and desMetrics.csv).  desSampler has many options
that you will have to read about in the source file.

Compile:
  go build -o desSampler desSampler.go

Run:
  ./desSampler [options] <file.json>


## desGraphics.py

desGraphics.py will take the .csv files output from desAnalysis and plot them in various graphs (in
pdf).  In some cases multiple graphs with different parameters (e.g., log/linear y-axis, line graphs
vs scatter plot, etc).  The plots will be written to the subdirectory ./ougputGraphics.

Run:
  python desGraphics.py
