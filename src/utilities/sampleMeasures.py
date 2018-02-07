
## This program is working to compute various metrics on samples extracted from large event trace
## files for the desMetrics project.  

import os
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp
from scipy.spatial.distance import directed_hausdorff
import matplotlib as mpl
# Force matplotlib to not use any Xwindows backend
mpl.use('Agg')
import palettable
import pylab
import argparse
from operator import itemgetter

# process the arguments on the command line
argparser = argparse.ArgumentParser(description='Generate various measures on how well the desAnalysis output results of desSamples files.')
argparser.add_argument('--fulltrace', 
                       action="store_true",
                       help='Compare samples against full trace.')
argparser.add_argument('--outDir', default='./measuresOutDir/', help='Directory to write output files (default: ./measuresOutDir/)')
argparser.add_argument('--sampleDir', default='./sampleDir/', help='Directory where sample directories are located (default: ./sampleDir/')
argparser.add_argument('--fullTraceDir', default='./', help='Directory where full trace analysis files reside (default: ./')

args = argparser.parse_args()

####----------------------------------------------------------------------------------------------------
#### define a set of functions to do things
####----------------------------------------------------------------------------------------------------

# define a function to display/save the pylab figures.
def display_plot(fileName) :
    print "Creating pdf graphic of: " + fileName
    pylab.savefig(measuresDir + fileName + ".pdf", bbox_inches='tight')
    pylab.clf()
    return

## ok this is my attempt to generate an analysis procedure that will work with either any trace as the base
## case; that said, because the full trace can have startup/teardown costs, we'll include a parameter that
## tells us how many (if any) events to skip at the head/tail of eventsAvailableBySimCycle.csv of the baseDir
def compute_metrics(sampleDirs, skippedEvents):

    ## let's look at events available
    print "working with " + sampleDirs[0] + "/analysisData/eventsAvailableBySimCycle.csv"

    # first we read all of the files 
    baseData = []
    for i in range(len(sampleDirs)) :
        baseData.append(np.loadtxt(sampleDirs[i] + "/analysisData/eventsAvailableBySimCycle.csv",
                                   dtype=np.intc, delimiter = ",", comments="#"))

    if skippedEvents > 0 : baseData[0] = baseData[0][skippedEvents:len(baseData[0])-skippedEvents]
    
    # now plot the data points normalized to an x-axis range of 0-100
    
    # we need to record the shortest sample so we can compute the metrics on equal sized arrays when necessary
    minLength = len(baseData[0])

    colorIndex = 0
    # we will set the alpha value to 1.0 for the base sample and 0.5 for all other samples 
    alphaValue = 1.0

    for index in range(len(baseData)) :
        x_index = np.arange(len(baseData[index])).astype(float)/float(len(baseData[index]))*100
        pylab.plot(x_index, sorted(baseData[index]),
                   color=colors[colorIndex], label=sampleDirs[index], alpha=alphaValue)
        colorIndex = (colorIndex + 1) % len(colors)
        alphaValue=0.5
        if len(baseData[index]) < minLength : minLength = len(baseData[index])

    #pylab.legend(loc='best')
    display_plot('eventsAvailable')

    # now let's compute the various distance metrics of interest
    # most of these require that the compared vectors have the same length; so we will actually take minLength
    # (computed in above loop) samples (equally distributed) from each vector

    metricFile.write("Base sample: " + sampleDirs[0] + "\n")
    metricFile.write("Comparison Sample, Wasserstein Distance, Directed Hausdorff, Kolmogorov-Smirnov\n")

    # let's extract equal lengthsamples from the samples :-)
    sampleExtract = []
    for x in range(len(sampleDirs)) :
        extract = []
        sampleLen = len(baseData[x]) - 1
        for i in range(minLength) :
            extract.append(baseData[x][int(float(sampleLen)/float(minLength) * float(i))])
        sampleExtract.append(extract)

    x_index = np.arange(len(sampleExtract[0]))
    baseSorted = sorted(sampleExtract[0])
    baseSortedTuple = np.vstack((x_index,sorted(sampleExtract[0]))).T
    for x in range(len(sampleDirs)-1) :
        metricFile.write(sampleDirs[x+1])
        metricFile.write(", %.8f" % wasserstein_distance(baseSorted,sorted(sampleExtract[x+1])))
        metricFile.write(", %.8f" % directed_hausdorff(baseSortedTuple,
                                                       np.vstack((x_index,sorted(sampleExtract[x+1]))).T)[0])
        metricFile.write(", %.8f" % ks_2samp(baseSorted,sorted(sampleExtract[x+1]))[0])
        metricFile.write("\n")

    #pylab.legend(loc='best')
    #display_plot('eventsAvailable')

## PHIL: need to fix; what if the base sample doesn't have all the LP names??

    ## ok, now let's look at the percent of events executed by each LP
#    baseData = np.genfromtxt(baseDir + "/analysisData/eventsExecutedByLP.csv", dtype='str', delimiter = ",", comments="#", usecols=(0,3))
#    sampleData = []
#    for x in sampleDirs :
#        sampleData.append(np.genfromtxt(x + "/analysisData/eventsExecutedByLP.csv", dtype='str', delimiter = ",", comments="#", usecols=(0,3)))

    # populate a dictionary with the names in all of the samples and initialize the entries to zero
#    lpNames = {}
#    for name in np.unique(np.append(baseData[:,0],[x[:,0] for x in sampleData])) :
#        lpNames[name] = 0

    # compute the percentatage of all events executed by each LP
#    eventTotal = sum(baseData[:,1].astype(float))
#    baseData = sorted(baseData, key=lambda entry: entry[1])
#    for i in baseData :
#        lpNames[i[0]] = i[1].astype(float)/eventTotal
#    pylab.xlabel('LPs (sorted by by events executed in base data)')
#    pylab.ylabel('Percent of Events Executed')
#    print len(lpNames)
#    print len(baseData)
#    print lpNames[:,1]
#    x_index = np.arange(len(lpNames))
#    pylab.plot(x_index, lpNames[:,1], color='black')
                          
    # plot data from samples
#    display_plot('eventsByLP')
                        

    ## now let's look at the summaries of event chains.

    # since we don't want to have the size of the measured data to skew the wasserstein distance results, 
    # we'll work from percentages of the populations in question.  furthermore, we don't really want to
    # see the differences by event change length; our real interest is in the impact on the overall event
    # chain computation, we will create a vector that includes measures from local, linked, and global
    # chains as one vector to be analyzed....
    
    print "wasserstein against " + sampleDirs[0] + "/analysisData/eventChainsSummary.csv"
    baseData = np.loadtxt(sampleDirs[0] + "/analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter = ",", comments="#")
    percentagesOfBaseData = baseData.astype(float)/float(np.sum(baseData))
    
    print "To Sample, Wasserstein Distance"
    for x in sampleDirs[1:len(sampleDirs)] :
        sampleData = np.loadtxt(x + "/analysisData/eventChainsSummary.csv", dtype=np.intc,
                                delimiter = ",", comments="#")
        percentagesOfSampleData = sampleData.astype(float)/float(np.sum(sampleData))
        print x + ", %.8f" % wasserstein_distance(
            np.append(np.append(np.array(percentagesOfBaseData[:,1]),np.array(percentagesOfBaseData[:,2])),np.array(percentagesOfBaseData[:,3])),
            np.append(np.append(np.array(percentagesOfSampleData[:,1]),np.array(percentagesOfSampleData[:,2])),np.array(percentagesOfSampleData[:,3])))
            
    ## now let's look at modularity....

    return

####----------------------------------------------------------------------------------------------------
#### let us begin
####----------------------------------------------------------------------------------------------------

# create a directory to write output graphs
measuresDir = args.outDir
if not os.path.exists(measuresDir):
    os.makedirs(measuresDir)

## ok so what we're going to do is assemble a list of directories (dirs) where the traces are to be found.  we
## will assume that the desAnalysis trace files are located in a subdirectory called analysisData/ below the
## directories named in this list.  the first element (dirs[0]) of this list will be contain the data for the
## base case to be compared against.

dirs = []

# since we see startup/teardown effects in most of the full traces we've examined to date, we will setup the
# system to automatically trim out the first/last 1% of (only) the eventsAvailableBySimCycle analysis.  if
# we're not using a full trace, nothing will be trimmed.
numSkippedEvents = 0

if args.fulltrace :
    data = np.loadtxt(args.fullTraceDir + "analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", comments="#")
    totalEvents = len(data)
    # how many events in 1%
    numSkippedEvents = int(len(data)*.01)
    dirs.append(args.fullTraceDir)

# setup a function to enable correct sorting of the sample directory names (basically we assume a standard
# naming of these subdirectories of the form produced by desSampler.go; basically:
#    <index of first event from source>-<index of last event from source> 
def left_index(x):
    return int(x.split("-")[0])

for x in sorted(os.listdir(args.sampleDir), key=left_index) :
    dirs.append(args.sampleDir + x)
              
# set colormap and make it the default; prepend the color map with black for plotting the base case
# we should actually tie this to the len(dirs), but that'll have to wait until later.   
colors = [(0.0, 0.0, 0.0)] + palettable.colorbrewer.qualitative.Dark2_7.mpl_colors

# change the plots to a grey background grid w/o solid x/y-axis lines
mpl.style.use('ggplot')

metricFile = open(measuresDir + "/metricFile.csv","w")

compute_metrics(dirs,numSkippedEvents)

metricFile.close()
