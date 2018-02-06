
## This program is working to compute various metrics on samples extracted from large event trace
## files for the desMetrics project.  

import os
import numpy as np
from scipy.stats import wasserstein_distance
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

# create a directory to write output graphs
measuresDir = args.outDir
if not os.path.exists(measuresDir):
    os.makedirs(measuresDir)

# set colormap and make it the default
colors = palettable.colorbrewer.qualitative.Dark2_7.mpl_colors

# change the plots to a grey background grid w/o solid x/y-axis lines
mpl.style.use('ggplot')

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
    baseData = np.loadtxt(sampleDirs[0] + "/analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", comments="#")
    totalEvents = len(baseData)

    # while we do this, we're also going to plot the various events available curves normalized to an x-scale
    # of 0-99
    x_index = np.arange(totalEvents).astype(float)/float(totalEvents)*100
    pylab.xlabel('Simulation cycle (normalized to a range of 0-100)')
    pylab.ylabel('Events Available for Execution')
    pylab.plot(x_index, sorted(baseData), color="black", label=sampleDirs[0])

    print "wasserstein against " + sampleDirs[0] + "/analysisData/eventsAvailableBySimCycle.csv"
    print "To Sample, Wasserstein Distance"
    colorIndex = 0
    for x in sampleDirs[1:len(sampleDirs)] :
        sampleData = np.loadtxt(x + "/analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", comments="#") 
        print x + ", %.8f" % wasserstein_distance(
            sorted(baseData[skippedEvents:totalEvents-skippedEvents]),
            sorted(sampleData))
        x_index = np.arange(len(sampleData)).astype(float)/float(len(sampleData))*100
        pylab.plot(x_index, sorted(sampleData), color=colors[colorIndex], label=x, alpha=0.5)
        colorIndex = colorIndex + 1

    #pylab.legend(loc='best')
    display_plot('eventsAvailable')

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

## ok so what we're going to do is assemble a list of directories where the traces are to be found. we will
## assume that the desAnalysis trace files are located in a subdirectory called analysisData/ below the
## directories named in this list.  the first element of this list will be contain the data for the base case
## to be compared against.

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
              
#print dirs
#print dirs[0]
#print dirs[1:len(dirs)]

compute_metrics(dirs,numSkippedEvents)
