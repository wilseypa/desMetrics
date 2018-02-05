
## this program is working to compute various metrics on samples extracted from large event trace
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

# process the arguments on the command line
argparser = argparse.ArgumentParser(description='Generate various measures on how well the desAnalysis output results of desSamples files.')
argparser.add_argument('--fulltrace', 
                       action="store_true",
                       help='Compare samples against full trace.')
args = argparser.parse_args()

# create a directory to write output graphs
measuresDir = 'measuresOutDir/'
if not os.path.exists(measuresDir):
    os.makedirs(measuresDir)

# set colormap and make it the default
colors = palettable.colorbrewer.qualitative.Dark2_7.mpl_colors

# define a function to display/save the pylab figures.
def display_plot(fileName) :
    print "Creating graphics " + fileName
    print "    ....writing pdf"
    pylab.savefig(measuresDir + fileName + ".pdf", bbox_inches='tight')
    pylab.clf()
    return

## ok this is my attempt to generate an analysis procedure that will work with either any trace as the base
## case; that said, because the full trace can have startup/teardown costs, we'll include a parameter that
## tells us how many (if any) events to skip at the head/tail of eventsAvailableBySimCycle.csv of the baseDir
def compute_metrics(baseDir, sampleDirs, skippedEvents):
    ## let's look at events available
    baseData = np.loadtxt(baseDir + "/analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", comments="#")
    totalEvents = len(baseData)
    # is this correct??
    percentagesOfBaseData = baseData.astype(float)/float(np.sum(baseData))

    # while we do this, we're also going to plot the various events available curves normalized to an x-scale
    # of 0-99
    x_index = np.arange(totalEvents).astype(float)/float(totalEvents)*100
    pylab.title('Events available normalized to a range of 0-100')
    pylab.ylabel('Percent of total events')
    pylab.plot(x_index, sorted(percentagesOfBaseData), color=colors[0], label="baseDir")

    print "wasserstein against " + baseDir + "/eventsAvailableBySimCycle.csv"
    print "To Sample, Wasserstein Distance"
    for x in sampleDirs :
        sampleData = np.loadtxt(x + "/analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", comments="#") 
        print x + ", %.8f" % wasserstein_distance(
            sorted(baseData[skippedEvents:totalEvents-skippedEvents]),
            sorted(sampleData))

    display_plot('eventsAvailable')

    ## now let's look at the summaries of event chains.

    # since we don't want to have the size of the measured data to skew the wasserstein distance results, 
    # we'll work from percentages of the populations in question.  furthermore, we don't really want to
    # see the differences by event change length; our real interest is in the impact on the overall event
    # chain computation, we will create a vector that includes measures from local, linked, and global
    # chains as one vector to be analyzed....
    
    print "wasserstein against " + baseDir + "/analysisData/eventChainsSummary.csv"
    baseData = np.loadtxt(baseDir + "/analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter = ",", comments="#")
    percentagesOfBaseData = baseData.astype(float)/float(np.sum(baseData))
    
    print "To Sample, Wasserstein Distance"
    for x in sampleDirs :
        sampleData = np.loadtxt(x + "/analysisData/eventChainsSummary.csv", dtype=np.intc,
                                delimiter = ",", comments="#")
        percentagesOfSampleData = sampleData.astype(float)/float(np.sum(sampleData))
        print x + ", %.8f" % wasserstein_distance(
            np.append(np.append(np.array(percentagesOfBaseData[:,1]),np.array(percentagesOfBaseData[:,2])),np.array(percentagesOfBaseData[:,3])),
            np.append(np.append(np.array(percentagesOfSampleData[:,1]),np.array(percentagesOfSampleData[:,2])),np.array(percentagesOfSampleData[:,3])))
            
    ## now let's look at modularity....

    return

dirs = []

numSkippedEvents = 0
if args.fulltrace :
    data = np.loadtxt("analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", comments="#")
    totalEvents = len(data)
    # how many events in 1%
    numSkippedEvents = int(len(data)*.01)
    dirs.append("./")

for x in sorted(os.listdir("sampleDir")) :
    dirs.append("sampleDir/" + x)
              
#print dirs
#print dirs[0]
#print dirs[1:len(dirs)]

compute_metrics(dirs[0],dirs[1:len(dirs)],numSkippedEvents)
