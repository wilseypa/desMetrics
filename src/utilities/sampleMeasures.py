
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
from palettable.colorbrewer.qualitative import Set1_9
import pylab
import argparse
import collections
import networkx as nx
import community # install from python-louvain

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

def build_comm_graph(fileName):
	data = np.loadtxt(fileName, dtype=np.intc, delimiter = ",", skiprows=2, usecols=(0,1,2))
	nodes = [x[0] for x in data]
	edges = [x[1] for x in data]
	weights = [int(x[2]) for x in data]
	graph = nx.Graph()
	for i in np.arange(len(data)):
		graph.add_node(int(nodes[i]))
		graph.add_edge(int(nodes[i]),int(edges[i]), weight=int(weights[i]))	
	return graph

## ok this is my attempt to generate an analysis procedure that will work with either any trace as the base
## case; that said, because the full trace can have startup/teardown costs, we'll include a parameter that
## tells us how many (if any) events to skip at the head/tail of eventsAvailableBySimCycle.csv of the baseDir
def compute_metrics(sampleDirs, skippedEvents):

    #### let's look at events available
    print "working with " + sampleDirs[0] + "/analysisData/eventsAvailableBySimCycle.csv"

    # first we read all of the files 
    baseData = []
    for i in range(len(sampleDirs)) :
        baseData.append(np.loadtxt(sampleDirs[i] + "/analysisData/eventsAvailableBySimCycle.csv",
                                   dtype=np.intc, delimiter = ",", comments="#"))

    if skippedEvents > 0 : baseData[0] = baseData[0][skippedEvents:len(baseData[0])-skippedEvents]
    
    ## now plot the data points normalized to an x-axis range of 0-100
    
    # we need to record the shortest sample so we can compute the metrics on equal sized arrays when necessary
    minLength = len(baseData[0])

    colorIndex = 0
    # we will set the alpha value to 1.0 for the base sample and 0.5 for all other samples 
    alphaValue = 1.0

    pylab.title('Events Available')
    pylab.xlabel('Simulation Cycles (normalized 0-100)')
    pylab.ylabel('Num Events')
    for index in range(len(baseData)) :
        x_index = np.arange(len(baseData[index])).astype(float)/float(len(baseData[index]))*100
        pylab.plot(x_index, sorted(baseData[index]),
                   color=colors[colorIndex], label=sampleDirs[index], alpha=alphaValue)
        colorIndex = (colorIndex + 1) % len(colors)
        alphaValue=0.5
        if len(baseData[index]) < minLength : minLength = len(baseData[index])

    #pylab.legend(loc='best')
    display_plot('eventsAvailable')

    ## now let's compute the various distance metrics of interest
    # most of these require that the compared vectors have the same length; so we will actually take minLength
    # (computed in above loop) samples (equally distributed) from each vector

    metricFile.write("Events Available\n")
    metricFile.write("Base sample: " + sampleDirs[0] + "\n")
    metricFile.write("Comparison Sample, Wasserstein Distance, Directed Hausdorff, Kolmogorov-Smirnov\n")

    # let's extract equal length samples from the samples :-)
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

    #### now let's look at the how many events are executed by the LPs

    ## to keep our numbers in a reasonable range, we're actually going to plot a count of the rate of events
    ## executed by the LPs relative to the one executing the most; so basically we're going to count how many
    ## LPs execute X% of the max executed by one of the LPs.  an odd computation, but if we try to show the
    ## percentage of events executed, the numbers get really small.
    ## getting 
    ## numbers too small
    ## that execute 

    print "working with " + sampleDirs[0] + "/analysisData/eventsExecutedByLP.csv"

    plotData = []
    min = 100
    max = 0
    for i in range(len(sampleDirs)) :
        percentOfMaxEvents = np.zeros(100, dtype=np.intc)
        # saving this next line in case i ever want to read a csv file with mixed strings/ints in the columns
        #sampleData = np.genfromtxt(sampleDirs[i] + "/analysisData/eventsExecutedByLP.csv",
        #                              dtype='str', delimiter = ",", comments="#", usecols=(0,3))
        sampleData = np.loadtxt(sampleDirs[i] + "/analysisData/eventsExecutedByLP.csv",
                                      dtype=np.intc, delimiter = ",", comments="#", usecols=(3))
        maxNumEvents = np.max(sampleData)
        for j in range(len(sampleData)) :
            percentOfMaxEvents[int(sampleData[j].astype(float)/maxNumEvents*100.0)-1] += 1

        if min > np.min(np.nonzero(percentOfMaxEvents)) : min = np.min(np.nonzero(percentOfMaxEvents))
        if max < np.max(np.nonzero(percentOfMaxEvents)) : max = np.max(np.nonzero(percentOfMaxEvents))
        plotData.append(percentOfMaxEvents)

    colorIndex = 0
    alphaValue= 1.0
    pylab.title('Num LPs executing x% of Max events executed by one LP')
    pylab.xlabel('Percent of Events')
    pylab.ylabel('Number of LPs')
    x_index = np.arange(min,max+1)
    for i in plotData :
        pylab.plot(x_index, i[min:max+1], 
                  color=colors[colorIndex], label=sampleDirs[index], alpha=alphaValue)
        colorIndex = (colorIndex + 1) % len(colors)
        alphaValue=0.5

    #pylab.legend(loc='best')
    display_plot('percentEventsOfMaxbyLP')

    # do we want to compute distance metrics??  not yet, but i assume we will rewrite that component so it
    # becomes a simple procedure call with a matrix argument
    

    #### now let's look at the summaries of event chains.

    # since we don't want to have the size of the measured data to skew the wasserstein distance results, 
    # we'll work from percentages of the populations in question.  furthermore, we don't really want to
    # see the differences by event change length; our real interest is in the impact on the overall event
    # chain computation, we will create a vector that includes measures from local, linked, and global
    # chains as one vector to be analyzed....
 
    print "working with " + sampleDirs[0] + "/analysisData/eventChainsSummary.csv"

    ## import the data and setup the local, linked, and global chain data in one long vector
    baseData = []
    x_index = np.arange(15)
    colorIndex = 0
    alphaValue = 1.0
    pylab.title('Event Chains')
    pylab.xlabel('Local (0-4) Linked(5-9), Global(10-15)')
    pylab.ylabel('Percent of Chain Class (Local, Lined, or Global)')
    for i in range(len(sampleDirs)) :
        sample = np.loadtxt(sampleDirs[i] + "/analysisData/eventChainsSummary.csv",
                                   dtype=np.intc, delimiter = ",", comments="#", usecols=(1,2,3))
        # compute percent of each event chain class
        percentSample = []
        for j in range(len(sample[0])) :
            percentSample.append(sample[:,j].astype(float)/float(np.sum(sample[:,j])))
            #            sample[:,j] = sample[:,j].astype(float)/float(np.sum(sample[:,j])))
        baseData.append(np.concatenate(percentSample).ravel())

        pylab.plot(x_index, np.concatenate(percentSample).ravel(),
                   color=colors[colorIndex], label=sampleDirs[index], alpha=alphaValue)
        colorIndex = (colorIndex + 1) % len(colors)
        alphaValue=0.5
    
    #pylab.legend(loc='best')
    display_plot('eventChains')

    ## now let's compute the various distance metrics of interest
    # most of these require that the compared vectors have the same length; so we will actually take minLength
    # (computed in above loop) samples (equally distributed) from each vector

    metricFile.write("Event Chains\n")
    metricFile.write("Base sample: " + sampleDirs[0] + "\n")
    metricFile.write("Comparison Sample, Wasserstein Distance, Directed Hausdorff, Kolmogorov-Smirnov\n")

    # everything is already equal so this is easy

    for x in range(len(sampleDirs)-1) :
        metricFile.write(sampleDirs[x+1])
        metricFile.write(", %.8f" % wasserstein_distance(baseData[0], baseData[x]))
        metricFile.write(", %.8f" % directed_hausdorff(np.vstack((x_index,baseData[0])).T,
                                                       np.vstack((x_index,baseData[x])).T)[0])
        metricFile.write(", %.8f" % ks_2samp(baseData[0], baseData[x])[0])
        metricFile.write("\n")


    # let's extract equal length samples from the samples :-)
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

    #### now let's look at modularity....

    print "working with " + sampleDirs[0] + "/analysisData/eventsExchanged-remote.csv"

    sampleModularityValues = []

    # compute modularity and store
    for x in sampleDirs :
        sampleModularityValues.append(community.best_partition(build_comm_graph(x + "/analysisData/eventsExchanged-remote.csv")).values())

    pylab.title('Communities of LP Event Communications')
    pylab.xlabel('Modularity Class')
    pylab.ylabel('Number of LPs')
    colorIndex = 0
    alphaValue = 1.0
    index = 0
    # count modularity membership and create scatter plot
    for x in sampleModularityValues :
        modularity = collections.Counter()
        for i in np.arange(len(x)) :
            modularity[x[i]] += 1
        pylab.scatter(modularity.keys(), sorted(modularity.values(), reverse=True), marker='o', 
                      color=colors[colorIndex], label=sampleDirs[index], alpha=alphaValue)
        index += 1
        colorIndex = (colorIndex + 1) % len(colors)
        alphaValue=0.25
                      
    pylab.legend(loc='best')
    display_plot('modularity')

    return

####----------------------------------------------------------------------------------------------------
#### let us begin
####----------------------------------------------------------------------------------------------------

# create a directory to write output graphs
measuresDir = args.outDir + "/"
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
    dirs.append(args.sampleDir + "/" + x)
              
# set colormap and make it the default; prepend the color map with black for plotting the base case
# we should actually tie this to the len(dirs), but that'll have to wait until later.   
#colors = [(0.0, 0.0, 0.0)] + palettable.colorbrewer.qualitative.Dark2_7.mpl_colors
#colors = [(0.0, 0.0, 0.0)] + palettable.colorbrewer.qualitative.Set1_8.mpl_colors
colors = [(0.0, 0.0, 0.0)] + Set1_9.mpl_colors

# change the plots to a grey background grid w/o solid x/y-axis lines
mpl.style.use('ggplot')

metricFile = open(measuresDir + "/metricFile.csv","w")

compute_metrics(dirs,numSkippedEvents)

metricFile.close()
