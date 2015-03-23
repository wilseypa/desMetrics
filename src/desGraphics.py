#import sys
import os
import json
import pylab
import matplotlib as mpl
import seaborn as sns
import numpy as np
import brewer2mpl

#for arg in sys.argv:
#    print arg

# create a directory to write output graphs
outDir = 'outputGraphics/'
if not os.path.exists(outDir):
    os.makedirs(outDir)

# set brewer colormap and make it the default
bmap = brewer2mpl.get_map('Set1', 'qualitative', 4)
colors = bmap.mpl_colors
mpl.rcParams['axes.color_cycle'] = colors

#--------------------------------------------------------------------------------
# here are some helper functions that we can use.

# define a function to display/save the pylab figures.
def display_graph(fileName) :
    pylab.savefig(fileName, bbox_inches='tight')
    pylab.show()

# function to remove outliers from the mean
def reject_outliers(data, m=2):
    outData = data[abs(data - np.mean(data)) < m * np.std(data)]
    if len(outData >0) : return outData 
    return data

#--------------------------------------------------------------------------------
# import the json file of model summary information

# read the json file
json_data = open("analysisData/modelSummary.json")
model_summary = json.load(json_data)

model_name = model_summary["model_name"]
total_lps = model_summary["total_lps"]
total_events = model_summary["total_events"]

#--------------------------------------------------------------------------------
# plot the number of events that are available by simulation cycle

# it is not unusual for simulations to have large outliers on number of events available
# during simulation startup and/or shutdown.  these outliers can dominate making the
# following graphs less than useful.  therefore, we will remove outliers before plotting.

data = np.loadtxt("analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", skiprows=2)
outFile = outDir + 'eventsAvailableBySimCycle.pdf'

pylab.title('Simulation Cycle-by-Cycle Record of Events Available for Execution.')
pylab.plot(data)
pylab.ylabel('Number of Events Available for Execution\n(log scale to minimize outlier dominance)')
if (max(data) / 10) > np.mean(data): pylab.yscale('log')
pylab.xlabel('Simulation Cycle (assumes instantaneous event execution)')
display_graph(outFile)

#data = np.loadtxt("analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", skiprows=2)
outFile = outDir + 'eventsAvailableBySimCycleAsPercentOfTotalLPs.pdf'

data = data.astype(float)
pylab.title('Percent of LPs with Events Available.')
pylab.plot(data/float(total_lps))
pylab.ylabel('Percent of LPs with Events Available')
pylab.xlabel('Simulation Cycle (assumes instantaneous event execution)')
display_graph(outFile)

#--------------------------------------------------------------------------------
# plot histograms of the number of simulation cycles that X events are available

outFile = outDir + 'eventsAvailableBySimCycle-histogram.pdf'
pylab.title('Histogram of Events Available for Execution (outliers removed).')
#pylab.hist(reject_outliers(data), bins=50, normed=True)
pylab.hist(reject_outliers(data))
pylab.xlabel('Number of Events')
pylab.ylabel('Number of Simulation Cycles')
display_graph(outFile)

# THOMAS/CHI: repeat above two functions showing the "Number of Events Availiable" as a
# percentage of the total number of LPs in the simulation model.  follow naming
# conventions.  i.e., add suffix -historgram/-histogram-normalized to the above suggested
# name. 

outFile = outDir + 'eventsAvailableBySimCycle-histogram-normalized.pdf'
pylab.title('Histogram of Events Available for Execution (outliers removed)\n(Normalized so Integral will sum to 1)')
#pylab.hist(reject_outliers(data), bins=50, normed=True)
pylab.hist(reject_outliers(data), normed=True)
pylab.xlabel('Number of Events')
pylab.ylabel('Number of Simulation Cycles')
display_graph(outFile)

#--------------------------------------------------------------------------------
# plot the local/total events executed by each LP (sorted)

# skip the first column of LP names
data = np.loadtxt("analysisData/eventsExecutedByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3))
outFile = outDir + 'totalEventsProcessedByLP.pdf'

# sort the data by the total events executed
sorted_data = data[data[:,2].argsort()]

# need a vector of values for the for the x-axis
x_index = np.arange(len(data))

pylab.title('Total Events processed by each LP (sorted)')
# plotting as bars exhausted memory
pylab.plot(x_index, sorted_data[:,0], color=colors[0], label="Local")
pylab.plot(x_index, sorted_data[:,2], color=colors[1], label="Local+Remote (Total)")
pylab.legend(loc='upper left')
display_graph(outFile)

#--------------------------------------------------------------------------------
# histograms of events executed by each LP

local_events = []
for i in np.arange(len(data)) :
    local = float(data[i,0])
    total = float(data[i,2])
    percent = round((local / total) * 100,2)
    local_events.append(percent)

outFile = outDir + 'localEventsAsPercentofTotalByLP-histogram.pdf'
pylab.title('Histogram of Local Events Executed by each LP')
pylab.hist(sorted(local_events), bins=100)
# turn off scientific notation on the x-axis
ax = pylab.gca()
ax.get_xaxis().get_major_formatter().set_useOffset(False)
pylab.xlabel('Percent of Executed Events that were Locally Generated')
pylab.ylabel('Number of LPs Containing Said Percentage')
display_graph(outFile)

outFile = outDir + 'localAndRemoteEventsExecuted-histogram.pdf'
pylab.title('Local and Remote Events Executed by the LPs')
pylab.hist((data[:,0], data[:,1]), histtype='barstacked', label=('Local', 'Remote'), color=(colors[0], colors[1]), bins=100)
pylab.xlabel('Number of Events')
pylab.ylabel('Number of LPs Executing Said Events')
pylab.legend(loc='best')
display_graph(outFile)

# THOMAS/CHI: repeat (only) the above graph but show totals of local and remote events as
# a percent of the total executed by that each LP.  name the file
# localAndRemoteEventsExecutedAsPercentofTotal-histogram.pdf'


#--------------------------------------------------------------------------------
# plot the percent of local events executed by each LP

### using variables from above....

outFile = outDir + 'percentOfExecutedEventsThatAreLocal.pdf'
pylab.title('Percent of Events Executed that are Local (sorted)')
pylab.plot(x_index, sorted(local_events))
pylab.xlabel('LPs (sorted by percent local)')
pylab.ylabel('Percent of Total Executed')
pylab.ylim((0,100))
# fill the area below the line
ax = pylab.gca()
ax.fill_between(x_index, sorted(local_events), 0, facecolor=colors[0])
display_graph(outFile)

outFile = outDir + 'percentOfExecutedEventsThatAreLocal-histogram.pdf'
pylab.title('Histogram of Events Executed that are Local')
pylab.hist(sorted(local_events))
pylab.xlabel('Percent of Local Events Executed')
pylab.ylabel('Number of LPs Executing Said Percentage')
display_graph(outFile)

#--------------------------------------------------------------------------------
# display graphs of the event chain summaries

data = np.loadtxt("analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter = ",", skiprows=2)
outFile = outDir + 'eventChainSummary-individual.pdf'

bar_width = .3

pylab.title('Number Local, Linked, and Global Chains of length X')
pylab.bar(data[:,0], data[:,1], bar_width, color=colors[0], label="Local")
pylab.bar(data[:,0] + bar_width, data[:,2], bar_width, color=colors[1], label="Linked")
pylab.bar(data[:,0] + bar_width + bar_width, data[:,3], bar_width, color=colors[2], label="Global")
pylab.xticks(data[:,0] + bar_width, ('1', '2', '3', '4', '>=5'))
pylab.legend(loc='best')
pylab.xlabel('Chain Length')
pylab.ylabel('Total Chains of Length X Found')
display_graph(outFile)


outFile = outDir + 'eventChainSummary-cumulative.pdf'

for i in range(len(data)-2,0,-1) :
    for j in range(1,len(data[0:])-1) :
                   data[i,j] = data[i,j] + data[i+1,j]
                   
outFile = outDir + 'eventChainSummary-cumulative.pdf'

bar_width = .3

pylab.title('Number Local, Linked, and Global Chains of length X')
pylab.bar(data[:,0], data[:,1], bar_width, color=colors[0], label="Local")
pylab.bar(data[:,0] + bar_width, data[:,2], bar_width, color=colors[1], label="Linked")
pylab.bar(data[:,0] + bar_width + bar_width, data[:,3], bar_width, color=colors[2], label="Global")
pylab.xticks(data[:,0] + bar_width, ('1', '>=2', '>=3', '>=4', '>=5'))
pylab.xlabel('Chain Length')
pylab.ylabel('Total Chains of Length >= X Found')
pylab.legend(loc='best')
display_graph(outFile)

# THOMAS/CHI: repeat the above two graphs where you compute the number of chains as a
# percentage of the total events.  the question we have to ask here, is how do we achieve
# this.  more specifically, do we compute the number of chains against the total events or
# do we use the number of events in the chain agains the total.  as an example, if the
# source files says there are 20 chains of length 3 and a total of 200 events, do we
# compute the percentage as 20/200 or do we compute the percentage as (3*20)/200?  i do
# not know.  if we have to, i suppose we could just do both and decide which to use
# later. 

#--------------------------------------------------------------------------------
# histograms of event chains longer than 1

## Local Chains

data = np.loadtxt("analysisData/localEventChainsByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5))
outFile = outDir + 'eventChains-byLP-local-histogram.pdf'

percent_long_chains = []
for i in np.arange(len(data)) :
    num_chains = 0
    for j in np.arange(len(data[i])) :
                        num_chains = num_chains + float(data[i,j])
    percent = round(((num_chains - float(data[i,0])) / num_chains) * 100,2)
    percent_long_chains.append(percent)
    
pylab.title('Histogram of Local Event Chains Longer than 1.')
pylab.hist(sorted(percent_long_chains), bins=100)
pylab.xlabel('Percent of Total Local Chains of Length > 1')
pylab.ylabel('Number of LPs Containing Said Percent')
display_graph(outFile)

outFile = outDir + 'eventChains-byLP-local-histogram-normalized.pdf'
pylab.title('Histogram of Local Event Chains Longer than 1\n(Normalized so Integral will sum to 1)')
pylab.hist(sorted(percent_long_chains), bins=100, normed=True)
pylab.xlabel('Percent of Total Local Chains of Length > 1')
pylab.ylabel('Number of LPs Containing Said Percent\n(Normalized so Integral will sum to 1)')
display_graph(outFile)

## Linked Chains

data = np.loadtxt("analysisData/linkedEventChainsByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5))
outFile = outDir + 'eventChains-byLP-linked-histogram.pdf'

percent_long_chains = []
for i in np.arange(len(data)) :
    num_chains = 0
    for j in np.arange(len(data[i])) :
                        num_chains = num_chains + float(data[i,j])
    percent = round(((num_chains - float(data[i,0])) / num_chains) * 100,2)
    percent_long_chains.append(percent)
    
pylab.title('Histogram of Linked Event Chains Longer than 1.')
pylab.hist(sorted(percent_long_chains), bins=100)
pylab.xlabel('Percent of Total Linked Chains of Length > 1')
pylab.ylabel('Number of LPs Containing Said Percent')
display_graph(outFile)

outFile = outDir + 'eventChains-byLP-linked-histogram-normalized.pdf'
pylab.title('Histogram of Linked Event Chains Longer than 1\n(Normalized so Integral will sum to 1)')
pylab.hist(sorted(percent_long_chains), bins=100, normed=True)
pylab.xlabel('Percent of Total Linked Chains of Length > 1')
pylab.ylabel('Number of LPs Containing Said Percent\n(Normalized so Integral will sum to 1)')
display_graph(outFile)

## Global Chains

data = np.loadtxt("analysisData/globalEventChainsByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5))
outFile = outDir + 'eventChains-byLP-global-histogram.pdf'

percent_long_chains = []
for i in np.arange(len(data)) :
    num_chains = 0
    for j in np.arange(len(data[i])) :
                        num_chains = num_chains + float(data[i,j])
    percent = round(((num_chains - float(data[i,0])) / num_chains) * 100,2)
    percent_long_chains.append(percent)
    
pylab.title('Histogram of Global Event Chains Longer than 1.')
pylab.hist(sorted(percent_long_chains), bins=100)
pylab.xlabel('Percent of Total Global Chains of Length > 1')
pylab.ylabel('Number of LPs Containing Said Percent')
display_graph(outFile)

outFile = outDir + 'eventChains-byLP-global-histogram-normalized.pdf'
pylab.title('Histogram of Global Event Chains Longer than 1\n(Normalized so Integral will sum to 1)')
pylab.hist(sorted(percent_long_chains), bins=100, normed=True)
pylab.xlabel('Percent of Total Global Chains of Length > 1')
pylab.ylabel('Number of LPs Containing Said Percent\n(Normalized so Integral will sum to 1)')
display_graph(outFile)

#--------------------------------------------------------------------------------
# plots of the number of LPs each LP receives events from

## NOTE: i believe the data file has incorrectly computed this result.  it looks like i may have accidently included local events in the total which would skew the results (smaller).  need to investigate.

# let's look at how many LPs provide 95% of the messages to each LP
# column 5 has the data we need
data = np.loadtxt("analysisData/numOfLPsToCoverPercentEventMessagesSent.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(4,5))
outFile = outDir + 'numberOfLPsSending95PercentOfRemoteEvents.pdf'

pylab.title('How many LPs are involved in sending 95% of remote events')
pylab.hist(data[:,1], bins=20, normed=True)
pylab.xlabel('Number of Sending LPs')
pylab.ylabel('Normalized Frequency (sum of the integral is 1)')
display_graph(outFile)



