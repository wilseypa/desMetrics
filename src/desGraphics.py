import sys
import os
import json
import pylab
import matplotlib as mpl
import seaborn as sns
import numpy as np
import brewer2mpl
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

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

# need this as a global variable for the axis printing function
total_num_of_sim_cycles = 0

def setNumOfSimCycles(x):
    global total_num_of_sim_cycles
    total_num_of_sim_cycles = x
    return

def toPercentOfTotalLPs(x, pos=0):
    return '%.1f%%'%((100*x)/total_lps)

def toPercentOfTotalSimCycles(x, pos=0):
    return '%.1f%%'%((100*x)/total_num_of_sim_cycles)

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

# TOM/CHI: it would be nice to have a bit more spacing between the right ylabel and the
# scale on the right y-axis.  can you make that happen for the next two plots?? 

def events_per_sim_cycle_raw():
    fig, ax1 = pylab.subplots()
    pylab.title('Simulation Cycle-by-Cycle Record of Events Available for Execution')
    outFile = outDir + 'eventsAvailableBySimCycle.pdf'
    data = np.loadtxt("analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", skiprows=2)
    ax1.plot(data)
    ax1.set_xlabel('Simulation Cycle (assumes instantaneous event execution)')
    ax1.set_ylabel('Number of Events Available for Execution')
    ax2=ax1.twinx()
    # this is an unnecessary computation
    #data = data.astype(float)
    #ax2.plot(data/float(total_lps))
    ax2.plot(data)
    ax2.set_ylabel('Percent of Total LPs (%s) w/ Events Available' % total_lps)
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(toPercentOfTotalLPs))
    display_graph(outFile)

    # it is not unusual for simulations to have large outliers on number of events available
    # during simulation startup and/or shutdown.  these outliers can dominate making the
    # following graphs less than useful.  therefore, we will remove outliers and plot again

    fig, ax1 = pylab.subplots()
    pylab.title('Simulation Cycle-by-Cycle Record of Events Available for Execution (outliers removed)')
    outFile = outDir + 'eventsAvailableBySimCycle-outliersRemoved.pdf'
    data = np.loadtxt("analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", skiprows=2)
    data = reject_outliers(data)
    #ax1.plot(data, color=colors[0], label='Outliers Removed')
    ax1.plot(data)
    ax1.set_xlabel('Simulation Cycle (assumes instantaneous event execution)')
    ax1.set_ylabel('Number of Events Available for Execution')
    ax2=ax1.twinx()
    #data = gaussian_filter1d(data, sigma=9)
    #ax2.plot(data, color=colors[1], label='Filtered')
    ax2.plot(data)
    ax2.set_ylabel('Percent of Total LPs (%s) w/ Events Available' % total_lps)
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(toPercentOfTotalLPs))
    #pylab.legend(loc='best')
    display_graph(outFile)

    # keeping these around as examples in case....
    #pylab.plot(gaussian_filter1d(data, sigma=9), label='Gaussian Filter')
    #pylab.plot(savgol_filter(data, window_length=9, polyorder=2), label='Savitzky-Golay Filter')
    #if (max(data) / 10) > np.mean(data): pylab.yscale('log')

    return

events_per_sim_cycle_raw()

#--------------------------------------------------------------------------------
# plot histograms on simulation cycles and the number/percentage of events available for execution

# standard histogram using builtin pylab.hist plotting command

def events_per_sim_cycle_histograms():
    data = np.loadtxt("analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", skiprows=2)
    outFile = outDir + 'eventsAvailableBySimCycle-histogram-std.pdf'
    pylab.title('Events Available for Execution (outliers removed).')
    pylab.hist(reject_outliers(data), bins=10, histtype='stepfilled')
    pylab.xlabel('Number of Events')
    pylab.ylabel('Number of Simulation Cycles')
    display_graph(outFile)

    # ok, let's try to build a histogram to show (i) the raw number of cycles that X events
    # are available for execution on the left y-axis and (ii) the percent of cycles that X
    # events are available for execution on the right y-axis. 
    
    fig, ax1 = pylab.subplots()
    pylab.title('Events Available for Execution (outliers removed).')
    outFile = outDir + 'eventsAvailableBySimCycle-histogram-dual.pdf'
    #data = np.loadtxt("analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", skiprows=2)
    #total_num_of_sim_cycles = len(data)+1
    setNumOfSimCycles(len(data)+1)
    mean_events_available = np.mean(reject_outliers(data))
    data = pd.Series(reject_outliers(data)).value_counts()
    data = data.sort_index()
    x_values = np.array(data.keys())
    y_values = np.array(data)
    ax1.plot(x_values, y_values)
    ax1.set_xlabel('Number of Events (Average=%.2f)' % mean_events_available)
    ax1.set_ylabel('Number of Simulation Cycles')
    ax2=ax1.twinx()
    ax2.plot(x_values, y_values)
    ax2.set_ylabel('Percent of Simulation Cycles (%d) w/ Events Available' % total_num_of_sim_cycles)
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(toPercentOfTotalSimCycles))
    display_graph(outFile)

    return

events_per_sim_cycle_histograms()

sys.exit()

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

# pie charts of event chains as a percent of num total chains (in that category)

pylab.title('Distribution of Local Event Chains')
outFile = outDir + 'eventChainSummary-local-pieChart.pdf'
#data = np.loadtxt("analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter = ",", skiprows=2)
labels = '1', '2', '3', '4', '>=5'
percentages = data[:,1].astype(float)/float(np.sum(data[:,1]))
pylab.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=-90)
pylab.axis('equal')
display_graph(outFile)

pylab.title('Distribution of Linked Event Chains')
outFile = outDir + 'eventChainSummary-linked-pieChart.pdf'
#data = np.loadtxt("analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter = ",", skiprows=2)
labels = '1', '2', '3', '4', '>=5'
percentages = data[:,2].astype(float)/float(np.sum(data[:,2]))
pylab.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=-90)
pylab.axis('equal')
display_graph(outFile)


pylab.title('Distribution of Global Event Chains')
outFile = outDir + 'eventChainSummary-global-pieChart.pdf'
#data = np.loadtxt("analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter = ",", skiprows=2)
labels = '1', '2', '3', '4', '>=5'
percentages = data[:,3].astype(float)/float(np.sum(data[:,3]))
pylab.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=-90)
pylab.axis('equal')
display_graph(outFile)


pylab.title('Number Local, Linked, and Global Chains of length X')
outFile = outDir + 'eventChainSummary-cumulative.pdf'
#data = np.loadtxt("analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter = ",", skiprows=2)

for i in range(len(data)-2,0,-1) :
    for j in range(1,len(data[0:])-1) :
                   data[i,j] = data[i,j] + data[i+1,j]
                   
outFile = outDir + 'eventChainSummary-cumulative.pdf'

bar_width = .3
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

## NOTE: i suspect the data file has incorrectly computed this result.  it looks like i
## may have accidently included local events in the total which would skew the results
## (smaller).  need to investigate.  

# let's look at how many LPs provide 95% of the messages to each LP
# column 5 has the data we need
data = np.loadtxt("analysisData/numOfLPsToCoverPercentEventMessagesSent.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(4,5))
outFile = outDir + 'numberOfLPsSending95PercentOfRemoteEvents.pdf'

pylab.title('How many LPs are involved in sending 95% of remote events')
pylab.hist(data[:,1], bins=20, normed=True)
pylab.xlabel('Number of Sending LPs')
pylab.ylabel('Normalized Frequency (sum of the integral is 1)')
display_graph(outFile)



