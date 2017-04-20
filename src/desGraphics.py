import sys
import os
import json
import pylab
import matplotlib as mpl
import numpy as np
import brewer2mpl
import seaborn as sb
import pandas as pd
import collections
import networkx as nx
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

#for arg in sys.argv:
#    print arg

# create a directory to write output graphs
outDir = 'outputGraphics/'
if not os.path.exists(outDir):
    os.makedirs(outDir)

# set brewer colormap and make it the default
bmap = brewer2mpl.get_map('Set1', 'qualitative', 8)
colors = bmap.mpl_colors
mpl.rcParams['axes.color_cycle'] = colors

# ok, here's another (better) way to change colors (maybe)....
mpl.style.use('ggplot')

# tell matplotlib to use type 1 fonts....
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = False

# increase size of default fonts
mpl.rcParams.update({'font.size': 16})
mpl.rcParams.update({'font.size': 16})
mpl.rcParams.update({'axes.titlesize': 16})
mpl.rcParams.update({'xtick.labelsize': 14})
mpl.rcParams.update({'ytick.labelsize': 14})


#--------------------------------------------------------------------------------
# here are some helper functions that we can use.

# define a function to display/save the pylab figures.
def display_graph(fileName) :
    print "Creating graphics " + fileName
    print "    ....writing pdf"
    pylab.savefig(fileName + ".pdf", bbox_inches='tight')
#    print "    ....writing eps"
#    pylab.savefig(fileName + ".eps", bbox_inches='tight')
# this conversion takes an enormous amount of time.  uncomment and use only when you really need it.
#    print "    ....converting to jpg"
#    os.system("convert " + fileName + ".eps " + fileName + ".jpg")
    pylab.clf()
#    pylab.show()
    return


# function to remove outliers from the mean
def reject_outliers(data, m=2):
    outData = data[abs(data - np.mean(data)) < m * np.std(data)]
    if len(outData > 0) : return outData 
    return data

# function to remove the first and last 1% of events as outliers
def reject_first_last_outliers(data):
    trimLength = int((len(data)+1)/100)
    return data[trimLength-1:len(data)-trimLength]

# need this as a global variable for the axis printing function
total_num_of_sim_cycles = 0

def setNumOfSimCycles(x):
    global total_num_of_sim_cycles
    total_num_of_sim_cycles = x
    return

# need this as a global variable for x-axis printing function on trimmed plots
xLabelOffsetVal = 0

def setxLabelOffset(x):
    global xLabelOffsetVal
    xLabelOffsetVal = x
    return

def toPercentOfTotalLPs(x, pos=0):
    return '%.3f%%'%(100*(x/total_lps))

def toPercent(x, pos=0):
    return '%.1f%%'%(100*x)

def toPercentOfTotalSimCycles(x, pos=0):
    return '%.1f%%'%((100*x)/total_num_of_sim_cycles)

def xlabelsOffset(x, pos=0):
    return int(x+xLabelOffsetVal)

#--------------------------------------------------------------------------------
# import the json file of model summary information

# read the json file
json_data = open("analysisData/modelSummary.json")
model_summary = json.load(json_data)

model_name = model_summary["model_name"]
total_lps = model_summary["total_lps"]
total_events = model_summary["total_events"]

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# functions to plot events available for execution

#--------------------------------------------------------------------------------
# plot the number of events that are available by simulation cycle

def events_per_sim_cycle_raw():
    fig, ax1 = pylab.subplots()
    outFile = outDir + 'eventsAvailableBySimCycle'
    data = np.loadtxt("analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", skiprows=2)
    pylab.title('Total LPs: %s. ' % "{:,}".format(total_lps) +
                'Total Sim Cycles: %s. '% "{:,}".format(len(data)))
    setNumOfSimCycles(len(data)+1)
    ax1.set_xlabel('Simulation Cycle (Total=%s)' % "{:,}".format(total_num_of_sim_cycles))
    ax1.set_ylabel('Number of Events (Ave=%.2f)' % np.mean(data))
    ax1.plot(data)
    ax2=ax1.twinx()
    ax2.plot(data)
    ax2.set_ylabel('Percent of Total LPs (Ave=%.3f%%)' % ((np.mean(data)/total_lps)*100))
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(toPercentOfTotalLPs))
    display_graph(outFile)

    # it is not unusual for simulations to have large outliers on number of events available
    # during simulation startup and/or shutdown.  these outliers can dominate making the
    # following graphs less than useful.  often these outliers occur at startup/teardown.
    # thus, we will use two separate techniques to explore removing these outliers.  in
    # the first, we will simply trim the first and last 1% of the simulation cycles and
    # plot the middle 98%.  in the second will remove outliers that lie outside 2 std
    # deviations of the mean. 

    fig, ax1 = pylab.subplots()
    outFile = outDir + 'eventsAvailableBySimCycle-trimmed'
    trimmedData = reject_first_last_outliers(data)
    trimLength = len(data) - len(trimmedData)
    pylab.title('Total LPs: %s; ' % "{:,}".format(total_lps) +
                'Total Sim Cycles: %s. '% "{:,}".format(len(trimmedData)))
    setNumOfSimCycles(len(trimmedData))
    ax1.set_xlabel('Simulation Cycle (Total=%s)' % "{:,}".format(total_num_of_sim_cycles))
    ax1.set_ylabel('Number of Events (Ave=%.2f)' % np.mean(trimmedData))
    ax1.plot(trimmedData)
    setxLabelOffset(trimLength-1)
    ax1.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(xlabelsOffset))
    ax2=ax1.twinx()
    ax2.plot(trimmedData)
    ax2.set_ylabel('Percent of Total LPs (Ave=%.3f%%)' % ((np.mean(trimmedData)/total_lps)*100))
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(toPercentOfTotalLPs))
    display_graph(outFile)

    fig, ax1 = pylab.subplots()
    outFile = outDir + 'eventsAvailableBySimCycle-trimmed-withSorted'
    pylab.title('Total LPs: %s; ' % "{:,}".format(total_lps) +
                'Total Sim Cycles: %s. '% "{:,}".format(len(trimmedData)))
    ax1.set_xlabel('Simulation Cycle (Total=%s)' % "{:,}".format(total_num_of_sim_cycles))
    ax1.tick_params(axis='x',labelbottom='off')
    ax1.set_ylabel('Number of Events (Ave=%.2f)' % np.mean(trimmedData))
    lns1 = ax1.plot(trimmedData, color=colors[0], label='Events Available: Runtime order')
    ax2=ax1.twinx()
    lns2 = ax2.plot(sorted(trimmedData), color=colors[1], label='Events Available: Sorted order')
    ax2.set_ylabel('Percent of Total LPs (Ave=%.3f%%)' % ((np.mean(trimmedData)/total_lps)*100))
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(toPercentOfTotalLPs))
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best', frameon=True)
    ax1.set_xlim([0, len(trimmedData)+(trimLength/2)])
    ax2.set_xlim([0, len(trimmedData)+(trimLength/2)])
    display_graph(outFile)

    fig, ax1 = pylab.subplots()
    outFile = outDir + 'eventsAvailableBySimCycle-outliersRemoved'
    data = reject_outliers(data)
    pylab.title('Total LPs: %s; ' % "{:,}".format(total_lps) +
                'Total Sim Cycles: %s. '% "{:,}".format(len(data)))
    ax1.plot(data)
    ax1.set_xlabel('Simulation Cycle (Total=%s)' % "{:,}".format(total_num_of_sim_cycles))
    ax1.set_ylabel('Number of Events (Ave=%.2f)' % np.mean(data))
    ax2=ax1.twinx()
    ax2.plot(data)
    ax2.set_ylabel('Percent of Total LPs (Ave=%.3f%%)' % ((np.mean(data)/total_lps)*100))
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(toPercentOfTotalLPs))
    display_graph(outFile)

    fig, ax1 = pylab.subplots()
    pylab.title('Total LPs: %s; ' % "{:,}".format(total_lps) +
                'Total Sim Cycles: %s. '% "{:,}".format(len(data)))
    outFile = outDir + 'eventsAvailableBySimCycle-outliersRemoved-sorted'
    sorted_data = sorted(data)
    ax1.plot(sorted_data)
    ax1.set_xlabel('Simulation Cycles sorted by # events avail (Total=%s)' % "{:,}".format(total_num_of_sim_cycles))
    ax1.tick_params(axis='x',labelbottom='off')
    ax1.set_ylabel('Number of Events (Ave=%.2f)' % np.mean(data))
    ax2=ax1.twinx()
    ax2.plot(sorted_data)
    ax2.set_ylabel('Percent of Total LPs (Ave=%.3f%%)' % ((np.mean(data)/total_lps)*100))
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(toPercentOfTotalLPs))
    display_graph(outFile)
    return

#--------------------------------------------------------------------------------
# plot histograms on simulation cycles and the number/percentage of events available for execution

# standard histogram using builtin pylab.hist plotting command

def events_per_sim_cycle_histograms():
    data = np.loadtxt("analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", skiprows=2)
    trimmedData = reject_first_last_outliers(data)
    fig, ax1 = pylab.subplots()
    outFile = outDir + 'eventsAvailableBySimCycle-histogram-std'
    pylab.title('Total LPs: %s; ' % "{:,}".format(total_lps) +
                'Total Sim Cycles: %s. '% "{:,}".format(len(trimmedData)))
    ax1.hist(trimmedData, bins=100, histtype='stepfilled')
    ax1.set_xlabel('Number of Events')
    ax1.set_ylabel('Number of Simulation Cycles')
    ax2=ax1.twinx()
    ax2.hist(trimmedData, bins=100, histtype='stepfilled')
    ax2.set_ylabel('Percent of Simulation Cycles')
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(toPercentOfTotalSimCycles))
    display_graph(outFile)

    # ok, so now let's build a histogram of the % of LPs with active events.

    fig, ax1 = pylab.subplots()
    outFile = outDir + 'percentOfLPsWithAvailableEvents'
    pylab.title('Percent of LPs w/ Available Events as a Percentage of the Total Sim Cycles')
    
    ax1.hist(trimmedData.astype(float)/float(total_lps), bins=100, histtype='stepfilled')
    ax1.set_xlabel('Number of Events as a percentage of Total LPs')
    ax1.set_ylabel('Number of Sim Cycles said Percentage Occurs')
#    ax1 = pylab.gca()
    ax1.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(toPercent))
#    ax.get_yaxis().set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f%%'))
    ax2=ax1.twinx()
    ax2.hist(trimmedData, bins=100, histtype='stepfilled')
    ax2.set_ylabel('Percent of Simulation Cycles')
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(toPercentOfTotalSimCycles))
    display_graph(outFile)

    # ok, let's present the histogram data using pandas series/value_counts.  much nicer plot.
    fig, ax1 = pylab.subplots()
    outFile = outDir + 'eventsAvailableBySimCycle-histogram-dual'
    pylab.title('Total LPs: %s; ' % "{:,}".format(total_lps) +
                'Total Sim Cycles: %s. '% "{:,}".format(len(trimmedData)))
    setNumOfSimCycles(len(data)+1)
    mean_events_available = np.mean(trimmedData)
    data = pd.Series(trimmedData).value_counts()
    data = data.sort_index()
    x_values = np.array(data.keys())
    y_values = np.array(data)
    ax1.plot(x_values, y_values)
    ax1.set_xlabel('Number of Events (Ave=%.2f)' % mean_events_available)
    ax1.set_ylabel('Number of Simulation Cycles')
    ax2=ax1.twinx()
    ax2.plot(x_values, y_values)
    ax2.set_ylabel('Percent of Simulation Cycles')
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(toPercentOfTotalSimCycles))
    display_graph(outFile)

    return

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# functions to plot the local/total event counts

#--------------------------------------------------------------------------------
# plot the local/total events executed by each LP (sorted)

def total_local_events_exec_by_lp():
    fig, ax1 = pylab.subplots()
    pylab.title('Total Events processed by each LP (sorted)')
    # skip the first column of LP names
    data = np.loadtxt("analysisData/eventsExecutedByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3))
    outFile = outDir + 'totalEventsProcessedByLP'
    # sort the data by the total events executed
    sorted_data = data[data[:,2].argsort()]
    # need a vector of values for the for the x-axis
    x_index = np.arange(len(data))
    # plotting as bars exhausted memory
    lns1 = ax1.plot(x_index, sorted_data[:,0], color=colors[0], label="Local")
    lns2 = ax1.plot(x_index, sorted_data[:,2], color=colors[1], label="Local+Remote (Total)")
#    ax1.legend(loc='upper left')
    ax1.set_xlabel('LPs (sorted by total events executed)')
    ax1.tick_params(axis='x',labelbottom='off')
    ax1.set_ylabel('Number of Events Executed')
    ax2=ax1.twinx()
    lns3 = ax2.plot(percent_of_LP_events_that_are_local(sorted_data), color=colors[2], label="% Local (scale right)")
    ax2.set_ylabel('Percent of Local Events (Ave=%.2f%%)' % np.mean(percent_of_LP_events_that_are_local(sorted_data)))
    ax2.set_ylim(0, 100)
    ax2.get_yaxis().set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f%%'))
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')
    display_graph(outFile)
    return

#--------------------------------------------------------------------------------
# histograms of events executed by each LP

# expand to show % of LPs on right y-axis


# helper function to compute percent of events that are local
def percent_of_LP_events_that_are_local(data):
    local_events = []
    for i in np.arange(len(data)) :
        local = float(data[i,0])
        total = float(data[i,2])
        percent = round((local / total) * 100,2)
        local_events.append(percent)
    return local_events

def histograms_of_events_exec_by_lp():
    pylab.title('Local Events Executed by each LP')
    outFile = outDir + 'localEventsAsPercentofTotalByLP-histogram'
    data = np.loadtxt("analysisData/eventsExecutedByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3))
    # convert data to percentage of total executed by that LP
    pylab.hist(sorted(percent_of_LP_events_that_are_local(data)), bins=100)
    # set the x-axis formatting....<ugh>
    ax = pylab.gca()
    ax.get_xaxis().set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f%%'))
    pylab.xlabel('Percent of Total Events that are Local')
    pylab.ylabel('Number of LPs (Total=%s)' % "{:,}".format(total_lps))
    display_graph(outFile)

    pylab.title('Local and Remote Events Executed by the LPs')
    outFile = outDir + 'localAndRemoteEventsExecuted-histogram-stacked'
    pylab.hist((data[:,0], data[:,1]), histtype='barstacked', label=('Local', 'Remote'), color=(colors[0], colors[1]), bins=100)
    pylab.xlabel('Number of Events')
    pylab.ylabel('Number of LPs (Total=%s)' % "{:,}".format(total_lps))
    pylab.legend(loc='best')
    display_graph(outFile)
    return

#--------------------------------------------------------------------------------
# plot the percent of local events executed by each LP

def profile_of_local_events_exec_by_lp():
    pylab.title('Locally Generated Events')
    outFile = outDir + 'percentOfExecutedEventsThatAreLocal'
    data = np.loadtxt("analysisData/eventsExecutedByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3))
    x_index = np.arange(len(data))
    pylab.plot(x_index, sorted(percent_of_LP_events_that_are_local(data)))
    pylab.xlabel('LPs (sorted by percent local)')
    pylab.tick_params(axis='x',labelbottom='off')
    pylab.ylabel('Percent of Total Executed (Ave=%.2f%%)' % np.mean(percent_of_LP_events_that_are_local(data)))
    pylab.ylim((0,100))
    # fill the area below the line
    ax = pylab.gca()
#    ax.fill_between(x_index, sorted(percent_of_LP_events_that_are_local(data)), 0, facecolor=colors[0])
    ax.get_yaxis().set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f%%'))
    display_graph(outFile)

    pylab.title('Locally Generated Events Executed')
    outFile = outDir + 'percentOfExecutedEventsThatAreLocal-histogram'
    pylab.hist(sorted(percent_of_LP_events_that_are_local(data)))
    ax = pylab.gca()
    ax.get_xaxis().set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f%%'))
    pylab.xlabel('Percent of Local Events Executed')
    pylab.ylabel('Number of LPs (Total=%s)' % "{:,}".format(total_lps))
    display_graph(outFile)
    return

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# functions to plot event chain results

#--------------------------------------------------------------------------------
# display graphs of the event chain summaries

def plot_event_chain_summaries():
    pylab.title('Number of Event Chains of length X')
    data = np.loadtxt("analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter = ",", skiprows=2)
    outFile = outDir + 'eventChainSummary-individual'
    bar_width = .3
    pylab.bar(data[:,0], data[:,1], bar_width, color=colors[0], label="Local")
    pylab.bar(data[:,0] + bar_width, data[:,2], bar_width, color=colors[1], label="Linked")
    pylab.bar(data[:,0] + bar_width + bar_width, data[:,3], bar_width, color=colors[2], label="Global")
    pylab.xticks(data[:,0] + bar_width, ('1', '2', '3', '4', '>=5'))
    pylab.legend(loc='best')
    pylab.xlabel('Chain Length')
    pylab.ylabel('Total Chains of Length X Found')
    display_graph(outFile)
    return

# show the cumulative event chains (i.e., for chains of length 2, also count longer chains)
def plot_event_chain_cumulative_summaries():
    pylab.title('Cumulative Number Event Chains of length X')
    outFile = outDir + 'eventChainSummary-cumulative'
    data = np.loadtxt("analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter = ",", skiprows=2)
    for i in range(len(data)-2,-1,-1) :
        for j in range(1,len(data[0])) :
            data[i,j] = data[i,j] + data[i+1,j]
    bar_width = .3
    pylab.bar(data[:,0], data[:,1], bar_width, color=colors[0], label="Local")
    pylab.bar(data[:,0] + bar_width, data[:,2], bar_width, color=colors[1], label="Linked")
    pylab.bar(data[:,0] + bar_width + bar_width, data[:,3], bar_width, color=colors[2], label="Global")
    pylab.xticks(data[:,0] + bar_width, ('1', '>=2', '>=3', '>=4', '>=5'))
    pylab.xlabel('Chain Length')
    pylab.ylabel('Total Chains of Length >= X Found')
    pylab.legend(loc='best')
    display_graph(outFile)
    return

# display pie charts of event chain summaries

def plot_event_chain_summaries_pie_charts(data, type):
    pylab.title('Distribution of %s Event Chains\n' % type)
    outFile = outDir + 'eventChainSummary-pie-chart-%s'%type
    labels = '1', '2', '3', '4', '>=5'
    percentages = data.astype(float)/float(np.sum(data))
    pylab.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%')
    pylab.axis('equal')
    display_graph(outFile)
    return

def plot_percent_of_events_in_event_chains(data, total_events_of_class, type):
    pylab.title('Percent of Events in %s Event Chains\n' % type)
    outFile = outDir + 'eventChainEventTotals-pie-chart-%s'%type
    labels = '1', '2', '3', '4', '>=5'
    # convert the counts of chains to counts of events 
    data[1] = data[1] * 2
    data[2] = data[2] * 3
    data[3] = data[3] * 4
    data[4] = 0
    data[4] = total_events_of_class - np.sum(data)
    percentages = data.astype(float)/float(total_events_of_class)
    pylab.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%')
    pylab.axis('equal')
    display_graph(outFile)
    return

# plot event chains by LP
def plot_event_chains_by_lp(data, type):
    pylab.title('%s Event Chains by LP (individually sorted)' % type)
    outFile = outDir + 'eventChains-byLP-%s'%type
    sorted_data = data[data[:,0].argsort()]
    pylab.plot(sorted_data[:,0], label='Len=1')
    sorted_data = data[data[:,1].argsort()]
    pylab.plot(sorted_data[:,1], label='Len=2')
    sorted_data = data[data[:,2].argsort()]
    pylab.plot(sorted_data[:,2], label='Len=3')
    sorted_data = data[data[:,3].argsort()]
    pylab.plot(sorted_data[:,3], label='Len=4')
    sorted_data = data[data[:,4].argsort()]
    pylab.plot(sorted_data[:,4], label='Len>=5')
    pylab.xlabel('LPs')
    pylab.tick_params(axis='x',labelbottom='off')
    pylab.ylabel('Number of Chains')
    pylab.legend(loc='best')
    display_graph(outFile)
    return

#--------------------------------------------------------------------------------
# plots of the number of LPs each LP receives events from

def plot_number_of_lps_sending_remote_events(data):
    pylab.title('Number of LPs Sending Remote Events (sorted)')
    outFile = outDir + 'numberOfSendingLPs'
    pylab.plot(data[data[:,5].argsort()][:,5], color=colors[0], label = '100% of total remote events')
    pylab.plot(data[data[:,4].argsort()][:,4], color=colors[1], label = '95% of total remote events')
#    pylab.plot(data[data[:,2].argsort()][:,2], color=colors[3], label = '80% of total remote events')
#    pylab.plot(data[data[:,3].argsort()][:,3], color=colors[4], label = '90% of total remote events')
    pylab.plot(data[data[:,1].argsort()][:,1], color=colors[2], label = '75% of total remote events')
    pylab.xlabel('Receiving LP (Total=%s)' % "{:,}".format(total_lps))
    pylab.tick_params(axis='x',labelbottom='off')
    pylab.ylabel('Number of Sending LPs')
    pylab.legend(loc='best')
    display_graph(outFile)
    return

# let's look at how many LPs provide 95% of the messages to each LP
# column 5 has the data we need
def histogram_of_lps_sending_95_percent_of_remote_events(data):
    pylab.title('How many LPs are involved in sending 95% of remote events')
    outFile = outDir + 'sending95PercentOfRemoteEvents-hist'
    pylab.hist(data[:,5], bins=20)
    pylab.xlabel('Number of Sending LPs')
    pylab.ylabel('Frequency')
    display_graph(outFile)
    return

def plots_of_lp_event_exchanges():
    pylab.title('Remote Events Sent Between LPs')
    data = np.loadtxt("analysisData/eventsExchanged-remote.csv", dtype=np.float_, delimiter = ",", skiprows=2, usecols=(2,3,4,5))
    outFile = outDir + 'countsOfLpToLpEventExchanges'
    pylab.plot(data[data[:,0].argsort()][:,0].astype(np.intc))
#    pylab.xlabel('Number of Events')
    pylab.tick_params(axis='x',labelbottom='off')
    pylab.ylabel('Number of Events Sent')
    display_graph(outFile)

    pylab.title('Timestamp Deltas of Remote Events')
    outFile = outDir + 'timeStampDeltasOfRemoteEvents'
    stride = max(int(max(len(data[:,1]),len(data[:,2]),len(data[:,3]))/20),1)
    pylab.plot(data[data[:,1].argsort()][:,1], color=colors[0], label="Minimum", marker='o', markevery=stride)
    pylab.plot(data[data[:,3].argsort()][:,3], color=colors[1], label="Average", marker='x', markevery=stride)
#    pylab.plot(data[data[:,2].argsort()][:,2], color=colors[2], label="Maximum", marker='*', markevery=stride)
    pylab.tick_params(axis='x',labelbottom='off')
    pylab.ylabel('Timestamp Delta (ReceiveTime - SendTime)')
    pylab.ylim([-.1,np.amax(data[:,3].astype(np.intc))+1])
#    pylab.yscale('log')
    pylab.legend(loc='best')
    display_graph(outFile)

    pylab.title('Histogram of Timestamp Deltas of Remote Events')
    outFile = outDir + 'timeStampDeltasOfRemoteEvents-hist'
    pylab.hist((data[:,1],data[:,3],data[:,2]), label=('Minimum', 'Average', 'Maximum'), color=(colors[0], colors[1], colors[2]), bins=10)
    pylab.xlabel('Timestamp Delta (ReceiveTime - SendTime)')    
    pylab.ylabel('Number of LPs')
    pylab.legend(loc='best')
    display_graph(outFile)

    return

# plots in and out degree of LPs (degree = # of LPs an LP sends to or receives from)
def plot_lp_degrees():
	outFile = outDir + 'countsOfDegreeLPbyLP'
	data = np.loadtxt("analysisData/eventsExchanged-remote.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(0,1,2))

	# data structures for holding LPs sent, received, and the number of events
	inLP = [x[1] for x in data]
	outLP = [x[0] for x in data]
	weights = [int(x[2]) for x in data]
	inDegree = collections.Counter()
	outDegree = collections.Counter()
	inCount = collections.Counter()
	outCount = collections.Counter()
	eventsSent = collections.Counter()
	eventsCount = collections.Counter()
	eventsAvg = {}

	# count the in and out degree of a given LP, and total the events sent
	for i in np.arange(len(data)):
		inDegree[inLP[i]] += 1
		outDegree[outLP[i]] += 1
		eventsSent[inLP[i]] += weights[i]

	# count the number of LPs who have the same degree and their events
	for i in np.arange(len(inDegree)):
		inCount[inDegree[i]] += 1
		outCount[outDegree[i]] += 1
		eventsCount[inDegree[i]] += eventsSent[i]
		
	# take the average events sent by LP degree
	for i in inCount:
		eventsAvg[i] = float(eventsCount[i]) / int(inCount[i])
	
	keyList = sorted(list(set(inCount.keys() + outCount.keys())))
	for key in keyList:
		if key not in inCount:
			inCount[key] = 0
		if key not in outCount:
			outCount[key] = 0
		if key not in eventsAvg:
			eventsAvg[key] = 0
		
	fig, ax1 = pylab.subplots()
	bar_width = 0.30
	ax1.plot(np.nan, '-', marker='o', color=colors[2], label = "average events")
	ax1.bar(np.arange(len(keyList)), inCount.values(), width=bar_width, align='edge', label='In-Degree', color=colors[0])
	ax1.bar(np.arange(len(keyList))+bar_width, outCount.values(), width=bar_width, align='edge', label='Out-Degree',color=colors[1])
	ax1.set_xticklabels(sorted(keyList))
	ax2 = ax1.twinx()
	ax2.plot(np.arange(len(keyList)),sorted(eventsAvg.values()), marker='o',color=colors[2], label="average events")
	ax1.grid(b=False)
	ax2.grid(b=False)
	ax1.set_xlabel('LP Degree Counts')
	ax1.set_ylabel('Number of LPs')
	ax2.set_ylabel('Events Sent')
	pylab.title('LP by LP Communication')
	ax1.legend(loc='best')
	display_graph(outFile)
	return

# plots both betweenness and closeness centralities
def plot_graph_centrality():
	outFile = outDir + 'Betweeness Centrality'
	data = np.loadtxt("analysisData/eventsExchanged-remote.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(0,1,2))

	nodes = [x[0] for x in data]
	edges = [x[1] for x in data]
	weights = [int(x[2]) for x in data]
	G = nx.Graph()
	for i in np.arange(len(data)):
		G.add_node(int(nodes[i]))
		G.add_edge(int(nodes[i]),int(edges[i]), weight=int(weights[i]))
		
	centrality = nx.betweenness_centrality(G)
	fig, ax = pylab.subplots()
	# bins vary by graphs, need to find a better way to make them
	ax.hist(centrality.values(), bins=[0,.002,.004,.006,.008,.01])
	ax.set_ylabel('Frequency')
	ax.set_xlabel('Betweeness Centrality Value')
	pylab.title('Betweenness Centrality of LP by LP Communication')
	pylab.legend(loc='best')
	display_graph(outFile)
	
	outFile = outDir + 'Closeness Centrality'
	centrality = nx.closeness_centrality(G)
	fig, ax = pylab.subplots()
	ax.hist(centrality.values(), bins=[0,.01,.02,.03,.04,.05])
	ax.set_ylabel('Frequency')
	ax.set_xlabel('Closeness Centrality Value')
	pylab.title('Closeness Centrality of LP by LP Communication')
	pylab.legend(loc='best')
	display_graph(outFile)
	return
	
# plots modularity of a graph. Right now, needs csv from gephi until modularity is calculated here
def plot_modularity():
	outFile = outDir + 'Communities'
	data = np.loadtxt(outFileName, dtype=np.intc, delimiter = ",", skiprows=1, usecols=(4,))
	modularity = collections.Counter()
	for i in np.arange(len(data)):
		modularity[data[i]] +=1
		
	mean = np.mean(modularity.values())
	std_dev = np.std(modularity.values())
	start = min(modularity.keys(), key=int)
	end = max(modularity.keys(), key=int)

	fig, ax = pylab.subplots()
	ax.scatter(modularity.keys(),modularity.values(),color=colors[0])
	pylab.axhline(mean,color=colors[2],label="mean")
	pylab.axhline(mean+std_dev,color=colors[1],label="standard deviation")
	pylab.axhline(mean-std_dev,color=colors[1])
	ax.set_ylabel('Number of LPs')
	ax.set_xlabel('Modularity Class')
	pylab.xticks(np.arange(start, end+1, 10))
	pylab.title('Communities in LP Communication Graph')
	pylab.legend(loc='best', shadow=True)
	display_graph(outFile)
	
#--------------------------------------------------------------------------------
# functions to plot graphs by category

def plot_event_execution_data():
    # plot total event data
    events_per_sim_cycle_raw()
    events_per_sim_cycle_histograms()

    # plot events by being local (self generated) or remote (generated by some other LP) 
    total_local_events_exec_by_lp()
    histograms_of_events_exec_by_lp()
    profile_of_local_events_exec_by_lp()
    return

def plot_event_chain_data():
    # plot summary of all event chains in the system
    plot_event_chain_summaries()
    plot_event_chain_cumulative_summaries()

    data = np.loadtxt("analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter = ",", skiprows=2)
    # plot chain data by chain count (chains of length n count as 1)
    plot_event_chain_summaries_pie_charts(data[:,1], 'Local')
    plot_event_chain_summaries_pie_charts(data[:,2], 'Linked')
    plot_event_chain_summaries_pie_charts(data[:,3], 'Global')
    # plot chain data by event count (chains of length n count as n)
    total_local_events = np.sum(np.loadtxt("analysisData/eventsExecutedByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2))[:,0])
    plot_percent_of_events_in_event_chains(data[:,1], total_local_events, 'Local')
    plot_percent_of_events_in_event_chains(data[:,2], total_local_events, 'Linked')
    plot_percent_of_events_in_event_chains(data[:,3], total_events, 'Global')
    data = np.loadtxt("analysisData/localEventChainsByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5))
    plot_event_chains_by_lp(data, 'Local')
    data = np.loadtxt("analysisData/linkedEventChainsByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5))
    plot_event_chains_by_lp(data, 'Linked')
    data = np.loadtxt("analysisData/globalEventChainsByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5))
    plot_event_chains_by_lp(data, 'Global')
    return

def plot_communication_data():
    data = np.loadtxt("analysisData/numOfLPsToCoverPercentEventMessagesSent.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5,6))
    plot_number_of_lps_sending_remote_events(data)
    histogram_of_lps_sending_95_percent_of_remote_events(data)
    plots_of_lp_event_exchanges()
    plot_lp_degrees()
    
	# plotting these graphs can take some time, leave commented until needed
    #plot_graph_centrality()
    
    # uncomment after getting csv from gephi. Working on a way to do it in desGraphics
    #plot_modularity()
    return

#--------------------------------------------------------------------------------
# the start plotting by analsysis class

plot_event_execution_data()
plot_event_chain_data()
plot_communication_data()

sys.exit()

# i do not believe that the plots below are of any further use.  i am leaving the code
# here, but removing them from our normal plot generation. 

#--------------------------------------------------------------------------------
# histograms of event chains longer than 1

## Local Chains

data = np.loadtxt("analysisData/localEventChainsByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5))
outFile = outDir + 'eventChains-byLP-local-histogram'

percent_long_chains = []
for i in np.arange(len(data)) :
    num_chains = 0
    for j in np.arange(len(data[i])) :
                        num_chains = num_chains + float(data[i,j])
    percent = round(((num_chains - float(data[i,0])) / num_chains) * 100,2)
    percent_long_chains.append(percent)
    
pylab.title('Histogram of Local Event Chains Longer than 1')
pylab.hist(sorted(percent_long_chains), bins=100)
pylab.xlabel('Percent of Total Local Chains of Length > 1')
pylab.ylabel('Number of LPs Containing Said Percent')
display_graph(outFile)

outFile = outDir + 'eventChains-byLP-local-histogram-normalized'
pylab.title('Histogram of Local Event Chains Longer than 1\n(Normalized so Integral will sum to 1)')
pylab.hist(sorted(percent_long_chains), bins=100, normed=True)
pylab.xlabel('Percent of Total Local Chains of Length > 1')
pylab.ylabel('Number of LPs Containing Said Percent\n(Normalized so Integral will sum to 1)')
display_graph(outFile)

## Linked Chains

data = np.loadtxt("analysisData/linkedEventChainsByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5))
outFile = outDir + 'eventChains-byLP-linked-histogram'

percent_long_chains = []
for i in np.arange(len(data)) :
    num_chains = 0
    for j in np.arange(len(data[i])) :
                        num_chains = num_chains + float(data[i,j])
    percent = round(((num_chains - float(data[i,0])) / num_chains) * 100,2)
    percent_long_chains.append(percent)
    
pylab.title('Histogram of Linked Event Chains Longer than 1')
pylab.hist(sorted(percent_long_chains), bins=100)
pylab.xlabel('Percent of Total Linked Chains of Length > 1')
pylab.ylabel('Number of LPs Containing Said Percent')
display_graph(outFile)

outFile = outDir + 'eventChains-byLP-linked-histogram-normalized'
pylab.title('Histogram of Linked Event Chains Longer than 1\n(Normalized so Integral will sum to 1)')
pylab.hist(sorted(percent_long_chains), bins=100, normed=True)
pylab.xlabel('Percent of Total Linked Chains of Length > 1')
pylab.ylabel('Number of LPs Containing Said Percent\n(Normalized so Integral will sum to 1)')
display_graph(outFile)

## Global Chains

data = np.loadtxt("analysisData/globalEventChainsByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5))
outFile = outDir + 'eventChains-byLP-global-histogram'

percent_long_chains = []
for i in np.arange(len(data)) :
    num_chains = 0
    for j in np.arange(len(data[i])) :
                        num_chains = num_chains + float(data[i,j])
    percent = round(((num_chains - float(data[i,0])) / num_chains) * 100,2)
    percent_long_chains.append(percent)
    
pylab.title('Histogram of Global Event Chains Longer than 1')
pylab.hist(sorted(percent_long_chains), bins=100)
pylab.xlabel('Percent of Total Global Chains of Length > 1')
pylab.ylabel('Number of LPs Containing Said Percent')
display_graph(outFile)

outFile = outDir + 'eventChains-byLP-global-histogram-normalized'
pylab.title('Histogram of Global Event Chains Longer than 1\n(Normalized so Integral will sum to 1)')
pylab.hist(sorted(percent_long_chains), bins=100, normed=True)
pylab.xlabel('Percent of Total Global Chains of Length > 1')
pylab.ylabel('Number of LPs Containing Said Percent\n(Normalized so Integral will sum to 1)')
display_graph(outFile)
