#import sys
import pylab
import seaborn as sns
import numpy as np
#from collections import Counter

#for arg in sys.argv:
#    print arg

#--------------------------------------------------------------------------------
# plot the number of events that are available by simulation cycle

data = np.loadtxt("analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", skiprows=2)

pylab.title('Cycle by Cycle Record of the Number of Events Available for Execution')
pylab.plot(data)
pylab.ylabel('Number of Events Available for Exec (log scale to minimize outlier dominance)')
pylab.yscale('log')
pylab.xlabel('Simulation Cycle')
pylab.show()
#pylab.savefig('filename.svg')
#pylab.savefig('filename.pdf')

#--------------------------------------------------------------------------------
# plot a histogram of the number of simulation cycles that X events are available

# NOTE: how do we decide how to size the std deviations argument for outlier rejection??

# reject outliers from the mean
def reject_outliers(data, m=8):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

pylab.title('Number of Simulation Cycles that X Events are Available (outliers removed)')
pylab.hist(reject_outliers(data))
pylab.xlabel('Number of Events')
pylab.show()

#--------------------------------------------------------------------------------
# plot the percent of total events executed by each LP sorted and partitioned into local/remote events

# skip the first column of LP names
data = np.loadtxt("analysisData/eventsExecutedByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3))

# sort the data by the total events executed
sorted_data = data[data[:,2].argsort()]

# need a vector of values for the for the x-axis
x_index = np.arange(len(data))

pylab.title('Total (Local/Remote) Events processed by each LP')
pylab.bar(x_index, sorted_data[:,0], color='r', width=1.0, linewidth=0, label="Local")
pylab.bar(x_index, sorted_data[:,1], color='b', width=1.0, linewidth=0, label="Remote", bottom=sorted_data[:,1])
pylab.legend()
pylab.show()

#--------------------------------------------------------------------------------
# histogram the percent of local events executed by each LP

local_events = []
for i in np.arange(len(data)) :
    local = float(data[i,0])
    total = float(data[i,2])
    percent = round((local / total) * 100,2)
    local_events.append(percent)

pylab.title('Histogram of % of Events that were Self (by the LP) Generated')
pylab.hist(sorted(local_events))
#pylab.xlabel('Percent of Executed Events that were Locally Generated')
pylab.show()

#--------------------------------------------------------------------------------
# display graphs of the event chain summaries

data = np.loadtxt("analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter = ",", skiprows=2)

bar_width = .2

pylab.title('Number Local, Linked, and Global Chains of length X')
pylab.bar(data[:,0], data[:,1], bar_width, color='b', label="Local")
pylab.bar(data[:,0] + bar_width, data[:,2], bar_width, color='g', label="Linked")
pylab.bar(data[:,0] + bar_width + bar_width, data[:,3], bar_width, color='r', label="Global")
pylab.xticks(data[:,0] + bar_width, ('1', '2', '3', '4', '>=5'))
pylab.legend()
pylab.show()

#--------------------------------------------------------------------------------
# histogram/heatmap of the number of messages exchanged between LP (colums: senders, rows: receivers)

#pylab.title('Heatmap of total messages exchanged between each pair of LPs')
#pylab.imshow(data["totals_of_events_exchanged_between_lps"], cmap='Greens', interpolation="none")
#pylab.colorbar()
#pylab.show()



