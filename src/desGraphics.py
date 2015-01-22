import sys
import json
import pylab
import seaborn as sns
import numpy as np
from collections import Counter

#for arg in sys.argv:
#    print arg

# read the json file
json_data = open(sys.argv[1])
data = json.load(json_data)

#--------------------------------------------------------------------------------
# plot the number of events that are available by simulation cycle

pylab.title('Cycle by Cycle Record of the Number of Events Available for Execution')
pylab.plot(data["events_available_by_sim_cycle"])
pylab.ylabel('Number of Events Available for Exec (log scale to minimize outlier dominance)')
pylab.yscale('log')
pylab.xlabel('Simulation Cycle')
pylab.show()
#pylab.savefig('filename.svg')
#pylab.savefig('filename.pdf')

#--------------------------------------------------------------------------------
# plot a histogram of the number of simulation cycles that X events are available

# reject outliers from the mean
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

# reject outliers from the median
def reject_outliers_from_median(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

pylab.title('Number of Simulation Cycles that X Events are Available (outliers removed)')
pylab.hist(reject_outliers(np.asarray(data["events_available_by_sim_cycle"])))
#pylab.ylabel('# Simulation Cycles X Events Available')
pylab.xlabel('Number of Events')
pylab.show()

#--------------------------------------------------------------------------------
# plot the percent of local events executed by each LP

# this plot stinks.  we need to find some way to improve it's presentation.

local_events = []
for tmp in data["lps"] :
    local = float(tmp["local_events_processed"])
    remote = float(tmp["local_events_processed"] + tmp["remote_events_processed"])
    percent = round((local / remote) * 100,2)
    local_events.append(percent)

pylab.plot(sorted(local_events))
pylab.ylabel('% of Executed events that were self generated')
#pylab.ylim(0,100)
pylab.show()

#--------------------------------------------------------------------------------
# histogram the percent of local events executed by each LP

pylab.title('Histogram of % of Events that were Self (by the LP) Generated')
pylab.hist(sorted(local_events))
pylab.ylabel('Frequency')
pylab.ylabel('Percent of Executed Events that were Locally Generated')
#pylab.ylim(0,100)
pylab.show()

#--------------------------------------------------------------------------------
# histogram/heatmap of the number of messages exchanged between LP (colums: senders, rows: receivers)

pylab.title('Heatmap of total messages exchanged between each pair of LPs')
pylab.imshow(data["totals_of_events_exchanged_between_lps"], cmap='Greens', interpolation="none")
pylab.colorbar()
pylab.show()
