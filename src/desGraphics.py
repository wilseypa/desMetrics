import json
import pylab
import seaborn

from collections import Counter

# read the json file
json_data = open("analysisResults.json")
data = json.load(json_data)

#--------------------------------------------------------------------------------
# NOTES:
#    1. we should probably perform outlier removal to sharpen the presentation.


#--------------------------------------------------------------------------------
# plot the number of events that are available by simulation cycle

pylab.plot(data["events_available_by_sim_cycle"])
pylab.ylabel('Number of Events Available for Exec')
pylab.xlabel('Simulation Cycle')
pylab.show()

#--------------------------------------------------------------------------------
# show the times that X events are available in a simulation cycle

# count the number of time that X events occur
events_avail = Counter(data["events_available_by_sim_cycle"])
# put the result into x and y vectors for plotting
tuple_events = []
for tmp in events_avail.keys() :
	tuple_events.append((tmp,events_avail[tmp]))
x, y = zip(*tuple_events)

pylab.bar(x, y)
pylab.ylabel('Number of Occurrences')
pylab.xlabel('Number of Events Available')
pylab.show()

#--------------------------------------------------------------------------------
# plot the percent of local events executed by each LP

local_events = []
for tmp in data["lps"] :
    local = float(tmp["local_events_processed"])
    remote = float(tmp["local_events_processed"] + tmp["remote_events_processed"])
    percent = round((local / remote) * 100,2)
    local_events.append(percent)

print local_events

pylab.plot(sorted(local_events))
pylab.ylabel('% of Executed events that were self generated')
pylab.show()
