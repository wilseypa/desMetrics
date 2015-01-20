import json
import matplotlib.pyplot as plt

from collections import Counter

json_data = open("analysisResults.json")
data = json.load(json_data)

#print (data["num_of_event_chains_of_len_i_plus_1"][1]["linked"])
#print (data["events_available_by_sim_cycle"])

plt.plot(data["events_available_by_sim_cycle"])
plt.ylabel('Number of Events Available for Exec')
plt.xlabel('Simulation Cycle')
plt.show()

#plt.plot(sorted(data["events_available_by_sim_cycle"]))
#plt.ylabel('Number of Events Available for Exec')
#plt.show()


#for num_events, freq in Counter(data["events_available_by_sim_cycle"]).most_common():
#	num_events_available[i] = num_events
#        frequency[i] = freq
#print (num_events_available)
#print (frequency)


#plt.plot(Counter(data["events_available_by_sim_cycle"]).most_common())
plt.ylabel('Number of Occurrences')
plt.xlabel('Number of Events Available for Exec')
plt.show()
