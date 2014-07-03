#!/usr/bin/python3	
import json,sys,lzma,time
from pprint import pprint
import matplotlib.pyplot as plt


start = time.clock()
# Have to download python3-matplotlib

with  lzma.open(sys.argv[1],"rt") as lzma_file:
	data = json.load(lzma_file)

# with  open('desTraceFile.json') as lzma_file:
# 	data = json.load(lzma_file)

number_events = len(data["events"])	

# Sanity check
error = False
for i in range(number_events):
	if data["events"][i][1] > data["events"][i][3]:
		print ("\nError: event %d (event 0 is the first event) send_time is less than receive_time" % i)
		pprint(data["events"][i])
		error = True

if (error == True):
	sys.exit()

localEvents = []
remoteEvents = []
totalEvents = []

objectList = []		

for i in range(number_events):
	if(objectList.__contains__(data["events"][i][0]) == False):
		objectList.append(data["events"][i][0])			


for i in range(len(objectList)):
	totalEvents.append(0)
	localEvents.append(0)
	remoteEvents.append(0)
	for j in range(number_events):
		if (data["events"][j][0] == objectList[i]):
			totalEvents[i] = totalEvents[i] + 1
			if(data["events"][j][2] == objectList[i]):
				localEvents[i] = localEvents[i] + 1
			else:
				remoteEvents[i] = remoteEvents[i] + 1


your_json = '[{"key": "Local Events Executed", "values": [ '
for i in range(len(objectList)):
	abc = '["%s", %d],' % (objectList[i],localEvents[i])
	if(i == len(objectList)-1):
		abc = '["%s", %d]' % (objectList[i],localEvents[i])
	your_json = your_json + abc

abc = ']},{"key": "Remote Events Executed", "values": [ '

your_json = your_json + abc

for i in range(len(objectList)):
	abc = '\n["%s", %d],' % (objectList[i],remoteEvents[i])
	if(i == len(objectList)-1):
		abc = '\n["%s", %d]' % (objectList[i],remoteEvents[i])
	your_json = your_json + abc

abc = "]}]"

your_json = your_json + abc
# print(your_json)
parsed = json.loads(your_json)

f1=open('./totalEventsExecuted.json', 'w+')
# print ( json.dumps(parsed, indent=1, sort_keys=True) )
f1.write(json.dumps(parsed, indent=1, sort_keys=True))

f,  ax1 = plt.subplots(1, figsize=(10,5))

ax1.bar(range(len(localEvents)), localEvents, label='Local Events', alpha=0.5, color='b')
ax1.bar(range(len(remoteEvents)), remoteEvents, bottom=localEvents, label='Remote Events', alpha=0.5, color='r')
plt.sca(ax1)
plt.legend(loc='upper right')

print("File totalEventsExecuted.json generated.")
elapsed = (time.clock() - start)
print(" Time to excute program : %f" % elapsed)
plt.show()