#!/usr/bin/pypy
import json,sys,time,bz2
# import matplotlib.pyplot as plt
# import numpy as np
from collections import deque

start = time.clock()
# Have to download python3-matplotlib

# with  lzma.open('warpedPingPong.lzma',"rt") as lzma_file:
# 	data = json.load(lzma_file)

def checkIfQueuesAreEmpty( number, List ):
	for i in range(number):
		if (List[i][0].__len__() != 0):
			return True
	return False

def getMinReceiveTime (List):
	minReceiveTime = -1
	for i in range(len(List)):
		if(List[i][0].__len__() != 0):
			aux = List[i][0].__getitem__(0)
			if(minReceiveTime > aux or minReceiveTime == -1):
				minReceiveTime = aux
	return minReceiveTime

with  bz2.BZ2File(sys.argv[1],'r') as bz2_file:
	data = bz2_file.read()
	# print(type(data))
	# sys.exit()
	data = json.loads(data)

number_events = len(data["events"])	

start = time.clock()
# Sanity check
error = False
for i in range(number_events):
	if data["events"][i][1] > data["events"][i][3]:
		print ("\nError: event %d (event 0 is the first event) send_time is less than receive_time" % i)
		print(data["events"][i])
		error = True

if (error == True):
	sys.exit()


objectList = []		
destList = []		# Store the Objects who are classified as a dest on each event

for i in range(number_events):
	if(objectList.__contains__(data["events"][i][0]) == False):
		objectList.append(data["events"][i][0])			
	destList.append(data["events"][i][2])			



LP = []
for i in range(len(objectList)):
	eventReceiveQueue = deque([0])
	eventSendQueue = deque([0])
	eventReceiveQueue.pop()
	eventSendQueue.pop()
	LP.append((eventReceiveQueue,eventSendQueue))
	for j in range(number_events):
		if data["events"][j][2] == objectList[i]:
			LP[i][0].append(data["events"][j][3])
			LP[i][1].append(data["events"][j][1])

events_available = []
scheduleTimeList = []
total_schedule_cycles = 0;
for i in range(number_events): 
	events_available.append(0)



while (checkIfQueuesAreEmpty(len(objectList) , LP) ):
	schedule_time =  getMinReceiveTime(LP)
	scheduleTimeList.append(schedule_time)
	for j in range(len(objectList)):
		if(LP[j][0].__len__() != 0):
			firstItemSendTime = LP[j][1].__getitem__(0)
			if(firstItemSendTime  <= schedule_time):
				events_available[scheduleTimeList.__len__()-1] = events_available[scheduleTimeList.__len__()-1] + 1 
				
				LP[j][0].popleft()
				LP[j][1].popleft()
	total_schedule_cycles = total_schedule_cycles + 1

eventCyclesWithXEventsAvailable = []
x_axis = []

for i in range(len(set(events_available))):
	count = 0
	for j in range(total_schedule_cycles):
		if (events_available[j] == events_available[i]):
			count = count + 1
	if( eventCyclesWithXEventsAvailable.__contains__(count) == False):
		eventCyclesWithXEventsAvailable.append(count)
		x_axis.append(events_available[i])

print(len(x_axis))
y_axis = []
for i in range(max(x_axis)+1):
	for j in range(len(x_axis)):
		if(i == x_axis[j]):
			print(i)
			y_axis.append(eventCyclesWithXEventsAvailable[j])

x_axis.sort()

average = sum(eventCyclesWithXEventsAvailable) / float(len(eventCyclesWithXEventsAvailable))
print(average)
your_json = '{ '
abc = '"total": %d, "summary": [%d , %d , %d], "values": [ ' % (total_schedule_cycles,min(eventCyclesWithXEventsAvailable),average,max(eventCyclesWithXEventsAvailable))
your_json = your_json + abc
for i in range(len(eventCyclesWithXEventsAvailable)):
	abc = '[%d, %d],' % (x_axis[i],y_axis[i])
	if(i == len(eventCyclesWithXEventsAvailable)-1):
		abc = '[%d, %d]' % (x_axis[i],y_axis[i])
	your_json = your_json + abc

abc = ']}'

# print(your_json)

your_json = your_json + abc
parsed = json.loads(your_json)

f2=open('./eventsCyclesWithXEventsAvailable.json', 'w+')
f2.write(json.dumps(parsed, indent=1, sort_keys=True))


print("File eventsCyclesWithXEventsAvailable.json created.")
elapsed = (time.clock() - start)
print(" Time to excute program : %f" % elapsed)