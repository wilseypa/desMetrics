#!/usr/bin/pypy
import json,sys,time
import bz2

start = time.clock()
# Have to download python3-matplotlib

# with  lzma.open(sys.argv[1],"rt") as lzma_file:
# 	data = json.load(lzma_file)

# with  open('desTraceFile.json') as lzma_file:
# 	data = json.load(lzma_file)

# with  tarfile.open(sys.argv[1],"r") as tar_file:
# 	members = tar_file.getmembers()
# 	tarFile = tar_file.extractfile(members[0])
# 	dataRead = tarFile.read()
# 	data = json.loads(dataRead)


# with  gzip.open('desTraceFile.json.gz') as gzip_file:
# 	data = gzip_file.read()
	
# 	data = json.loads(data)


with  bz2.BZ2File(sys.argv[1],'r') as bz2_file:
	data = bz2_file.read()
	# print(type(data))
	# sys.exit()
	data = json.loads(data)


number_events = len(data["events"])	
# Sanity check
error = False
for i in range(number_events):
	if data["events"][i][1] > data["events"][i][3]:
		print ("\nError: event %d (event 0 is the first event) send_time is less than receive_time" % i)
		print(data["events"][i])
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

print("soma: %d" (sum(localEvents) + sum(remoteEvents)))
lookahead = [0,0,0]
your_json = '[ '
for i in range(len(objectList)):
	abc = '{"name": "%s", "l_ev": %d, "r_ev": %d, "l_ahead": [0] },' % (objectList[i],localEvents[i],remoteEvents[i])
	if(i == len(objectList)-1):
		abc = '{"name": "%s", "l_ev": %d, "r_ev": %d, "l_ahead": [0] }' % (objectList[i],localEvents[i],remoteEvents[i])
	your_json = your_json + abc

abc = ']'

your_json = your_json + abc

parsed = json.loads(your_json)

f1=open('./totalEventsExecuted.json', 'w+')
# print ( json.dumps(parsed, indent=1, sort_keys=True) )



f1.write(json.dumps(parsed, indent=1))

# f,  ax1 = plt.subplots(1, figsize=(10,5))

# ax1.bar(range(len(localEvents)), localEvents, label='Local Events', alpha=0.5, color='b')
# ax1.bar(range(len(remoteEvents)), remoteEvents, bottom=localEvents, label='Remote Events', alpha=0.5, color='r')
# plt.sca(ax1)
# plt.legend(loc='upper right')

print("File totalEventsExecuted.json generated.")
elapsed = (time.clock() - start)
print(" Time to excute program : %f" % elapsed)
# plt.show()