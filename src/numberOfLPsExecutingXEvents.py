#!/usr/bin/pypy
import json,sys,bz2,time

# Have to download python3-matplotlib


start = time.clock()
with  bz2.BZ2File(sys.argv[1],'r') as bz2_file:
	data = bz2_file.read()
	data = json.loads(data)


# with  lzma.open(sys.argv[1],"rt") as lzma_file:
# 	data = json.load(lzma_file)

number_events = len(data["events"])	

# Sanity check
print("Running sanity check...")
error = False
for i in range(number_events):
	if data["events"][i][1] > data["events"][i][3]:
		print ("\nError: event %d (event 0 is the first event) send_time is less than receive_time" % i)
		print(data["events"][i])
		error = True
print("Sanity check complete.")

if (error == True):
	sys.exit()

objectList = []		
destList = []		# Store the Objects who are classified as a dest on each event

for i in range(number_events):
	if(objectList.__contains__(data["events"][i][0]) == False):
		objectList.append(data["events"][i][0])			
	destList.append(data["events"][i][2])			

numberOfEventsByObject = []			# Store the number of Events by Oject

for i in range(len(objectList)) :
	numberOfEventsByObject.append(destList.count(objectList[i]))	# Store the number of Events by Oject

xRangeMin = min(numberOfEventsByObject)
xRangeMax = max(numberOfEventsByObject)

numberOfSimObjBy_X_Event = []

for i in range(len(numberOfEventsByObject)):
	numberOfSimObjBy_X_Event.append(numberOfEventsByObject.count(numberOfEventsByObject[i]))

yRangeMin = min(numberOfSimObjBy_X_Event)
yRangeMax = max(numberOfSimObjBy_X_Event)
X = numberOfEventsByObject



your_json = '[ '
for i in range(len(objectList)):
	abc = '{"eventsByObject": %d, "LPsExecuting": %d },' % (numberOfEventsByObject[i],numberOfSimObjBy_X_Event[i])
	if(i == len(objectList)-1):
		abc = '{"eventsByObject": %d, "LPsExecuting": %d }' % (numberOfEventsByObject[i],numberOfSimObjBy_X_Event[i])
	your_json = your_json + abc

abc = ']'

# print(your_json)

your_json = your_json + abc
parsed = json.loads(your_json)

f1=open('./numberOfLPsExecutingXEvents.json', 'w+')
f1.write(json.dumps(parsed, indent=1, sort_keys=True))


print("File numberOfLPsExecutingXEvents.json generated.")
elapsed = (time.clock() - start)
print(" Time to excute program : %f" % elapsed)
# plt.ylabel('Number of LPs Executing X Events')
# plt.xlabel('Number Of Events Executed')
# plt.xlim(xRangeMin, 1.01*xRangeMax)
# if(xRangeMax == xRangeMin):
# 	plt.xlim(0, 1.01*xRangeMax)
	
# plt.ylim(yRangeMin, 1.01*yRangeMax)
# if(yRangeMax == yRangeMin):
# 	plt.ylim(0, 1.01*yRangeMax)
	

# your_json = '[{"key": "Local Events Executed", "values": [ '
# for i in range(len(objectList)):
# 	abc = '["%s", %d],' % (objectList[i],localEvents[i])
# 	if(i == len(objectList)-1):
# 		abc = '["%s", %d]' % (objectList[i],localEvents[i])
# 	your_json = your_json + abc

# abc = ']},{"key": "Remote Events Executed", "values": [ '

# your_json = your_json + abc

# for i in range(len(objectList)):
# 	abc = '\n["%s", %d],' % (objectList[i],remoteEvents[i])
# 	if(i == len(objectList)-1):
# 		abc = '\n["%s", %d]' % (objectList[i],remoteEvents[i])
# 	your_json = your_json + abc

# abc = "]}]"

# your_json = your_json + abc
# # print(your_json)
# parsed = json.loads(your_json)
# f1=open('./testfile.json', 'w+')
# # print ( json.dumps(parsed, indent=1, sort_keys=True) )
# f1.write(json.dumps(parsed, indent=1, sort_keys=True))

# plt.bar(X, numberOfSimObjBy_X_Event)
# plt.show()
