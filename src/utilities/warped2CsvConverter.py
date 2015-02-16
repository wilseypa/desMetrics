# USAGE:  python warped2CsvConverter.py warpedCsvFileName desMetricsJsonFileName

import time
import sys
import csv

# capture header info
#--------------------------------------------------------------------------------
model_name = raw_input('Enter the name of the simulation model: ')
capture_date = time.strftime("%c") + ' (when captured by warped2CsvConverter.py)'
print 'Capture date is currently: ' + capture_date
input = raw_input('Do you want to enter a different capture date (hit enter if no): ')
if input != '' : capture_date = input
command_line = raw_input('What was the command line invocation for this simulation model: ')

outFile = open(sys.argv[2], 'w')

# print header info
#--------------------------------------------------------------------------------
outFile.write("{\n\"simulator_name\" : \"warped2\",\n")
outFile.write("\"model_name\" : \"" + model_name + "\",\n")
outFile.write("\"capture_date\" : \"" + capture_date + "\",\n")
outFile.write("\"command_line_arguments\" : \"" + command_line + "\",\n")
outFile.write("\"events\" : [\n")

# print event info
#--------------------------------------------------------------------------------
file = open(sys.argv[1])
events = csv.reader(file)
# skip the first line of the csv file
events.next()
separator = ""
for event in events :
    outFile.write(separator + "[\"" + event[0] + "\"," + event[2] + ",\"" + event[1] + "\"," + event[3] + "]\n")
    separator = ","

# print closing
#--------------------------------------------------------------------------------
outFile.write("]\n}")
