
import os
import numpy as np
from scipy.stats import wasserstein_distance

## let's look at events available

# becuase we know there is startup/teardown costs, we'll trim them out at those costs will not be part of the
# samples; furthermore, we will sort the data as we're really interested in the shape of these curves not the
# instantaneous variations (which is quite rough if you look at the raw data)

data = np.loadtxt("analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc, delimiter = ",", comments="#")

totalEvents = len(data)

# how many events in 1%
numSkippedEvents = int(len(data)*.01)

print "wasserstein against /analysisData/eventsAvailableBySimCycle.csv"
print "to sample, distance"
for x in os.listdir("sampleDir") :
    approxData = np.loadtxt("sampleDir/" + x + "/analysisData/eventsAvailableBySimCycle.csv", dtype=np.intc,
                            delimiter = ",", comments="#") 
    print x + ", %.8f" % wasserstein_distance(
        sorted(data[numSkippedEvents:totalEvents-numSkippedEvents]),
        sorted(approxData))

## now let's look at the summaries of event chains.

# since we don't want to have the size of the measured data to skew the wasserstein distance results, we'll
# work from percentages of the populations in question.  furthermore, we don't really want to see the
# differences by event change length; our real interest is in the impact on the overall event chain
# computation, we will create a vector that includes measures from local, linked, and global chains as one
# vector to be analyzed....
    
print "wasserstein against /analysisData/eventChainsSummary.csv"
data = np.loadtxt("analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter = ",", comments="#")
percentagesOfData = data.astype(float)/float(np.sum(data))

print "to sample, distance"
for x in os.listdir("sampleDir") :
    approxData = np.loadtxt("sampleDir/" + x + "/analysisData/eventChainsSummary.csv", dtype=np.intc,
                            delimiter = ",", comments="#")
    percentagesOfApproxData = approxData.astype(float)/float(np.sum(approxData))
    print x + ", %.8f" % wasserstein_distance(
        np.append(np.append(np.array(percentagesOfData[:,1]),np.array(percentagesOfData[:,2])),np.array(percentagesOfData[:,3])),
                  np.append(np.append(np.array(percentagesOfApproxData[:,1]),np.array(percentagesOfApproxData[:,2])),np.array(percentagesOfApproxData[:,3])))


## now let's look at modularity....

