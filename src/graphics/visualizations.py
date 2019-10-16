#!/usr/bin/env python3

## plot the desAnalysis files for PADS19 manuscript

import os
import sys
import json
import numpy as np
#from sklearn.metrics import mean_squared_error
import matplotlib as mpl
# Force matplotlib to not use any Xwindows backend
mpl.use('Agg')
from palettable.colorbrewer.qualitative import Set1_9
from palettable.tableau import Tableau_20
import pylab
import argparse
import itertools
from math import sqrt

# define a function to display/save the pylab figures.
def display_plot(fileName) :
    print(f"\tCreating pdf graphic of: {fileName}")
    pylab.savefig(fileName + ".pdf", bbox_inches='tight')
    pylab.clf()
    return

#### function to generate overlay plots analysis data from the various samples

# generate four plots alternating yAxis to linear/log scales and plot type to line/scatter
#
#  sampleNames: string names of samples for printing/legend labels
#
#  data: nxm matrix: n samples, m data values for each sample
#
#  xRange: x-axis bounding range for plotting the m data values (if non-zero, data will be plotted in the
#          bounding range 0..xRange; if zero, the data values will be plotted as an integer enumeration to the
#          x-axis
#
# nameOfAnalysis: string to use to name files and denote the type of desAnalysis result these plots are for
#
def plot_data(sampleNames, data, xRange, nameOfAnalysis) :

    global plotTitle
    global xAxisLabel
    global yAxisLabel
    global colors
        
    for yAxisType, plotType in itertools.product(('linear', 'log'), ('line', 'scatter')) :

        if args.skip_scatter_plots and plotType == 'scatter' : continue

        if args.no_plot_titles :
            pylab.title('')
        else :
            pylab.title(plotTitle)
        pylab.xlabel(xAxisLabel)
        pylab.ylabel(yAxisLabel)
        pylab.yscale(yAxisType)
        
        alphaInitial = 1.0
        alphaSubsequent = 0.25

        colorIndex = 0
        alphaValue = alphaInitial

        for i in range(len(data)) :
            # if we don't have enough data points to plot; skip
            if xRange > len(data[i]) :
                print(f"Insufficient data to support plotting points in the requested range {sampleNames[i]}. Desired Range/data available: {str(xRange)} / {str(len(data[i]))}")
                continue
            if xRange == 0 : xIndex = np.arange(len(data[i]))
            else : xIndex = np.arange(len(data[i])).astype(float)/float(len(data[i]))*xRange
            if plotType == 'line' :
                pylab.plot(xIndex, data[i], color=colors[colorIndex], alpha=alphaValue, label=sampleNames[i], linewidth=2)
            else :
                pylab.scatter(xIndex, data[i], marker = 'o', color=colors[colorIndex], alpha=alphaValue, label=sampleNames[i])
            colorIndex = (colorIndex + 1) % len(colors)
            alphaValue = alphaSubsequent

#        pylab.legend(loc='best')
        display_plot(f"{nameOfAnalysis}_{yAxisType}_{plotType}")
    return

#### let's look at the eventsAvailableBySimCycle.csv file
def events_available(modelDirs) :

    global plotTitle
    global xAxisLabel
    global yAxisLabel
    global colors
    global modelTotalsByName
        
    modelNames = []
    modelData = []
    normalizedEventsAvailableBySimCycle = []
    numModels = 0

    print("Plotting Events Available Data by Sim Cycle")

    for i in modelDirs :

        outDir = plotsDir + i[0]
        if not os.path.exists(outDir): os.makedirs(outDir)

        modelNames.append(i[1])
        print(f"\tWorking on: {i[0]}")
        rawData = np.loadtxt(i[0] + "/analysisData/eventsAvailableBySimCycle.csv", delimiter = ",", comments="#")
        modelData.append(np.array(sorted(rawData, reverse=True)))

# removing, it just doesn't add anything
#        plotTitle = 'Events Available by Simulation Cycle'
#        xAxisLabel = 'Simulation Cycle'
#        yAxisLabel = 'Number of Events'
#        plot_data([modelNames[numModels]], [modelData[numModels]], 0, outDir + "/eventsAvailableBySimCycle_raw")

        plotTitle = ''
        xAxisLabel = 'Sim Cycles (sorted)'
        yAxisLabel = 'Number of Events'
        plot_data([modelNames[numModels]], [modelData[numModels]], 0, outDir + "/eventsAvailableBySimCycle_sorted")

        plotTitle = ''
        xAxisLabel = 'Sim Cycles (sorted)'
        yAxisLabel = 'Percent of LPs with Events Available'
        normalizedEventsAvailableBySimCycle.append((modelData[numModels]/float(modelTotalsByName[i[0]][0]))*100.0)
        plot_data([modelNames[numModels]], [normalizedEventsAvailableBySimCycle[numModels]], 0, outDir + "/eventsAvailableBySimCycle_sorted_normalized")
        numModels = numModels + 1

    print(f"\tCreating Summary Graphic for All Models (Events Available by Sim Cycle)")
    plotTitle = '% of LPs with Events Available (sorted)'
    xAxisLabel = ''
    yAxisLabel = '% of LPs with Events Available'
    plot_data(modelNames, normalizedEventsAvailableBySimCycle, 100, plotsDir + "/eventsAvailableBySimCycle_sorted_normalized")

    return

#### let's look at the totalEventsProcessed.csv file
def total_events_processed(modelDirs) :

    global plotTitle
    global xAxisLabel
    global yAxisLabel
    global colors
        
    modelNames = []
    modelData = []
    normalizedEventsProcessedByLP = []
    numModels = 0

    ## first let's load the data and plot the normalized total events processed data

    print("Plotting Total Events Processed Data")

    for i in modelDirs :

        outDir = plotsDir + i[0]
        if not os.path.exists(outDir): os.makedirs(outDir)

        modelNames.append(i[1])
        print(f"\tWorking on: {i[0]}")
        modelData.append(np.loadtxt(i[0] + "/analysisData/totalEventsProcessed.csv", delimiter = ",", comments="#", usecols=(1,2,3,4,5)))

        maxEventsProcessedByAnyLP = np.max(modelData[numModels][:,0])
        normalizedEventsProcessedByLP.append(sorted(modelData[numModels][:,0].astype(float)/maxEventsProcessedByAnyLP, reverse = True))

        plotTitle = ''
        xAxisLabel = 'LPs'
        yAxisLabel = 'Events Processed relative to Max Processed by any LP'
        plot_data([modelNames[numModels]], [normalizedEventsProcessedByLP[numModels]], 0, outDir + "/eventsProcessedNormalizedToMax")
        numModels = numModels + 1
    
    print(f"\tCreating Summary Graphic of All Models (eventsProcessedNormalizedToMax)")
    plotTitle = 'Total Events Processed Normalized to the Max Processed by any LP'
    xAxisLabel = 'LPs'
    yAxisLabel = 'Events processed as a % of max processed by any LP'
    plot_data(modelNames, normalizedEventsProcessedByLP, 100, plotsDir + "/eventsProcessedNormalizedToMax")

    ## now let's look at the average timestamp delta

    normalizedTimestampDelta = []

    for index in range(len(modelNames)) :

        print(f"\tWorking on normalizedAverageTimestampDelta for: {modelDirs[index,0]}")

        # find the LP with the largest timestamp delta in the model
        maxTimestampDelta = max(modelData[index][:,2])
        normalizedTimestampDelta.append(sorted(modelData[index][:,3].astype(float)/float(maxTimestampDelta), reverse = True))
        outDir = plotsDir + modelDirs[index,0]

        plotTitle = ''
        xAxisLabel = 'LPs'
        yAxisLabel = r'$\frac{\mathrm{ave(ReceiveTime - SendTime)}}{\mathrm{max(ReceiveTime-SendTime)}}$'
        plot_data([modelNames[index]], [normalizedTimestampDelta[index]], 0, outDir + "/normalizedAverageTimestampDelta")
    
    print(f"\tCreating Summary Graphic of All Models (Normalized Average TimeStamp Deltas)")
    # keep the same labeling
#    plotTitle = ''
#    xAxisLabel = 'LPs'
#    yAxisLabel = 'ave(ReceiveTime-SendTime)/max(ReceiveTime-SendTime)'
    plot_data(modelNames, normalizedTimestampDelta, 100, plotsDir + "/normalizedAverageTimestampDelta")

    return

#### let's look at the eventReceiveTimeDeltasByLP.csv file
def event_time_deltas(modelDirs) :

    global plotTitle
    global xAxisLabel
    global yAxisLabel
    global colors
        
    modelNames = []
    modelData = []
#    normalizedEventsProcessedByLP = []
    numModels = 0
    normalizedTimeDeltaMeans = []
    stdDevOfReceiveTimeDeltaMeans = []
    coefficientOfVariationOfReceiveTimeDeltaMeans = []
    timesAnLPEventHadSameReceiveTime = []
    normalizedTimeIntervalToExecute100Events = []
    coeffOfVariationOfSamples = []

    ## first let's load the data and plot the normalized total events processed data

    print(f"Plotting Event Time Deltas data")

    for i in modelDirs :

        outDir = plotsDir + i[0]
        if not os.path.exists(outDir): os.makedirs(outDir)

        modelNames.append(i[1])
        print(f"\tWorking on: {i[0]}")
        rawData = (np.loadtxt(i[0] + "/analysisData/eventReceiveTimeDeltasByLP.csv", delimiter = ",", comments="#", usecols=range(1,18)))

        modelData.append(rawData[rawData[:,4]>0])
        # first let's plot the mean time delta of each LP

        normalizedTimeDeltaMeans.append(sorted(modelData[numModels][:,3]/np.mean(modelData[numModels][:,3]), reverse=True))
        stdDevOfReceiveTimeDeltaMeans.append(sorted(np.sqrt(modelData[numModels][:,4]), reverse=True))
        coefficientOfVariationOfReceiveTimeDeltaMeans.append(sorted(np.sqrt(modelData[numModels][:,4])/modelData[numModels][:,3], reverse=True))
        timesAnLPEventHadSameReceiveTime.append(sorted(modelData[numModels][:,5], reverse=True))
        normalizedTimeIntervalToExecute100Events.append(sorted(modelData[numModels][:,7]/np.mean(modelData[numModels][:,7]), reverse=True))

        plotTitle = 'Event receive time delta by LP'
        xAxisLabel = 'LPs (sorted)'
        yAxisLabel = r'$(mean \; of \; LP_i \; \mathbf{/} \; (mean \; of \; all \; LPs)$'
        plot_data([modelNames[numModels]],[normalizedTimeDeltaMeans[numModels]], 0, outDir + "/relativeReceiveTimeDeltaMeans")

        # now let's examine the variance: first we'll plot the std dev

        plotTitle = 'STDDEV of event receive time delta by LP'
        xAxisLabel = 'LPs (sorted)'
        yAxisLabel = 'STDDEV'
        plot_data([modelNames[numModels]],[stdDevOfReceiveTimeDeltaMeans[numModels]], 0, outDir + "/relativeReceiveTimeStdDev")

        # now we'll plot the coefficient of variation 
        plotTitle = 'Coefficient of Variation of Receive Time Deltas'
        xAxisLabel = 'LPs (sorted)'
        yAxisLabel = r'$\mathrm{(std \; deviation \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; LP_i)}$'
        plot_data([modelNames[numModels]],[coefficientOfVariationOfReceiveTimeDeltaMeans[numModels]], 0, outDir + "/coefficientOfVariationOfReceiveTimeDeltasMean")

        # next up is the number of times events occur at the same time
        plotTitle = ''
        xAxisLabel = 'LPs (sorted)'
        yAxisLabel = 'Times an event had the same receive time'
        plot_data([modelNames[numModels]],[timesAnLPEventHadSameReceiveTime[numModels]], 0, outDir + "/timesAnEventHadSameReceiveTime")

        # now to the real challenge; reporting the sampling results (number of events in a time interval sampled at random

        # ok, so first let's plot the mean time interval for 100 events that was used for each LP.  we'll actually normalize this to the mean
        plotTitle = 'Relative Time Interval that an LP executes 100 events'
        xAxisLabel = 'LPs (sorted)'
        yAxisLabel = r'$\mathrm{(mean \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; all \; LPs)}$'
        plot_data([modelNames[numModels]],[normalizedTimeIntervalToExecute100Events[numModels]], 0, outDir + "/relativeTimeIntervalToExecute100Events")

        # what do to for the sampled event totals in the mean subinterval??  let's try this, but i don't think it's gonna work
        plotTitle = 'Coefficient Of Variation of Sampled intervals'
        xAxisLabel = 'LPs (sorted)'
        yAxisLabel = r'$\mathrm{(std \; deviation \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; LP_i)}$'
        coeffOfVariation = []
        # foreach LP in the model compute the coefficient of variation of the samples
        for j in modelData[numModels] :
            coeffOfVariation.append(np.std(j[8:])/np.mean(j[8:]))
        coeffOfVariationOfSamples.append(sorted(coeffOfVariation, reverse=True))
        plot_data([modelNames[numModels]],[coeffOfVariationOfSamples[numModels]], 0, outDir + "/coefficientOfVariationOfSamples")

        numModels = numModels + 1

    print(f"\tCreating Summary Graphic of All Models (LP receiveTime delta data)")

    plotTitle = 'Event receive time delta by LP'
    xAxisLabel = 'LPs (sorted)'
    yAxisLabel = r'$\mathrm{(mean \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; all \; LPs)}$'
    plot_data(modelNames, normalizedTimeDeltaMeans, 100, plotsDir + "/relativeReceiveTimeDeltaMeans")
    
    plotTitle = 'STDDEV of event receive time delta by LP'
    xAxisLabel = 'LPs (sorted)'
    yAxisLabel = 'STDDEV'
    plot_data(modelNames, stdDevOfReceiveTimeDeltaMeans, 100, plotsDir + "/relativeReceiveTimeStdDev")
    
    # now we'll plot the coefficient of variation 
    plotTitle = 'Coefficient of Variation of Receive Time Deltas'
    xAxisLabel = 'LPs (sorted)'
    yAxisLabel = r'$(std \; deviation \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; LP_i)}$'
    plot_data(modelNames, coefficientOfVariationOfReceiveTimeDeltaMeans, 100, plotsDir + "/coefficientOfVariationOfReceiveTimeDeltasMean")

    # next up is the number of times events occur at the same time
    plotTitle = ''
    xAxisLabel = 'LPs (sorted)'
    yAxisLabel = 'Times an event had the same receive time'
    plot_data(modelNames, timesAnLPEventHadSameReceiveTime, 100, plotsDir + "/timesAnEventHadSameReceiveTime")

    # ok, so first let's plot the mean time interval for 100 events that was used for each LP.  we'll actually normalize this to the mean
    plotTitle = 'Relative Time Interval that an LP executes 100 events'
    xAxisLabel = 'LPs (sorted)'
    yAxisLabel = r'$\mathrm{(mean \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; all \; LPs)}$'
    plot_data(modelNames, normalizedTimeIntervalToExecute100Events, 100, plotsDir + "/relativeTimeIntervalToExecute100Events")

    plotTitle = 'Coefficient Of Variation of Sampled intervals'
    xAxisLabel = 'LPs (sorted)'
    yAxisLabel = r'$\mathrm{(std \; deviation \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; LP_i)}$'
    coeffOfVariation = []
    plot_data(modelNames,coeffOfVariationOfSamples, 100, plotsDir + "/coefficientOfVariationOfSamples")

    return

#######--------------------------------------------------------------------------------



####----------------------------------------------------------------------------------------------------
#### let us begin
####----------------------------------------------------------------------------------------------------

# process the arguments on the command line
argparser = argparse.ArgumentParser(description='Generate various graphics to show desMetrics results for a collection of DES models.')
argparser.add_argument('--modelDirList', default='./allModelDirs.csv', help='CSV file naming the subdirs to walk (default: ./allModelDirs.csv')
argparser.add_argument('--plotsDir', default='./plotsDir/', help='Directory to write output files (default: ./plotsDir)')
argparser.add_argument('--no_legend', help='Turn off the legend in all plots', action="store_true")
argparser.add_argument('--gen_all', help='Generate all known plots.', action="store_true")
argparser.add_argument('--gen_events_available', help='Generate events available plots.', action="store_true")
argparser.add_argument('--gen_events_processed', help='Generate total events processed plots.', action="store_true")
argparser.add_argument('--gen_event_time_deltas', help='Generate LP event time deltas plots.', action="store_true")
argparser.add_argument('--skip_scatter_plots', help='Do not generate scatter plots (saving time).', action="store_true")
argparser.add_argument('--no_plot_titles', help='Do not generate plot titles.', action="store_true")

args = argparser.parse_args()

plotsDir = args.plotsDir
# create a directory to write output graphs
if not os.path.exists(plotsDir): os.makedirs(plotsDir)

# change the plots to a grey background grid w/o solid x/y-axis lines
#mpl.style.use('ggplot')
# change the plots to a white background with grey grids
mpl.style.use('seaborn-whitegrid')

# set colormap and make it the default
# this only has 9 colors
#colors = Set1_9.mpl_colors
# this has 20 colors....
colors = Tableau_20.mpl_colors

# ok, this is an easy way for us to setup labels and pass them across our functions (yea i'm lazy)
plotTitle = ''
xAxisLabel = ''
yAxisLabel = ''

# input the list of model directories that we're working on
## WARNING: this will blow up if there is only one directory in the file; sorry....
modelDirectories = np.loadtxt(args.modelDirList, dtype=np.str, delimiter = ",", comments="#")

# get total number of LPs and events executed for each model
modelTotalsByName = {}
for i in modelDirectories :
    # read the json file
    jsonData = open(i[0] + "/analysisData/modelSummary.json")
    modelSummary = json.load(jsonData)
    modelTotalsByName[i[0]] = [modelSummary["total_lps"], modelSummary["event_data"]["total_events"]]

## ok, let's process the total events processed by LPs in the models

if args.gen_all or args.gen_events_available :
    events_available(modelDirectories)
if args.gen_all or args.gen_events_processed :
    total_events_processed(modelDirectories)
if args.gen_all or args.gen_event_time_deltas :
    event_time_deltas(modelDirectories)
