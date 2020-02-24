#!/usr/bin/python
# 
# # Sean Kane
# desMetrics Project
# 1/16/2020

import os, sys, json
import numpy as np
import matplotlib as mpl
import pylab
import argparse
import itertools
from palettable.colorbrewer.qualitative import Set1_9
from palettable.tableau import Tableau_20
import collections
import networkx as nx

#### Global Variables ####
total_num_of_sim_cycles = 0 # need this as a global variable for the axis printing function
xLabelOffsetVal = 0 # need this as a global variable for x-axis printing function on trimmed plots



# Helper Functions
def display_plot(fileName):
  # Function to save the plot
  print(f"\t\tCreating pdf graphic of: {fileName}")
  pylab.savefig(fileName+".pdf", bbox_inches='tight')
  pylab.clf()
  return

def reject_outliers(data, m=2):
  out_data = data[abs(data - np.mean(data)) < m * np.std(data)]
  if len(out_data > 0): 
    return out_data
  return data

# Function to remove the first and last 1% of events as outliers
def reject_first_last_outliers(data):
  trim_length = int((len(data)+1)/100)
  return data[trim_length-1 : len(data)-trim_length]

def toPercent(x, pos=0):
  return '%.1f%%'%(100*x)

def toPercentOfTotalSimCycles(x, pos=0):
  return '%.1f%%'%((100*x)/total_num_of_sim_cycles)

def xlabelsOffset(x, pos=0):
  return int(x+xLabelOffsetVal)

def setNumOfSimCycles(x):
    global total_num_of_sim_cycles
    total_num_of_sim_cycles = x
    return

def setxLabelOffset(x):
    global xLabelOffsetVal
    xLabelOffsetVal = x
    return

def percent_of_LP_events_local(data):
    local_events = []
    for i in np.arange(len(data)) :
        local = float(data[i,0])
        total = float(data[i,2])
        percent = round((local / total) * 100,2)
        local_events.append(percent)
    return local_events

def histograms_of_events_exec_by_lp(model_dirs):
  global plot_title
  global x_axis_label
  global y_axis_label
  global colors
  global model_totals

  model_names = []
  model_data = []

  x_axis_label = 'Percent of Total Events that are Local'
  y_axis_label = 'Number of LPs'


  for model in model_dirs:
    out_dir = plotsDir + model[0]

    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    model_names.append(out_dir[1])
    print(f"\tWorking on: {model[0]}")
    raw_data = np.loadtxt(model[0] + "/analysisData/eventsExecutedByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3))
    pylab.hist(sorted(percent_of_LP_events_local(raw_data)), bins=100)
    display_plot(out_dir + "/localEventsAsPercentofTotalByLP-histogram")

    plot_title = 'Local and Remote Events Execute by the LPs'
    x_axis_label = 'Number of Events'
    y_axis_label = 'Number of LPs'
    out_file = out_dir + '/localandRemoteEventsExecute-histogram-stacked'
    pylab.hist((raw_data[:, 0], raw_data[:1]), histtype='barstacked', label=('Local', 'Remote'), color=(colors[0], colors[1]), bins=100)
    display_plot(out_file)
  
  return

def profile_of_local_events_by_lp(model_dirs):
  global plot_title
  global x_axis_label
  global y_axis_label
  global colors
  global model_totals

  model_names = []
  model_data = []
  profile_local_events = []
  num_models = 0

  x_axis_label = 'Percent of Total Events that are Local'
  y_axis_label = 'Number of LPs'


  for model in model_dirs:
    out_dir = plotsDir + model[0]

    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    model_names.append(out_dir[1])
    print(f"\tWorking on: {model[0]}")
    raw_data = np.loadtxt(model[0] + "/analysisData/eventsExecutedByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3))
    plot_title = 'Locally Generated Events'
    x_axis_label = 'LPs (sorted by percent local)'
    y_axis_label = 'Percent of Total Executed (avg={:2})'.format(np.mean(percent_of_LP_events_local(raw_data)))
    profile_local_events.append(sorted(percent_of_LP_events_local(raw_data)))
    plot_data([model_names[num_models]], [profile_local_events[num_models]], 0, out_dir + "/percentOfExecutedEventsThatAreLocal")

    # Histogram
    x_axis_label = 'Percent of Local Events Executed'
    y_axis_label = 'Number of LPs'
    pylab.hist(sorted(percent_of_LP_events_local(raw_data)))
    display_plot(out_dir + '/percentOfExecutedEventsLocal-histogram')

    num_models += 1

# -------------------------------------------------------------------------
# Display graph of the event chain summaries
def plot_event_chain_summaries(model_dirs):
  global plot_title
  global x_axis_label
  global y_axis_label
  global colors
  global model_totals

  for model in model_dirs:
    out_dir = plotsDir + model[0]

    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    plot_title = 'Number of Event Chains of length X'
    raw_data = np.loadtxt(model[0] + "/analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter=',', skiprows=2)
    out_file = out_dir + '/eventChainSummary-individual'
    bar_width = .3
    pylab.bar(raw_data[:, 0], raw_data[:,1], bar_width, color=colors[0], label='Local')
    pylab.bar(raw_data[:, 0] + bar_width, raw_data[:,2], bar_width, color=colors[1], label='Linked')
    pylab.bar(raw_data[:,0] + bar_width + bar_width, raw_data[:,3], bar_width, color=colors[2], label="Global")
    pylab.xticks(raw_data[:, 0] + bar_width, ('1', '2', '3', '4', '>=5'))
    pylab.legend(loc='best')
    x_axis_label = 'Chain Length'
    y_axis_label = 'Total Chains of Length X Found'
    display_plot(out_file)

    # Cumulative event chains (ie, for chains of length 2, also count longer chains)
    plot_title = 'Cumulative Number Event Chains of Length X'
    out_file = out_dir + '/eventChainSummary-cumulative'
    for i in range(len(raw_data)-2, -1, -1):
      for j in range(1, len(raw_data[0])): 
        raw_data[i, j] += raw_data[i+1, j]
    pylab.bar(raw_data[:, 0], raw_data[:,1], bar_width, color=colors[0], label='Local')
    pylab.bar(raw_data[:, 0] + bar_width, raw_data[:,2], bar_width, color=colors[1], label='Linked')
    pylab.bar(raw_data[:,0] + bar_width + bar_width, raw_data[:,3], bar_width, color=colors[2], label="Global")
    pylab.xticks(raw_data[:, 0] + bar_width, ('1', '>=2', '>=3', '>=4', '>=5'))
    pylab.legend(loc='best')
    x_axis_label = 'Chain Length'
    y_axis_label = 'Total Chains of Length >= X Found'
    display_plot(out_file)

# -------------------------------------------------------------------------
# Display pie charts of the event chain summaries

def plot_event_chain_summaries_pie_charts(out_dir, data, type):
  global plot_title
  global x_axis_label
  global y_axis_label
  global colors
  global model_totals

  plot_title = 'Distribution of %s Event Chains\n' % type
  outFile = out_dir + '/eventChainSummary-pie-chart-%s'%type
  labels = '1', '2', '3', '4', '>=5'
  percentages = data.astype(float)/float(np.sum(data))
  pylab.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%')
  pylab.axis('equal')
  display_plot(outFile)
  return

def plot_percent_of_events_in_event_chains(out_dir, data, total_events_of_class, type):
  global plot_title
  global x_axis_label
  global y_axis_label
  global colors
  global model_totals

  plot_title = 'Percent of Events in %s Event Chains\n' % type
  out_file = out_dir + '/eventChainEventTotals-pie-chart-%s'%type
  labels = '1', '2', '3', '4', '>=5'
  # convert the counts of chains to counts of events 
  data[1] = data[1] * 2
  data[2] = data[2] * 3
  data[3] = data[3] * 4
  data[4] = 0
  data[4] = total_events_of_class - np.sum(data)
  percentages = data.astype(float)/float(total_events_of_class)
  pylab.pie(percentages, labels=labels, colors=colors, autopct='%1.1f%%')
  pylab.axis('equal')
  display_plot(out_file)
  return

# plot event chains by LP
def plot_event_chains_by_lp(out_dir, data, type):
  global plot_title
  global x_axis_label
  global y_axis_label
  global colors
  global model_totals

  plot_title = '%s Event Chains by LP (individually sorted)' % type
  x_axis_label = 'LPs'
  y_axis_label = 'Number of Chains'

  out_file = out_dir + '/eventChains-byLP-%s'%type
  sorted_data = data[data[:,0].argsort()]
  pylab.plot(sorted_data[:,0], label='Len=1')
  sorted_data = data[data[:,1].argsort()]
  pylab.plot(sorted_data[:,1], label='Len=2')
  sorted_data = data[data[:,2].argsort()]
  pylab.plot(sorted_data[:,2], label='Len=3')
  sorted_data = data[data[:,3].argsort()]
  pylab.plot(sorted_data[:,3], label='Len=4')
  sorted_data = data[data[:,4].argsort()]
  pylab.plot(sorted_data[:,4], label='Len>=5')
  pylab.tick_params(axis='x',labelbottom='off')
  pylab.legend(loc='best')
  display_plot(out_file)
  return

def plot_event_chain_data_pies(model_dirs):
  global plot_title
  global x_axis_label
  global y_axis_label
  global colors
  global model_totals

  model_names = []
  model_data = []
  event_chains_data = []
  num_models = 0

  for model in model_dirs:
    out_dir = plotsDir + model[0]
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    json_data = open(model[0] + "/analysisData/modelSummary.json")
    model_summary = json.load(json_data)
    total_events = model_summary["event_data"]["total_events"]

    data = np.loadtxt(model[0] + "/analysisData/eventChainsSummary.csv", dtype=np.intc, delimiter = ",", skiprows=2)
    # plot chain data by chain count (chains of length n count as 1)
    plot_event_chain_summaries_pie_charts(out_dir, data[:,1], 'Local')
    plot_event_chain_summaries_pie_charts(out_dir, data[:,2], 'Linked')
    plot_event_chain_summaries_pie_charts(out_dir, data[:,3], 'Global')
    # plot chain data by event count (chains of length n count as n)
    total_local_events = np.sum(np.loadtxt(model[0] + "/analysisData/eventsExecutedByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2))[:,0])
    plot_percent_of_events_in_event_chains(out_dir, data[:,1], total_local_events, 'Local')
    plot_percent_of_events_in_event_chains(out_dir, data[:,2], total_local_events, 'Linked')
    plot_percent_of_events_in_event_chains(out_dir, data[:,3], total_events, 'Global')
    data = np.loadtxt(model[0] + "/analysisData/localEventChainsByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5))
    plot_event_chains_by_lp(out_dir, data, 'Local')
    data = np.loadtxt(model[0] + "/analysisData/linkedEventChainsByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5))
    plot_event_chains_by_lp(out_dir, data, 'Linked')
    data = np.loadtxt(model[0] + "/analysisData/globalEventChainsByLP.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5))
    plot_event_chains_by_lp(out_dir, data, 'Global')
  
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
def plot_data(sampleNames, data_samples, x_range, nameOfAnalysis):
  global plot_title
  global x_axis_label
  global y_axis_label
  global colors

  for y_axis_type, plot_type in itertools.product(('linear', 'log'), ('line', 'scatter')):

    if args.skip_scatter_plots and plot_type == 'scatter': 
      # Skip these
      continue

    if args.no_plot_titles:
      pylab.title('')
    else:
      pylab.title(plot_title)

    pylab.xlabel(x_axis_label)
    pylab.ylabel(y_axis_label)
    pylab.yscale(y_axis_type)

    alpha_initial = 1.0
    alpha_subsequent = 0.25

    color_index = 0
    alpha_value = alpha_subsequent

    for (i, data) in enumerate(data_samples):
      if x_range > len(data):
        print(f'Insufficient data to support plotting points in requested range {sampleNames[i]}. Desired range / data available: {x_range} / {len(data)} ')
        continue

      if i > len(sampleNames):
        print(f'More data_samples than sampleNames')
        continue
      
      if x_range == 0:
        x_index = np.arange(len(data))
      else:
        x_index = np.arange(len(data)).astype(float)/float(len(data))*x_range
      if plot_type == 'line':
        pylab.plot(x_index, data, color=colors[color_index], alpha=alpha_value, label=sampleNames[i])
      else:
        pylab.scatter(x_index, data, marker='o', color=colors[color_index], alpha=alpha_value, label=sampleNames[i])
      color_index = (color_index + 1) % len(colors)
      alpha_value = alpha_subsequent

    display_plot(f'{nameOfAnalysis}_{y_axis_type}_{plot_type}')
  return

def events_available(model_dirs):
  global plot_title
  global x_axis_label
  global y_axis_label
  global colors
  global model_totals

  model_names = []
  model_data = []
  normalized_events_avail_sim_cycle = []
  trimmed_data = []
  num_models = 0

  for model in model_dirs:
    out_dir = plotsDir + model[0]

    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    model_names.append(out_dir[1])
    print(f"\tWorking on: {model[0]}")
    raw_data = np.loadtxt(model[0] + "/analysisData/eventsAvailableBySimCycle.csv", delimiter=',', comments='#')
    model_data.append(np.array(sorted(raw_data, reverse=True)))

    plot_title = 'Events Available by Simulation Cycle'
    x_axis_label = 'Simulation Cycle'
    y_axis_label = 'Number of Events'
    plot_data([model_names[num_models]], [model_data[num_models]], 0, out_dir + "/eventsAvailableBySimCycle_raw")

    plot_title = ''
    x_axis_label = 'Sim Cycles (sorted)'
    y_axis_label = 'Number of Events'
    plot_data([model_names[num_models]], [model_data[num_models]], 0, f"{out_dir}/eventsAvailableBySimCycle_sorted")

    y_axis_label = 'Percent of LPs with Events Available'
    normalized_events_avail_sim_cycle.append((model_data[num_models] / float(model_totals[model[0]][0]))*100.0)
    plot_data([model_names[num_models]], [normalized_events_avail_sim_cycle[num_models]], 0, f"{out_dir}/eventsAvailableBySimCycle_sorted_normalized")

    normalized_events_avail_sim_cycle.append((model_data[num_models]/float(model_totals[model[0]][0]))*100.0)
    plot_data([model_names[num_models]], [normalized_events_avail_sim_cycle[num_models]], 0, f"{out_dir}/eventsAvailableBySimCycle_sorted_normalized")
    
    # Trimmed data
    plot_title = 'Events Available by Simulation Cycle Trimmed'
    y_axis_label = 'Number of Events'
    trimmed_data.append(reject_first_last_outliers(model_data[num_models]))
    plot_data([model_names[num_models]], [trimmed_data[num_models]], 0, f"{out_dir}/eventsAvailableBySimCycle-trimmed")
    
    # Outliers Removed
    plot_title = 'Events Available by Simulation Cycle (Outliers Removed)'
    plot_data([model_names[num_models]], [reject_outliers(model_data[num_models])], 0, f"{out_dir}/eventsAvailableBySimCycle-outliersRemoved")

    num_models += 1

  print("\tCreating Summary Graphic for All Models (Events Available by Sim Cycle)")
  plot_title = '% of LPs with Events Available (sorted)'
  x_axis_label = ''
  y_axis_label = '% of LPs with Events Available'
  plot_data(model_names, normalized_events_avail_sim_cycle, 100, plotsDir + "/eventsAvailableBySimCycle_sorted_normalized")

  return

def total_events_processed(model_dirs):

  global plot_title
  global x_axis_label
  global y_axis_label
  global colors

  model_names = []
  model_data = []
  normalized_events_processed_by_lp = []
  num_models = 0

  print('Plotting Total Events Processed Data')

  for model in model_dirs:
    out_dir = plotsDir + model[0]

    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    model_names.append(model[1])
    print(f"\tWorking on {model[0]}")
    model_data.append(np.loadtxt(model[0] + "/analysisData/totalEventsProcessed.csv", delimiter = ",", comments="#", usecols=(1,2,3,4,5)))

    max_events_proc = np.max(model_data[num_models][:, 0])
    normalized_events_processed_by_lp.append(sorted(model_data[num_models][:,0].astype(float) / max_events_proc, reverse=True))

    plot_title = ''
    x_axis_label = 'LPs'
    y_axis_label = 'Events Processed Relative to Max Processed by any LP'
    plot_data([model_names[num_models]], [normalized_events_processed_by_lp[num_models]], 0, f"{out_dir}/eventsProcessedNormalizedToMax")

  print('\tCreating Summary Graphic of All Models (eventsProcessedNormalized)')
  plot_title = 'Total Events Processed Normalized to the Max Processed'
  y_axis_label = 'Events Processed as a Percentage of Max By Any LP'
  plot_data(model_names, normalized_events_processed_by_lp, 100, f"{plotsDir}/eventsProcessedNormalizedToMax")

  # Average timestamp delta
  normalized_timestamp_delta = []

  for (i, model) in enumerate(model_dirs):
    print(f"\tWorking on normalized Average Timestamp Delta for: {model_dirs[i, 0]}")

    max_timestamp_delta = max(model_data[i][:, 2])
    normalized_timestamp_delta.append(sorted(model_data[i][:,3].astype(float) / float(max_timestamp_delta), reverse=True))
    out_dir = plotsDir + model_dirs[i, 0]

    plot_title = ''
    x_axis_label = 'LPs'
    y_axis_label = r'$\frac{\mathrm{ave(ReceiveTime - SendTime)}}{\mathrm{max(ReceiveTime-SendTime)}}$'
    plot_data([model_names[i]], [normalized_timestamp_delta[i]], 0, f"{out_dir}/normalizedAverageTimestampDelta")

  print('\tCreating Summary graphic of all models (Normalized Average TimeStamp Deltas)')
  plot_data(model_names, normalized_timestamp_delta, 100, f"{plotsDir}/normalizedAverageTimestampDelta")
  return

def event_time_deltas(model_dirs):
  global plot_title
  global x_axis_label
  global y_axis_label
  global colors
  global model_totals

  model_names = []
  model_data = []
  num_models = 0
  normalized_time_delta_means = []
  std_dev_receive_timedelta_means = []
  coeff_var_receivetime_delta_means = []
  times_lp_event_had_same_receivetime = []
  normalized_time_interval_to_execute_100 = []
  coeff_variation_samples = []

  print('\tPlotting Event Time Deltas data')

  for model in model_dirs:
    out_dir = plotsDir + model[0]

    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    model_names.append(model[1])
    print(f'\tWorking on: {model[1]}')
    raw_data = (np.loadtxt(model[0] + "/analysisData/eventReceiveTimeDeltasByLP.csv", delimiter = ",", comments="#", usecols=range(1,18)))

    model_data.append(raw_data[raw_data[:, 4]>0])
    
    normalized_time_delta_means.append(sorted(model_data[num_models][:,3]/np.mean(model_data[num_models][:,3]), reverse=True))
    std_dev_receive_timedelta_means.append(sorted(np.sqrt(model_data[num_models][:,4]), reverse=True))
    coeff_var_receivetime_delta_means.append(sorted(np.sqrt(model_data[num_models][:,4])/model_data[num_models][:,3], reverse=True))
    times_lp_event_had_same_receivetime.append(sorted(model_data[num_models][:,5], reverse=True))
    normalized_time_interval_to_execute_100.append(sorted(model_data[num_models][:,7]/np.mean(model_data[num_models][:,7]), reverse=True))

    plot_title = 'Event Receive Time delta by LP'
    x_axis_label = 'LPs (sorted)'
    y_axis_label = r'$(mean \; of \; LP_i \; \mathbf{/} \; (mean \; of \; all \; LPs)$'
    plot_data([model_names[num_models]],[normalized_time_delta_means[num_models]], 0, out_dir + "/relativeReceiveTimeDeltaMeans")

    plot_title = 'STDDEV of event receive time delta by LP'
    x_axis_label = 'LPs (sorted)'
    y_axis_label = 'STDDEV'
    plot_data([model_names[num_models]],[std_dev_receive_timedelta_means[num_models]], 0, out_dir + "/relativeReceiveTimeStdDev")

    # now we'll plot the coefficient of variation 
    plot_title = 'Coefficient of Variation of Receive Time Deltas'
    x_axis_label = 'LPs (sorted)'
    y_axis_label = r'$\mathrm{(std \; deviation \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; LP_i)}$'
    plot_data([model_names[num_models]],[coeff_var_receivetime_delta_means[num_models]], 0, out_dir + "/coefficientOfVariationOfReceiveTimeDeltasMean")

    # next up is the number of times events occur at the same time
    x_axis_label = 'LPs (sorted)'
    y_axis_label = 'Times an event had the same receive time'
    plot_data([model_names[num_models]],[times_lp_event_had_same_receivetime[num_models]], 0, out_dir + "/timesAnEventHadSameReceiveTime")

    # now to the real challenge; reporting the sampling results (number of events in a time interval sampled at random

    # ok, so first let's plot the mean time interval for 100 events that was used for each LP.  we'll actually normalize this to the mean
    plot_title = 'Relative Time Interval that an LP executes 100 events'
    x_axis_label = 'LPs (sorted)'
    y_axis_label = r'$\mathrm{(mean \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; all \; LPs)}$'
    plot_data([model_names[num_models]],[normalized_time_interval_to_execute_100[num_models]], 0, out_dir + "/relativeTimeIntervalToExecute100Events")

    # what do to for the sampled event totals in the mean subinterval??  let's try this, but i don't think it's gonna work
    plot_title = 'Coefficient Of Variation of Sampled intervals'
    x_axis_label = 'LPs (sorted)'
    y_axis_label = r'$\mathrm{(std \; deviation \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; LP_i)}$'
    coeffOfVariation = []
    # foreach LP in the model compute the coefficient of variation of the samples
    for j in model_data[num_models]:
        coeffOfVariation.append(np.std(j[8:])/np.mean(j[8:]))
    coeff_variation_samples.append(sorted(coeffOfVariation, reverse=True))
    plot_data([model_names[num_models]],[coeff_variation_samples[num_models]], 0, out_dir + "/coefficientOfVariationOfSamples")

    num_models += 1
  

  normalized_time_interval_to_execute_100 = []
  coeff_variation_samples = []
  print("\tCreating Summary Graphic of All Models (LP receiveTime delta data)")

  plot_title = 'Event receive time delta by LP'
  x_axis_label = 'LPs (sorted)'
  y_axis_label = r'$\mathrm{(mean \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; all \; LPs)}$'
  plot_data(model_names, normalized_time_delta_means, 100, plotsDir + "/relativeReceiveTimeDeltaMeans")
  
  plot_title = 'STDDEV of event receive time delta by LP'
  x_axis_label = 'LPs (sorted)'
  y_axis_label = 'STDDEV'
  plot_data(model_names, std_dev_receive_timedelta_means, 100, plotsDir + "/relativeReceiveTimeStdDev")
  
  # now we'll plot the coefficient of variation 
  plot_title = 'Coefficient of Variation of Receive Time Deltas'
  x_axis_label = 'LPs (sorted)'
  y_axis_label = r'$(std \; deviation \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; LP_i)}$'
  plot_data(model_names, coeff_var_receivetime_delta_means, 100, plotsDir + "/coefficientOfVariationOfReceiveTimeDeltasMean")

  # next up is the number of times events occur at the same time
  plot_title = ''
  x_axis_label = 'LPs (sorted)'
  y_axis_label = 'Times an event had the same receive time'
  plot_data(model_names, times_lp_event_had_same_receivetime, 100, plotsDir + "/timesAnEventHadSameReceiveTime")

  # ok, so first let's plot the mean time interval for 100 events that was used for each LP.  we'll actually normalize this to the mean
  plot_title = 'Relative Time Interval that an LP executes 100 events'
  x_axis_label = 'LPs (sorted)'
  y_axis_label = r'$\mathrm{(mean \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; all \; LPs)}$'
  plot_data(model_names, normalized_time_interval_to_execute_100, 100, plotsDir + "/relativeTimeIntervalToExecute100Events")

  plot_title = 'Coefficient Of Variation of Sampled intervals'
  x_axis_label = 'LPs (sorted)'
  y_axis_label = r'$\mathrm{(std \; deviation \; of \; LP_i) \; \mathbf{/} \; (mean \; of \; LP_i)}$'
  coeffOfVariation = []
  plot_data(model_names, coeff_variation_samples, 100, plotsDir + "/coefficientOfVariationOfSamples")

  return

# ----------------------------------------------------------------------------------------
# Communication Graphs plots

def plot_number_of_lps_sending_remote_events(data, out_dir):
  pylab.plot(data[data[:,5].argsort()][:,5], color=colors[0], label = '100% of total remote events')
  pylab.plot(data[data[:,4].argsort()][:,4], color=colors[1], label = '95% of total remote events')
  pylab.plot(data[data[:,1].argsort()][:,1], color=colors[2], label = '75% of total remote events')
  pylab.tick_params(axis='x',labelbottom='off')
  pylab.legend(loc='best')
  display_plot(out_dir)
  return

def histogram_of_lps_sending_95_percent_of_remote_events(data, out_dir):
  pylab.hist(data[:,5], bins=20)
  display_plot(out_dir)
  return

def plots_of_lp_event_exchanges(out_dir, dir):
  global plot_title
  global x_axis_label
  global y_axis_label
  global colors
  global model_totals

  plot_title = 'Remote Events Sent Between LPs'
  x_axis_label = 'Number of Events'
  y_axis_label = 'Number of Events Sent'
  data = np.loadtxt(dir + "/analysisData/eventsExchanged-remote.csv", dtype=np.float_, delimiter = ",", skiprows=2, usecols=(2,3,4,5))
  outFile = out_dir + '/countsOfLpToLpEventExchanges'
  pylab.plot(data[data[:,0].argsort()][:,0].astype(np.intc))
  pylab.tick_params(axis='x',labelbottom='off')
  display_plot(outFile)

  
  plot_title = 'Timestamp Deltas of Remote Events'
  x_axis_label = ''
  y_axis_label = 'Timestamp Delta (Receive Time - Send Time)'
  outFile = out_dir + '/timeStampDeltasOfRemoteEvents'
  stride = max(int(max(len(data[:,1]),len(data[:,2]),len(data[:,3]))/20),1)
  pylab.plot(data[data[:,1].argsort()][:,1], color=colors[0], label="Minimum", marker='o', markevery=stride)
  pylab.plot(data[data[:,3].argsort()][:,3], color=colors[1], label="Average", marker='x', markevery=stride)
#    pylab.plot(data[data[:,2].argsort()][:,2], color=colors[2], label="Maximum", marker='*', markevery=stride)
  pylab.tick_params(axis='x',labelbottom='off')
  pylab.ylim([-.1,np.amax(data[:,3].astype(np.intc))+1])
#    pylab.yscale('log')
  pylab.legend(loc='best')
  display_plot(outFile)

  plot_title = 'Histogram of Timestamp Deltas of Remote Events'
  x_axis_label = 'Timestamp Delta (ReceiveTime - SendTime)'
  y_axis_label = 'Number of LPs'
  outFile = out_dir + '/timeStampDeltasOfRemoteEvents-hist'
  pylab.hist((data[:,1],data[:,3],data[:,2]), label=('Minimum', 'Average', 'Maximum'), color=(colors[0], colors[1], colors[2]), bins=10)
  pylab.legend(loc='best')
  display_plot(outFile)

  return

# plots in and out degree of LPs (degree = # of LPs an LP sends to or receives from)
def plot_lp_degrees(out_dir, data_dir, total_lps):
	outFile = out_dir + 'countsOfDegreeLPbyLP'
	data = np.loadtxt(dir + "/analysisData/eventsExchanged-remote.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(0,1,2))

	# data structures for holding LPs sent, received, and the number of events. weights are events sent
	inLP = [x[0] for x in data]
	outLP = [x[1] for x in data]
	weights = [int(x[2]) for x in data]
	inDegree = collections.Counter()
	outDegree = collections.Counter()
	inCount = collections.Counter()
	outCount = collections.Counter()
	eventsSent = collections.Counter()
	eventsCount = collections.Counter()
	eventsAvg = {}

	# count the in and out degree of a given LP, and total the events sent
	for i in np.arange(len(data)):
		inDegree[inLP[i]] += 1
		outDegree[outLP[i]] += 1
		eventsSent[outLP[i]] += weights[i]

	# count the number of LPs who have the same degree and their events
	for i in np.arange(len(inDegree)):
		inCount[inDegree[i]] += 1
		outCount[outDegree[i]] += 1
		eventsCount[inDegree[i]] += eventsSent[i]
		
	# take the average events sent by LP degree
	for i in inCount:
		eventsAvg[i] = float(eventsCount[i]) / int(inCount[i])
	
	# get all x values for the easier graphing
	keyList = sorted(list(set(inCount.keys() + outCount.keys())))
	for key in keyList:
		if key not in inCount:
			inCount[key] = 0
		if key not in outCount:
			outCount[key] = 0
		if key not in eventsAvg:
			eventsAvg[key] = 0
			
	# these sort their respective dictionaries by their keys, and store their values in a list
	sort_inCount = [value for (key, value) in sorted(inCount.items())]
	sort_outCount = [value for (key, value) in sorted(outCount.items())]
	sort_eventsAvg = [value for (key, value) in sorted(eventsAvg.items())]
	
	fig, ax1 = pylab.subplots()
	bar_width = 0.30
	
	# plot in and out degrees and have average events show up in the legend
	ax1.plot(np.nan, '-', marker='o', color=colors[2], label = "average events") 
	ax1.bar(np.arange(len(keyList)), sort_inCount, width=bar_width, label='In-Degree', color=colors[0])
	ax1.bar(np.arange(len(keyList))+bar_width, sort_outCount, width=bar_width, label='Out-Degree',color=colors[1])
	pylab.xticks(np.arange(len(keyList))+bar_width,keyList)
	ax2 = ax1.twinx()
	# plot average events
	ax2.plot(np.arange(len(keyList)),sort_eventsAvg,ms=5, marker='o',color=colors[2], label="average events")
	ax2.grid(b=False)
	ax1.set_xlabel('LP Degree Counts')
	ax1.set_ylabel('Number of LPs(Total=%s)' % "{:,}".format(total_lps))
	ax2.set_ylabel('Events Sent')
	ax2.get_yaxis().get_major_formatter().set_scientific(False)
	pylab.title('LP Connectivity')
	ax1.legend(loc='upper right')
	display_plot(outFile)
	return


# plots both betweenness and closeness centralities. this is an expensive computation and may fail for very large graphs
def plot_graph_centrality(G, out_dir):
  global plot_title
  global x_axis_label
  global y_axis_label
  global colors
  global model_totals

  plot_title = 'Betweenness Centrality of LP by LP Communication'
  x_axis_label = 'Betweenness Centrality Value'
  y_axis_label = 'Frequency'
  outFile = out_dir + 'betweeness_centrality'

	# plot betweenness centrality 
  centrality = nx.betweenness_centrality(G)
  fig, ax = pylab.subplots()
	# bins vary by graphs, need to find a better way to make them
  ax.hist(centrality.values(), bins=10)
  pylab.legend(loc='best')
  display_plot(outFile)

# plots modularity of a graph
def plot_modularity(G, out_dir):
  outFile = out_dir + 'communities'
  modularity = collections.Counter()
  mod = community.best_partition(G)
  modList = mod.values()

  for i in np.arange(len(modList)):
    modularity[modList[i]] += 1

  mean = np.mean(modularity.values())
  std_dev = np.std(modularity.values())
  start = min(modularity.keys(), key=int)
  end = max(modularity.keys(), key=int)
  fig, ax = pylab.subplots()
  
  ax.scatter(modularity.keys(),modularity.values(),color=colors[0])
  pylab.axhline(mean,color=colors[2],label="mean")
  pylab.axhline(mean+std_dev,color=colors[1],label="standard deviation")
  pylab.axhline(mean-std_dev,color=colors[1])
  ax.set_ylabel('Number of LPs')
  ax.set_xlabel('Modularity Class')
  ax.ticklabel_format(useOffset=False)
  ya = ax.get_yaxis()
  ya.set_major_locator(pylab.MaxNLocator(integer=True))
  pylab.xticks(np.arange(start, end+1,10)) # change 10 to 1 (or smaller number) if # of communities is small
  pylab.title('Communities in LP Communication Graph')
  pylab.legend(loc='best', shadow=True)
  display_plot(outFile)
  return	

def create_comm_graph(file):
  data = np.loadtxt(file, dtype=np.intc, delimiter = ",", skiprows=2, usecols=(0,1,2))
  nodes = [x[0] for x in data]
  edges = [x[1] for x in data]
  weights = [int(x[2]) for x in data]
  G = nx.Graph()
  for i in np.arange(len(data)):
    G.add_node(int(nodes[i]))
    G.add_edge(int(nodes[i]), int(edges[i]), weight=int(weights[i]))
  return G

def plot_communication_data(model_dirs):
  global plot_title
  global x_axis_label
  global y_axis_label
  global colors
  global model_totals

  model_names = []
  model_data = []
  num_models = 0

  for model in model_dirs:
    data = np.loadtxt(model[0] + "/analysisData/numOfLPsToCoverPercentEventMessagesSent.csv", dtype=np.intc, delimiter = ",", skiprows=2, usecols=(1,2,3,4,5,6))
    out_dir = plotsDir + model[0]
    jsonData = open(dir[0] + "/analysisData/modelSummary.json")
    modelSummary = json.load(jsonData)
    model_totals[dir[0]] = [modelSummary["total_lps"], modelSummary["event_data"]["total_events"]]

    plot_title = 'Number of LPs Sending Remote Events (sorted)'
    x_axis_label = 'Receiving LP (Total={:,})'.format(modelSummary["total_lps"])
    y_axis_label = 'Number of SEnding LPs'
    out_file = out_dir + '/numberOfSendingLPs'
    plot_number_of_lps_sending_remote_events(data, out_file)
    
    out_file = out_dir + '/sending95PercentOfRemoteEvents-hist'
    plot_title = 'How many LPs are involved in sendign 955 of remote events'
    x_axis_label = 'Number of Sending LPs'
    y_axis_label = 'Frequency'
    histogram_of_lps_sending_95_percent_of_remote_events(data, out_file)

    plots_of_lp_event_exchanges(out_dir, model[0])
    
    # plot_lp_degrees(out_dir, model[0], modelSummary["total_lps"])
    
    Graph = create_comm_graph(model[0] + "/analysisData/eventsExchanged-remote.csv")
    # plotting these graphs can take some time, leave commented until needed
    plot_graph_centrality(Graph, out_dir)
    plot_modularity(Graph, out_dir)
    return


# process the arguments on the command line
argparser = argparse.ArgumentParser(description='Generate various graphics to show desMetrics results for a collection of DES models.')
argparser.add_argument('--modelDirList', default='./allModelDirs.csv', help='CSV file naming the subdirs to walk (default: ./allModelDirs.csv')
argparser.add_argument('--plotsDir', default='./plotsDir/', help='Directory to write output files (default: ./plotsDir)')
argparser.add_argument('--no_legend', help='Turn off the legend in all plots', action="store_true")
argparser.add_argument('--gen_all', help='Generate all known plots.', action="store_true", default=True)
argparser.add_argument('--gen_events_available', help='Generate events available plots.', action="store_true")
argparser.add_argument('--gen_events_processed', help='Generate total events processed plots.', action="store_true")
argparser.add_argument('--gen_event_time_deltas', help='Generate LP event time deltas plots.', action="store_true")
argparser.add_argument('--skip_scatter_plots', help='Do not generate scatter plots (saving time).', action="store_true")
argparser.add_argument('--no_plot_titles', help='Do not generate plot titles.', action="store_true")
argparser.add_argument('--histograms_of_events_exec_by_lp', help='Histograms of Events Executed By each LP', action='store_true')
argparser.add_argument('--event_chain_summaries', help='Plots of event chain summaries', action='store_true')
argparser.add_argument('--communication-data', action='store_true')
args = argparser.parse_args()

if args.gen_events_available or args.gen_events_processed or args.gen_event_time_deltas:
  args.gen_all = False

colors = Tableau_20.mpl_colors

plotsDir = args.plotsDir
if not os.path.exists(plotsDir):
  os.makedirs(plotsDir)

mpl.style.use('seaborn-whitegrid')

plot_title = ''
x_axis_label = ''
y_axis_label = ''

# All the directories to walk 
model_directories = np.loadtxt(args.modelDirList, dtype=np.str, delimiter=',', comments='#')

# Event and LP data for each simulation run
model_totals = {}
for dir in model_directories:
  jsonData = open(dir[0] + "/analysisData/modelSummary.json")
  modelSummary = json.load(jsonData)
  model_totals[dir[0]] = [modelSummary["total_lps"], modelSummary["event_data"]["total_events"]]

if args.gen_all or args.gen_events_available:
  events_available(model_directories)
if args.gen_all or args.gen_events_processed:
  total_events_processed(model_directories)
if args.gen_all or args.gen_event_time_deltas:
  event_time_deltas(model_directories)

if args.gen_all or args.histograms_of_events_exec_by_lp:
  histograms_of_events_exec_by_lp(model_directories)

if args.gen_all or args.event_chain_summaries:
  plot_event_chain_summaries(model_directories)
  plot_event_chain_data_pies(model_directories)
  profile_of_local_events_by_lp(model_directories)
if args.communication_data:
  plot_communication_data(model_directories)