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

# def toPercentOfTotalLPs(x, pos=0):
#   return f"{round((100*(x/total_lps)), 3)}"

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
  for i in np.arange(len(data)):
    local = float(data[i, 0])
    total = float(data[i, 2])
    percent = round((local / total) * 100, 2)
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
    raw_data = np.loadtxt(model[0] + "/analysisData/eventsExecutedByLP.csv")
    pylab.hist(sorted(percent_of_LP_events_local(raw_data)), bins=100)
    display_plot(out_dir + "/localEventsAsPercentofTotalByLP-histogram")

    plot_title = 'Local and Remote Events Execute by the LPs'
    x_axis_label = 'Number of Events'
    y_axis_label = 'Number of LPs'
    out_file = out_dir + '/localandRemoteEventsExecute-histogram-stacked'
    pylab.hist((raw_data[:, 0], raw_data[:1]), histtype='barstacked', label=('Local', 'Remote'), color=(colors[0], colors[1]), bins=100)
    display_plot(out_file)
  
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
    plot_data( [model_data[num_models]], [reject_outliers(model_names[num_models])], 0, f"{out_dir}/eventsAvailableBySimCycle-outliersRemoved")


    num_models = num_models + 1

    


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

    # TODO - Left off here Line 254 in visualizations.py

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

if args.histograms_of_events_exec_by_lp:
  histograms_of_events_exec_by_lp(model_directories)
  