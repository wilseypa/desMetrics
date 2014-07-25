
# DES Metrics: Quantitative Metrics from Discrete Event Simulation Models


## Overview

This project is designed to capture run time metrics of Discrete Event Simulation (DES)
models in order to evaluate quantitatively the event dependencies found therein.  While
the initial goals are to learn more about the structure and nature of the simulation
models and their suitability for parallelism.  Opportunities for learning about the
simulation models in order to draw any meaningful conclusions about them will also be
incorporated.  For example, anything we can learn about relationships between simulation
LPs that could be used to derive strategies for static analysis and model
transformation/optimization by mechanized tools would also be a highly desirable
objective.  Ideally the approach will capture the profile data in a generic intermediate
representation (e.g., JSON) so that the captured data can analyzed from a large number of
DES simulation tools.  While initially we will develop tools and methods to study
individual simulation models, the ultimate goal will be to figure out how to bring
information together from multiple simulation models to draw conclusions on the common
characteristics of, in particular large, simulation models.  Initially we will focus on
the capture and analysis of event data.

This project is developed from a PDES centric view. In particular, we use term LP (Logical
Process) throughout which is not necessarily a strict construct within a sequential
simulator.  However, we will stick with this it will help us to effectively organize and
structure our profile data and analysis activities in ways that will be most useful for us
and our parallel simulation work.


## Strategy

The general strategy will be to instrument existing simulators to capture event data for
offline analysis. Ideally we will use a wide array of different discrete event simulators
containing simulation models written by others (and ideally, domain experts from across
multiple disciplines). This will require that we have source code to the simulator or
programming cooperation from the community of the target simulation engine.

The second (larger) part of this project is the data analysis activity. The objective of
this is to discover meaningful understandings of the characteristics of the simulation
models.  While there are specific analysis results that we expect to see in this phase, as
we advance our studies in this space, we are likely to expand the types of analyses that
are performed.

To be meaningful, we will need to study large simulation models.  In our preliminary
studies, we have already discovered that the captured data and even the output analysis
files can be quite large (easily reaching multiple GBytes of data.  We will have to learn
how to manage this size problem.   First, we will have to work to determine what amount of
runtime data is actually needed to capture meaningful results that properly characterize
the properties of simulation models.  Second, we may have to adjust how we present our
analysis results so that we have meaningful graphs and visuals to examine.

## Data Capture

The basic strategy is to profile DES models, capturing information about LPs, event
generation, and relations among the events and LPs using a variety of DES simulation
engines (sequential or parallel).  The captured data will be recorded in a JSON format and
in addition to event data (recording the time that it is generated, what LP generates it,
what time it is destined to be processed, and the target LP to process it), the capture
file should have information about the simulation model, the simulation tool instrumented,
the date/time the information was captured, and any significant configuration/command line
arguments that were used to capture the data.  Ideally all of this information will be
captured automatically, however, manual capture of the non-event data will be acceptable
when necessary.

JSON will be used to capture the profile data from the simulation models.  JSON is
selected because it is easily processed, well-understood, and more compact than XML (YAML
could also work, but the warped tools already use JSON for configuration, so it seems best
to leverage our knowledge base with JSON).

We must be careful in the capturing of this profile data.  What can we do to ensure that
we have captured the correct data and that we have captured all of it (for example, if we
capture event data when the events are generated, we might miss any events that were
initially populated in the input event queues prior to simulation start).  We need to see
what we can do to validate the correctness and completeness of our profile data.  As a
(small) part of this validation step, the analysis tools should perform an initial sanity
check on the data to ensure at least its base integrity.

Finally, for instrumenting existing parallel simulation engines, this capture should be
relatively straight forward.  However, for existing sequential models, it may be difficult
to instrument the simulation and isolate the communicating LPs of a simulation model.
However, even for parallel simulation models, we may find that the LPs are artificially
partitioned into larger objects that could actually be broken down into multiple LPs.
This may negatively affect our analysis results.  Not sure what we can do about this, but
it is definitely something that we need to be aware of and to comment on.


### The JSON Format

While we should maintain a flexible plan for this format, initially we should capture the
basic information about the tools, platform, and simulation model as well as the run time
data itself.  Initially I suggest we simply capture base data and event information in the
following generic format: 

    {
    "simulator_name" : "warped, v5.0",

    "model_name" : "name of the simulation model",

    "capture_date" : "date/time that the profile data was captured",

    // optional, include as needed
    "command_line_arguments" : "significant command line arguments",

    // optional, include as needed
    "configuration_information" : "any significant configuration information",

    // optional, probably not needed as we can get this info from the "events" array
    "LPs" : [ "name of LP", "...." ],

    "events": [
      { "sLP":"source LP", "sTS": <send_time>, "rLP":"destination LP", "rTS":<receive_time> },
        "....forall events processed...."
      ]
    }

Initially we should keep this simple and build on it as we better understand what we
want/need to show in the reporting side of this.  I will be placing data files in my UC
webpages (![DES Metrics data files](http://secs.ceas.uc.edu/~paw/research/desMetrics))
from the various simulation tools that we instrument (warning: some of these files may be
quite large).


## Analysis

Ideally I would like to think about analysis in the broadest terms possible and see what
"interesting" information we can glean from the simulation model profile data.  However,
to have a meaningful beginning point, I suggest that we look for data that should be
useful for our work with parallel discrete event simulation (PDES).  To my mind, the most
important first items we should attempt to extract should focus on:

1. Event availability:
   1a. Plot an x/y graph where the x-axis is the number of events and the y-axis is the
       number of times where the corresponding number of events (from the x-axis) are
       available for execution.  The basic goal here is to show the potential parallelism
       in the simulation model.
   1b. Plot the total number of event loop cycles where n (on the x-axis) events are
       available for execution.
   1c. Plot the rate of change of events available for execution at each cycle of the
       execution of the event processing loop.

2. Understanding the path(s) of execution through the model.
   2a. Highlight the critical path of events and any nearby paths.
   2b. Count and show independent event paths in the system.

3. Irregularity/regularity of events processed by the various LPs.  Especially if we can
   capture data to help us develop partitioning strategies.

4. Lookahead (minimum, maximum, median, mode):
   4a. Across the entire model, and 
   4b. Per LP.

5. Event communication density between LPs (probably displayed as a heatmap):
   5a. Measured as frequency
   5b: Measured as a function of frequency/delta of send/receive times

6. Interesting cycles between LPs.  Basically can we uncover any
   interesting patterns in the populations ot events and LPs?  

I am also developing a separate markdown file (file indexTemplate.markdown in this
directory) that we can use as the basis for (i) defining the specific graphs to begin
implementing and (ii) that will also serve as the basis for viewing the results in a web
browser.

We might also want to broaden this to look for patterns in models that scale.  For now I
suggest we put this on the back burner and focus on direct analysis for the moment.  

### Approach

For this part of the project, I propose to develop/use a tool to import the JSON file and
produce various graphs/data for analysis.  The output of this analysis tool will be comma
separated csv files that we can use for graphical visualization of the results.  Because
of the size of these files, it will probably be necessary to use offline visualization
tools to present the data.  Initial experiments with web-based javascript tools (e.g.,
nvd3) have, so far, proven to be inadequate for this task.  In particular I am working up
gnuplot scripts to convert the data to various formats for display in webpages (svg, pdf)
and documents (pdf, eps).   I am also maintaining a file in this directory
(indexTemplate.md) that uses markdown to organize the data into webpage images with
affiliated descriptions so that the reader will have a solid understanding of what each
graphic is actually depicting. 

### Analysis Output Formats

#### LP Based Results

Recording information about the LP total event processing results.  Number of local events (l_ev)
is the number of events generated by the LP for itself.  Number of remote events (r_ev) is the
number of events generated by other LPs for this LP.  Lookahead (lookah) is an array of
(i) the minimum lookahead seen by the LP, (ii) the average lookahead by the LP, and (iii) the
maximum lookahead seen by the LP.  We will add to this format as we better understand the
LP specific data we want to capture.

    [ 
      { 
        "name": "name of the LP", 
        "l_ev": <number of local events>, 
        "r_ev": <number of remote events>, 
        "lookah": [<min>, <average>, <max>] 
      },
      { <repeated for each LP in the simulation> }
    ];

#### Event Chains

**This has all completely changed and needs to be updated**

An event chain is the number of events in an LP that could be executed as a group.  That
is, let t be the timestamp of a simulation cycle, how many events in the list of events
for that LP have (i) send times less than t, (ii) receive times equal or greater than t,
and (iii) that have not already been computed as part of an event chain (global chains are
all events, local chains are only events that were sent by this LP).  Once a chain has
been identified, the next event chain to be examined begins at the next event follwing the
last in the current chain (the simultaneous computation of local and remote event chains
may prove difficult and may need to be done in separate passes over the data).

    {
      "local": [
        [ 1, <number of local chains of length 1> ],
        [ 2, <number of local chains of length 2> ],
        [ ....repeat as necessary....], ],
      "global": [
        [ 1, <number of global chains of length 1> ],
        [ 2, <number of global chains of length 2> ],
        [ ....repeat as necessary....], ],
    }


#### Simulation Cycle Results on Parallelism

Recording results from our analysis of simulation cycles.  I am not sure this will
generalize much.  For now, the format is really quite simple

    {
      "total": <total number of simulation cycles>,
      "summary": [ <min>, <average>, max ],
      "values": [ 
          [ 1,  <number of simulation cycles with 1 event available> ],
          [ 2,  <number of simulation cycles with 2 events available> ],
          [ ....repeat as necessary....]
    }

## Additional Notes

We need to look for discrete event simulation engines/simulation models that we can
instrument.  It may be difficult to capture data from sequential simulators that are not
designed by groups engaged in parallel simulation as they may not easily breakdown into
independent LPs exchanging event data.

