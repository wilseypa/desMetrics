
# DES Metrics: Quantitative Metrics from Discrete Event Simulation Models


## Overview

This project is designed to capture run time metrics of Discrete Event Simulation (DES)
models in order to evaluate quantitatively the event dependencies found therein.  While
the initial goals are to learn more about the structure and nature of the simulation
models and their suitability for parallelism.  Opportunities for learning about the
simulation models in order to draw any meaningful conclusions about them will also be
incorporated.  For example, anything we can learn about relationships between simulation
objects that could be used to derive strategies for static analysis and model
transformation/optimization by mechanized tools would also be a highly desirable
objective.  Ideally the approach will capture the profile data in a generic intermediate
representation (e.g., JSON) so that the captured data can analyzed from a large number of
DES simulation tools.  While initially we will develop tools and methods to study
individual simulation models, the ultimate goal will be to figure out how to bring
information together from multiple simulation models to draw conclusions on the common
characteristics of, in particular large, simulation models.  Initially we will focus on
the capture and analysis of event data.

This project is developed from a PDES centric view. In particular, we use term "simulation
object" (also known in the PDES community as Logical Process or LP) throughout which is
not necessarily a strict construct within a sequential simulator.  However, we will stick
with this it will help us to effectively organize and structure our profile data and
analysis activities in ways that will be most useful for us and our parallel simulation
work.


## Strategy

The general strategy will be to instrument existing simulators to capture event data for
offline analysis. Ideally we will use a wide array of different discrete event simulators
containing simulation models written by others (and ideally, domain experts from across
multiple disciplines). This will require that we have source code to the simulator or
programming cooperation from the community of the target simulation engine.

The second (larger) part of this project is the data analysis activity. The objective of
this is to discover meaningful understandings of the characteristics of the simulation
models. While there are specific analysis results that we expect to see in this phase, as
we advance our studies in this space, we are likely to expand the types of analyses that
performed.


## Data Capture

The basic strategy is to profile DES models, capturing information about simulation
objects, event generation, and relations among the events and simulation objects using a
variety of DES simulation engines (sequential or parallel).  The captured data will be
recorded in a JSON format and in addition to event data (recording the time that it is
generated, what object generates it, what time it is destined to be processed, and the
target object to process it), the capture file should have information about the
simulation model, the simulation tool instrumented, the date/time the information was
captured, and any significant configuration/command line arguments that were used to
capture the data.  Ideally all of this information will be captured automatically,
however, manual capture of the non-event data will be acceptable when necessary.

JSON will be used to capture the profile data from the simulation models.  JSON is
selected because it is easily processed, well-understood, and more compact than XML (YAML
could also work, but the warped tools already use JSON for configuration, so it seems best
to leverage our knowledge base with JSON).  

We must be careful in the capturing of this profile data.  What can we do to ensure that
we've captured the correct data and that we've captured all of it (for example, if we
capture event data when the events are generated, we might miss any events that were
initially populated in the input event queues prior to simulation start).  We need to see
what we can do to validate the correctness and completeness of our profile data.  As a
(small) part of this validation step, the analysis tools should perform an initial sanity
check on the data to ensure at least its base integrity.


### The JSON Format

While we should maintain a flexible plan for this format, initially we should capture the
basic information about the tools, platform, and simulation model as well as the run time
data itself.  Initially I suggest we simply capture base data and event information that
looks something like this:

    {
    "simulator_name" : "warped, v5.0",

    "model_name" : "name of the simulation model",

    "capture_date" : "date/time that the profile data was captured",

    // optional, include as needed
    "command_line_arguments" : "significant command line arguments",

    // optional, include as needed
    "configuration_information" : "any significant configuration information, even copying an entire configuration file",

    // optional, probably not needed as we can get this info from the "events" array
    "simulation_objects" : [ "name of simulation object", "...." ],

    "events": [
      {
        "source_object" : "source object name",
        "send_time" : <integer time>,
        "destination_object" : "destination object name",
        "receive_time" : <integer time>,
      },
      ]
    }

Initially we should keep this simple and build on it as we better understand what we
want/need to show in the reporting side of this.  I have included a copy of the
warpedPingPong.json file in this doc directory as an initial example of a JSON file that
we captured from the ping-pong model of warped.


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

3. Irregularity/regularity of events processed by the various simulation objects.

4. Lookahead (minimum, maximum, median, mode):
   4a. Across the entire model, and 
   4b. Per simulation object.

5. Event communication density between simulation objects (probably displayed as a heatmap):
   5a. Measured as frequency
   5b: Measured as a function of frequency/delta of send/receive times

6. Interesting cycles between simulation objects.  Basically can we uncover any
   interesting patterns in the populations ot events and simulation objects?  

I am also developing a separate markdown file (file indexTemplate.markdown in this
directory) that we can use as the basis for (i) defining the specific graphs to begin
implementing and (ii) that will also serve as the basis for viewing the results in a web
browser. 

We might also want to broaden this to look for patterns in models that scale.  For now I
suggest we put this on the back burner and focus on direct analysis for the moment.  

### Approach

For this part of the project, I propose to develop/use a python-based tool to import the
JSON file and produce various graphs/data for analysis.  I do not care if we use the
python tool to simply produce ASCII tabular data (and, optionally, csv files) that can be
processed by gnuplot scripts or other tools (but given the size of the problem, it is
critical that the plotting event be scriptable), or if we generate the graphical results
(as pdf) directly from the python analysis program.  Using pdf as scalar vector format for
our graphical data should make it easy to render in computer displays, webpages, and
papers.  I also propose that we use a markdown file to organize the data into webpage
images with affiliated descriptions so that the reader will have a solid understanding of
what each image is actually depicting.

Lastly, I propose that we use rigid directory structuring to maintain our JSON source,
markdown source, and output graphics.  Initially I propose the following structure

```AsciiDoc
<name of simulation tool> 
      |
<name of simulation model>  ... <additional directories for each model studied in that tool> 
      | 
<graphs> <json file> <markdown file> <markdown output> 
      | 
<many output files from our analysis tool> 
```

I will develop a separate example of a file that we might use as the markdown driver.

I already have a representative JSON file from the ping-pong model in new warped.  We can
begin working with that immediately.  I expect to instrument ROSS next to capture run time
data from it.  We need to look for discrete event simulation engines/simulation models
that we can instrument.

