
# Gephi Documentation for Making Communication Graphs


## Overview

In order to help see and recognize communication patterns in the Discrete Event Simulation
(DES) models, the software Gephi is used to visualize network graphs created from
csv2graphml.py. Gephi is an open source and free graph visualization software, which can be
downloaded [here](https://gephi.org/), as well as finding more information and a quick
start tutorial.

Documented below are the steps used in order to create the communication graphs. If graphs
are already available, and all that is needed is to visualize them, then continue on to the
section __Vizualizing in Gephi__. If graphs need to be created from the csv files generated
by desAnalysis, then continue on to the secion __Creating a Graph__.

## Creating a graph

In order to create a graph from the csv file in desAnalysis, the _csv2graphml.py_ utility
program can be used, which takes a csv files provided to the program and writes out a
graphml file.

_csv2graphml.py_ requires the python networkx which can be downloaded using pip:
<br>`pip install networkx`<br>

Documentation on networkx for the latest release can be found
[here](https://networkx.readthedocs.io/en/stable/)

To use _csv2graphml.py_, type into the command line:
<br>`python csv2graphml.py inFile.csv outFileName`<br>

So far, this program is mainly used for the csv _eventsExchanged-remote.csv_ generated from
desAnalysis, and may not work properly on other csv files. With that being said, as long as
the csv data starts after two rows, the first two columns contain the source and target
nodes, and the third column contains weights, it should still function properly. The
program creates a nodes list, edge list, and weight list, as well as writing to a graphml
file. More information can be found inside the program itself. After writing out to a
graphml file, it can be used to visualize with tools such as graphviz, or with software
such as gephi.

## Visualizing in Gephi

This section goes over the steps necessary to visualize a graph, as well as detailing some
other options that can be used.

  + Once a graph file has been created, open up gephi and import a graph file using
  file->open

  + Go to choose a layout in the bottom left and select OpenOrd. Click run.

  _Different layouts can be chosen here, but OpenOrd seems to work best for large
  graphs created by the trace files. It is relatively quick and usually gives a nice graph
  without too much or too little overlap in the nodes. The default values for the layout
  are also pretty good and can be left alone. One thing that may want to be changed is the
  random seed at the bottom if trying to duplicate results._

  + In the graph window, click the gear symbol on the left, second from the bottom. This is
  were a heatmap can be created. In the upper left of the graph window click _Configure_
  next to Mouse Selection. This decides the size of the mouse to select nodes. Choose a
  size that is large enough to cover the graph. Click and drag over the graph to apply a
  heatmap.

  _Here is also where you can choose the color of the gradient for the heatmap, as well as
  how it is colored, either in a gradient or a palette. One other note here is if the mouse
  does not seem to be moving in the graph window after choosing a mouse size, you may just
  have to click away so the Diameter disappears_

  + On the right side, different filters can be applied to the graph. Drag a desired filter
  from the library to Queries below. Edges->Edge Weight is a good one to apply to filter
  edges to a desired range. If using a graph file from _csv2graphml.py_, this will filter
  edges based on number of events sent. Another good filter to use is Topology->Degree
  Range, if your graph has many In and Out Degrees. To view what the graph looks like with
  filters, click Filter in the bottom left.

  _After choosing any number of filters desired, if you want to permanently add the
  filtered results to the graph, click on the desired query, click 'export filtered graph
  as a true/false data column' under Filters next to Reset. Be sure to check if the column
  is there by going to the Data Laboratory._

  + In the Statistics tab, next to Filters, you can create other graphs based upon your
  own. To do this, simply click Run next to desired statistic.

  _Average Degree, Avg. Weighted Degree, Network Diameter, and Modularity seem to be the
  most useful. These graphs can also be saved to a file location. One thing to note here is
  that it is possible to change the graph appearance based upon these statistics, detailed
  next. In order to be able to, you must run the desired statistic._

  + In the appearance window in the upper left, you can change the appearance of the graph
  based upon certain attributes or statistics. This can be done for nodes, size and color,
  or for edges, color. For nodes, go to size (next to color palette) and then to Ranking.
  From here node size can be changed based upon attributes. Modularity class, Degree, or
  Weighted Degree are usually good options. A good max size is anywhere between 20-75
  depending on the attribute. Once done, click Apply to see results.

  _Labels can also be customized if there exist any in the graph. Changing the appearance
  based on color work the same way as applying a heatmap. Coloring the Graph by edge
  weight can be done through Edges->Ranking->Weight._

  + Once satisfied with how the graph looks, you can can either export it as a graph file
  or as an image by File->Export.

  _If exporting a graph file for use in other software or tools, make sure to export any
  desired filters to a column first._


__Additional Notes on Gephi:__
+ _To switch the background of the graph in order to see more clearly what is going on,
click the light bulb in the bottom left of the graph window. Right click to choose a
color._
+ _To reset all colors on the graph, click the icon below the magnifying glass on the
bottom left of the graph layout._

## Final Notes

So far, creating graphs with 10k nodes and 30k+ edges, and even up to 100k+ nodes and
edges, looks nice using the steps detailed above in Gephi. However, some of the csv files
generated by desAnalysis are very large and contain well over a million nodes and even
more edges. With this many nodes and edges, the file generated may even be too large for
Gephi to handle, and if it is able to handle it, may lead to less than desirable results.

With that being said, we are looking into ways to cut down on some of the data for these
large files, perhaps even taking a slice of it, while still correctly representing the
larger dataset that comes from the DES models.
