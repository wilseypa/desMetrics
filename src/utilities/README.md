
# Various scripts in support of the desMetrics project.

## A collection of scripts for building, running, and analyzing DES simulation models.

These scripts are designed to fail badly.  If anything dies at a midpoint, the entire script should
fail.  We do not want to be using any results from broken/partial builds/runs/analysis steps in this
process.  In general these scripts are built so that they execute from the subdirectory where you
want the output results to be placed.  

Naming conventions:
  * build<toolName> (e.g., buildWarped2): these scripts will retrieve and build the source code for
    the simulation kernel and simlation models for that simulation kernel.
  * run<toolName> (e.g., runWarped2): these scripts will run the simulation models and capture both
    runtime profile data from google-proftools as well as capturing event trace data for the
    desMetrics tools.  These scripts will likely trigger multiple runs for each simulation model
    with various configuration settings.

