# To use desAnalysis.go you first have to install and use golex.

## Installing golex

Set the environment variable GOPATH; I usually place it at ~/lib/go

Install (once) with this command
    go get github.com/cznic/golex

## Building/using desAnalysis.go

Build the lex.yy.go file with:
   $GOPATH/bin/golex lexer.gl

Compile:
  go build -o desAnalysis lex.yy.go desAnalysis.go

Run:
  `./desAnalysis <file.json>`

# To plot files generated from desAnalysis.go (uses python 2)

## Required Packages (pip install package_name)

+ Matplotlib
+ numpy
+ pandas
+ seaborn
+ brewer2mpl
+ networkx
+ python-louvain

## Using desGraphics.py

==== This section is under construction and will include optional flags ====

For now use:
<br>`python desGraphics.py`
