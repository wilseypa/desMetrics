# To use this you first have to install and use golex.

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
  ./desAnalysis <file.json>

