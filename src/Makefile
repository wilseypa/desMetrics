# Go params
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get

all: build

analysis:
	$(GOBUILD) -o desAnalysis analysis/analysis.go

sampler:
	$(GOBUILD) -o desSampler sampler/sampler.go

build: analysis sampler
	$(GOBUILD) -o desAnalysis analysis/analysis.go
	$(GOBUILD) -o desSampler sampler/sampler.go

clean:
	$(GOCLEAN)
	rm desAnalysis desSampler

