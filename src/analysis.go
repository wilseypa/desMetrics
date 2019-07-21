package main

import (
	"compress/bzip2"
	"compress/gzip"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"
	"time"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

// DesTraceData : Format of the modelSummary.json file
type DesTraceData struct {
	SimulatorName       string   `json:"simulator_name"`
	ModelName           string   `json:"model_name"`
	OriginalCaptureDate string   `json:"original_capture_date"`
	CaptureHistory      []string `json:"capture_history"`
	TotalLPs            int      `json:"total_lps"`
	NumInitialEvents    int      `json:"number_of_initial_events"`
	DateAnalyzed        string   `json:"date_analyzed"`
	EventData           struct {
		EventFile  string   `json:"file_name"`
		FileFormat []string `json:"format"`
		NumEvents  int      `json:"total_events"`
	} `json:"event_data"`
}

/*
	Data structure for events. Internally storing LP names with integer map value.
	Since we're storing events into an array indexed by the LP in question (sender or receiver)
	We will only store the other "companion" LP internally
	TODO - Do we need to do this? Can we just always store based on the sending LPs
*/
type eventData struct {
	companionLP int
	sendTime    float64
	receiveTime float64
}

/*
	Data structure for sent events, stored w/ their integer map value
*/
type eventSentData struct {
	companionLP int
	sendTime    float64
	receiveTime float64
}

/*
Data structure for LPs; each LP has unique ID and list of correlating events it generates.
*/
type lpData struct {
	lpID       int
	events     []eventData
	sentEvents int
}

// Functions to support sorting of events by their receive time
type byReceiveTime []eventData

func (b byReceiveTime) Len() int      { return len(b) }
func (b byReceiveTime) Swap(i, j int) { b[i], b[j] = b[j], b[i] }
func (b byReceiveTime) Less(i, j int) { return a[i].receiveTime < a[j].receiveTime }

// ReadInputFile : store into a struct and return
func ReadInputFile(fileName string) (d DesTraceData) {
	traceDataFile, err := os.Open(fileName)
	check(err)
	jsonDecoder := json.NewDecoder(traceDataFile)
	err = jsonDecoder.Decode(&d)
	check(err)
	return
}

// ReadDataOrder : returns the order of the four fields: "sLP", "rLP", "sTS", "rTS"
func ReadDataOrder(d *DesTraceData) []int {
	var ret = [4]int{-1, -1, -1, -1}
	for i, entry := range d.EventData.FileFormat {
		switch entry {
		case "sLP":
			ret[0] = i
		case "sTS":
			ret[1] = i
		case "rLP":
			ret[2] = i
		case "rTS":
			ret[3] = i
		default:
			fmt.Printf("Ignoring unknown element %v from EventData->Format of input JSON file.\n", entry)
		}
	}
	return ret
}

// GetTime : string of the current time for logging
func GetTime() string {
	return time.Now().Format(time.RFC850)
}

// OpenEventFile : creates pointers to a new CSV reader.
func OpenEventFile(fileName string) (*os.File, *csv.Reader) {
	eventFile, err := os.Open(fileName)
	check(err)

	var inputFile *csv.Reader
	if strings.HasSuffix(fileName, ".gz") || strings.HasSuffix(fileName, ".gzip") {
		unpack, err := gzip.NewReader(eventFile)
		check(err)
		infile = csv.NewReader(unpack)
	} else {
		if strings.HasSuffix(file, "bz2") || strings.HasSuffix(fileName, "bzip2") {
			unpack := bzip2.NewReader(eventFile)
			inFile = csv.NewReader(unpack)
		} else {
			infile = csv.NewReader(eventFile)
		}
	}
	infile.Comment = '#'
	infile.FieldsPerRecord = len(desTraceData.EventData.FileFormat)

	return eventFile, inFile
}

func updateRunningMeanVariance(curMean, varSum, newValue float64, numValues int) (float64, float64) {
	inc := newValue - curMean
	curMean += (incr / float64(numValues))
	varSum += (increment * (newValue - curMean))
	return curMean, varSum
}

func main() {

	var AnalyzeAllData bool
	flag.BoolVar(&AnalyzeAllData, "analyze-everything", false, "Turn on all analysis capabilities")

	// A flag for each of the output files
	var LocalEventChains bool
	flag.BoolVar(&LocalEventChains, "local-event-chains", false, "Compute local event chains")

	var LinkedEventChains bool
	flag.BoolVar(&LinkedEventChains, "linked-event-chains", false, "Compute linked event chains")

	var GlobalEventChains bool
	flag.BoolVar(&GlobalEventChains, "global-event-chains", false, "Compute global event chains")

	var EventsExecutedByLP bool
	flag.BoolVar(&EventsExecutedByLP, "events-executed-by-lp", false, "Compute summaries of local and remove events received")

	var NumLPSToCover bool
	flag.BoolVar(&NumLPSToCover, "num-lps-to-cover", false, "Compute number of LPs to cover percentage of events received")

	var EventChainSummary bool
	flag.BoolVar(&EventChainSummary, "event-chain-summary", false, "Compute the totals of the local and global event chains")

	var EventsExchangedRemote bool
	flag.BoolVar(&EventsExchangedRemote, "events-exchanged-remote", false, "Summary matrix for remote events exchanged")

	var EventsExchangedLocal bool
	flag.BoolVar(&EventsExchangedLocal, "events-exchanged-local", false, "Summary matrix of events exchanged between two LPs")

	var ReceiveTimeDeltas bool
	flag.BoolVar(&ReceiveTimeDeltas, "receive-time-deltas", false, "Compute receive time deltas between adjacent events in an LP")

	var TotalEventsProcessed bool
	flag.BoolVar(&TotalEventsProcessed, "total-events-processed", false, "Compute statistics on the total events processed")

	var EventsAvailableSimCycle bool
	flag.BoolVar(&EventsAvailableSimCycle, "events-available-sim-cycle", false, "Compute the events available per simulation cycle")

	var TimesXEventsAvailable bool
	flag.BoolVar(&TimesXEventsAvailable, "times-x-events-available", false, "Times X Events are available for execution")

	var debug bool
	flag.BoolVar(&debug, "debug", false, "Turn on debugging print statements")

	var help bool
	flag.BoolVar(&help, "help", false, "Usage")

	flag.Parse()

	if help {
		fmt.Println("Usage: desAnalysis [options...] FILE \n Analyze the event trace data described by the json file FILE.\n\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	desTraceData := ReadInputFile(flag.Arg(0))

	if debug {
		fmt.Printf("JSON file parsed successfully. Summary info:\n\tSimulator Name: %s\n\tOriginal Capture Date: %s\n\tCapture history: %s\n\tCSV File of Event Data: %s\n\tFormat of Event Data: %v\n",
			desTraceData.SimulatorName,
			desTraceData.ModelName,
			desTraceData.OriginalCaptureDate,
			desTraceData.CaptureHistory,
			desTraceData.EventData.EventFile,
			desTraceData.EventData.FileFormat)
	}

	desTraceData.TotalLPs = -1
	desTraceData.EventData.NumEvents = -1
	desTraceData.DateAnalyzed = ""
	desTraceData.NumInitialEvents = 0

	eventDataOrder := ReadDataOrder(&desTraceData)
	if debug {
		fmt.Printf("FileFormat: %v\neventDataOrderTable %v\n", desTraceData.EventData.FileFormat, eventDataOrderTable)
	}

	// Generate an error if there is a value of -1 in eventDataOrder, indicating not all values are filled in
	for _, entry := range eventDataOrder {
		if entry == -1 {
			log.Fatal("Missing critical field in EventData->Format of JSON file.\nRun with --debug to view more data.\n")
		}
	}

	// Enable the use of all CPUs on a system
	numThreads := runtime.NumCPU()
	// runtime.GOMAXPROCS(numThreads)
	// temp solution for kvack, doesn't fork to 32
	runtime.GOMAXPROCS(16)

	fmt.Printf("%v: Parallelism setup to support up to %v threads.\n", GetTime(), numThreads)

	// Now to connect to to CSV file (compressed (gz or bz2) or uncompressed)

	numLPs := 0
	numEvents := 0
	numInitialEvents := 0

	lpNameMap := make(map[string]*lpMap)

}
