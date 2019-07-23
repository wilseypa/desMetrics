package main

import (
	"compress/bzip2"
	"compress/gzip"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
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

/*
	Records a unique integer value for each LP and store number of sent/received events
*/
type lpMap struct {
	toInt          int
	sentEvents     int
	receivedEvents int
}

// Functions to support sorting of events by their receive time
type byReceiveTime []eventData

func (b byReceiveTime) Len() int           { return len(b) }
func (b byReceiveTime) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byReceiveTime) Less(i, j int) bool { return b[i].receiveTime < b[j].receiveTime }

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
func ReadDataOrder(d *DesTraceData) map[string]int {
	ret := make(map[string]int)
	for i, entry := range d.EventData.FileFormat {
		switch entry {
		case "sLP":
			ret["SLP"] = i
		case "sTS":
			ret["sTS"] = i
		case "rLP":
			ret["rLP"] = i
		case "rTS":
			ret["rTS"] = i
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
func OpenEventFile(fileName string, d DesTraceData) (*os.File, *csv.Reader) {
	eventFile, err := os.Open(fileName)
	check(err)

	var inFile *csv.Reader
	if strings.HasSuffix(fileName, ".gz") || strings.HasSuffix(fileName, ".gzip") {
		unpack, err := gzip.NewReader(eventFile)
		check(err)
		inFile = csv.NewReader(unpack)
	} else {
		if strings.HasSuffix(fileName, "bz2") || strings.HasSuffix(fileName, "bzip2") {
			unpack := bzip2.NewReader(eventFile)
			inFile = csv.NewReader(unpack)
		} else {
			inFile = csv.NewReader(eventFile)
		}
	}
	inFile.Comment = '#'
	inFile.FieldsPerRecord = len(d.EventData.FileFormat)

	return eventFile, inFile
}

func updateRunningMeanVariance(curMean, varSum, newValue float64, numValues int) (float64, float64) {
	incr := newValue - curMean
	curMean += (incr / float64(numValues))
	varSum += (incr * (newValue - curMean))
	return curMean, varSum
}

// ProcessLine : does the dirty work for each line, removing unnecessary characters, parsing floats
func ProcessLine(eventRecord *[]string, eventDataOrder map[string]int) (float64, float64) {
	for i := range *eventRecord {
		(*eventRecord)[i] = strings.Trim((*eventRecord)[i], "'")
		(*eventRecord)[i] = strings.Trim((*eventRecord)[i], "`")
	}

	// Convert timestamps to floats
	sendTime, err := strconv.ParseFloat((*eventRecord)[eventDataOrder["sTS"]], 64)
	check(err)
	receiveTime, err := strconv.ParseFloat((*eventRecord)[eventDataOrder["rTS"]], 64)
	check(err)

	if sendTime > receiveTime {
		log.Fatal("Event has send time greater than receive time: %v %v %v %v\n",
			(*eventRecord)[eventDataOrder["sLP"]],
			sendTime,
			(*eventRecord)[eventDataOrder["rLP"]],
			receiveTime)
		log.Fatal("Aborting analysis.")
	}

	return sendTime, receiveTime
}

// TimesXEventsAvailable : total events processed per LP.
func TimesXEventsAvailable(reader *csv.Reader, lpNameMap map[string]*lpMap, eventDataOrder map[string]int) {

	available := make(map[float64]int)
	var receiveTimes []float64

	for {
		eventRecord, err := reader.Read()
		if err == io.EOF {
			break
		} else {
			check(err)
		}

		_, receiveTime := ProcessLine(&eventRecord, eventDataOrder)

		if _, has := available[receiveTime]; has {
			available[receiveTime]++
		} else {
			available[receiveTime] = 1
			receiveTimes = append(receiveTimes, receiveTime)
		}

	}

	// Now write them to file
	sort.Float64s(receiveTimes)

	// Now write the file out to analysisData/timesXeventsavailable.csv
	outFile, err := os.Create("analysisData/timesXeventsAvailable.csv")
	check(err)
	fmt.Fprintf(outFile, "# times X events are available for execution\n")
	fmt.Fprintf(outFile, "# X, num of occurrences\n")

	for _, t := range receiveTimes {
		val := available[t]
		fmt.Fprintf(outFile, "%v,%v\n", t, val)
	}

	err = outFile.Close()
	check(err)

}

// LPData : data structure for creating the totalEventsProcessedFile
type LPData struct {
	recLPId         string
	eventsProcessed int
	minTSDelta      float64
	maxTSDelta      float64
	avgTSDelta      float64
	varianceSum     float64
}

// Update the running mean and standard deviation of a receiving LP while running through the
// trace file. Update min/max timestamp deltas as well.
func updateLPDataStatistics(val *LPData, newTS float64) {
	val.eventsProcessed++
	incr := newTS - val.avgTSDelta
	val.avgTSDelta += (incr / float64(val.eventsProcessed))
	val.varianceSum += (incr * (newTS - val.avgTSDelta))
}

func totalEventsProcessed(reader *csv.Reader, lpNameMap map[string]*lpMap, eventDataOrder map[string]int) {

	recLPs := make(map[string]*LPData)

	for {
		eventRecord, err := reader.Read()
		if err == io.EOF {
			break
		} else {
			check(err)
		}

		sendTime, receiveTime := ProcessLine(&eventRecord, eventDataOrder)

		timestamp := receiveTime - sendTime

		if _, has := recLPs[eventRecord[eventDataOrder["rLP"]]]; !has {
			newData := new(LPData)
			newData.recLPId = eventRecord[eventDataOrder["rLP"]]
			newData.eventsProcessed = 1
			newData.minTSDelta = timestamp
			newData.maxTSDelta = timestamp
			newData.avgTSDelta = timestamp
			newData.varianceSum = 0
		} else {
			updateLPDataStatistics(recLPs[eventRecord[eventDataOrder["rLP"]]], timestamp)
		}

	}

	// Now write out the file and release the memory (done through garbage collection technically)
	// These are not written out in any order, and I don't believe that it should matter
	// TODO - wilsey - does it matter whether or not they are in order?

	outFile, err := os.Create("analysisData/totalEventsProcessed.csv")
	check(err)
	fmt.Fprintf(outFile, "#  Total Events Processed Data (per LP)")
	fmt.Fprintf(outFile, "# receiving LP, number of events processed, min timestamp delta, max timestamp delta, average timestamp delta, standard deviation.")

	for _, data := range recLPs {
		fmt.Fprintf(outFile, "%v,%v,%v,%v,%v,%v\n",
			data.recLPId,
			data.eventsProcessed,
			data.minTSDelta,
			data.maxTSDelta,
			data.avgTSDelta,
			math.Sqrt(data.varianceSum/float64(data.eventsProcessed)))
	}

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

	var XEventsAvailable bool
	flag.BoolVar(&XEventsAvailable, "times-x-events-available", false, "Times X Events are available for execution")

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
		fmt.Printf("FileFormat: %v\neventDataOrderTable %v\n", desTraceData.EventData.FileFormat, eventDataOrder)
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

	// Map of all the LPs and count of sent/received events.
	lpNameMap := make(map[string]*lpMap)

	// Use LPs to hold event data and lpIndex to record advancements of analysis among the LPs
	// TODO - Do we need the lpIndex variable?
	var lps []lpData
	var lpIndex []int

	// Build lpNameMap for each new LP, return pointer to the new lpMap struct
	defineLP := func(lp string) *lpMap {
		val, has := lpNameMap[lp]
		if !has {
			lpNameMap[lp] = new(lpMap)
			lpNameMap[lp].toInt = len(lpNameMap)
			val = lpNameMap[lp]
			numLPs++
		}
		return val
	}

	processEvent := func(sLP, rLP string, sTS, rTS float64) {
		numEvents++
		lp := defineLP(sLP)
		lp.sentEvents++
		lp = defineLP(rLP)
		lp.receivedEvents++

		// Count all events w/ sending time stamp <= 0 as an initial event
		if sTS <= 0 {
			numInitialEvents++
		}
	}

	// Event processing function. Fill in information for the LPs matrix that records events received
	// events received by each LP
	// TODO - move awawy from this, this creates a need fOpenEor a lot of space
	addEvent := func(sLP, rLP string, sTS, rTS float64) {
		rLPint := lpNameMap[rLP].toInt
		lpIndex[rLPint]++
		if lpIndex[rLPint] > cap(lps[rLPint].events) {
			log.Fatal("Something went wrong, we should have computed the appropriate size on the first parse.\n")
		}
		lps[rLPint].events[lpIndex[rLPint]].companionLP = lpNameMap[sLP].toInt
		lps[rLPint].events[lpIndex[rLPint]].receiveTime = rTS
		lps[rLPint].events[lpIndex[rLPint]].sendTime = sTS
	}

	// TODO - get rid of this after using/removing addEvent function
	_ = addEvent

	// First pass through the Event Data file
	eventFile, csvReader := OpenEventFile(desTraceData.EventData.EventFile, desTraceData)

	// Create the output directory
	err := os.MkdirAll("analysisData", 0777)
	check(err)

	TimesXEventsAvailable(csvReader, lpNameMap, eventDataOrder)

	for {
		eventRecord, err := csvReader.Read()
		if err == io.EOF {
			break
		} else {
			check(err)
		}

		// Remove leading/trailing quoteation characters
		for i := range eventRecord {
			eventRecord[i] = strings.Trim(eventRecord[i], "'")
			eventRecord[i] = strings.Trim(eventRecord[i], "`")
		}

		// Convert timestamps to floats
		sendTime, err := strconv.ParseFloat(eventRecord[eventDataOrder["sTS"]], 64)
		receiveTime, err := strconv.ParseFloat(eventRecord[eventDataOrder["rTS"]], 64)

		// Send time needs to be less than the receive time.
		// ROSS data actually has data with the send/receive times, so we will weaken this constraint TODO - ??? What does this mean
		if sendTime > receiveTime {
			log.Fatal("Event has send time greater than receive time: %v %v %v %v\n",
				eventRecord[eventDataOrder["sLP"]],
				sendTime,
				eventRecord[eventDataOrder["rLP"]],
				receiveTime,
			)
			log.Fatal("Aborting")
		}

		// processEvent(eventRecord[eventDataOrder["sLP"]], eventRecord[eventDataOrder["rLP"]], sendTime, receiveTime)

	}

	err = eventFile.Close()
	check(err)

}
