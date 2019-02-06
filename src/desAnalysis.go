// this program performs the analysis for the desMetrics project at UC
// (http://github.com/wilseypa/desMetrics).  this program inputs a json file containing profile data of the
// simulation model for which the event trace was captured.  the events are stored in a separate (compressed
// or uncompressed) csv file.  the name of this file is captured in the json file.  this project is developed
// from a parallel simulation (PDES) perspective and so much of the jargon and analysis is related to that
// field.  to understand the documentation, familiarity with PDES terminology is essential.  the input json
// file and overall project perspective is available from the project website.

// operationally, this program parses the event trace data file twice, the first pass captures the general
// characteristics of the file such as number of LPs, total number of events and so on.  the second pass
// inputs and stores the event data into internal data structures for processing.  this approach is followed
// to maintain the memory footprint as these files tend to be quite large.  memory and time are issues so the
// program is organized accordingly.  in particular, whenever possible, the analysis is partitioned and
// performed in parallel threads.  the program is setup to for a number of threads equal to the processor
// cores.  these threads have minimal communication and the program can place a heavy load on the host
// processor, so plan accordingly.

package main

import "io"
import "os"
import "fmt"
import "sort"
import "math"
import "strconv"
import "time"
import "runtime"
import "flag"
import "log"
import "strings"
import "encoding/json"
import "compress/gzip"
import "compress/bzip2"
import "encoding/csv"
import "math/rand"

// setup a data structure for events.  internally we're going to store LP names with their integer map value.
// since we're storing events into an array indexed by the LP in question (sender or receiver), we will only
// store the other "companion" LP internally.
type eventData struct {
	companionLP int
	sendTime float64
	receiveTime float64
}

// setup a data structure for sent events, stored with their integer map value. 
type eventSentData struct {
	companionLP int
	sendTime float64
	receiveTime float64
}

// setup a data structure for LPs; each LP has a unique id and a list of events it generates.
type lpData struct {
	lpId int
	events []eventData
	sentEvents int
}

// functions to support sorting of the events by their receive time
type byReceiveTime []eventData
func (a byReceiveTime) Len() int           {return len(a)}
func (a byReceiveTime) Swap(i, j int)      {a[i], a[j] = a[j], a[i]}
func (a byReceiveTime) Less(i, j int) bool {return a[i].receiveTime < a[j].receiveTime}

func main() {

	// --------------------------------------------------------------------------------
	// process the command line

	// i am removing this next argument for now; it is my belief that we should have a json profile for every
	// sampled data set so we can trace it back to it's origins 

	// switches to turn on/off the different types of analysis to perform
	var analyzeAllData bool
	flag.BoolVar(&analyzeAllData, "analyze-everything", false, "Turn on all analysis capabilities")

	var analyzeReceiveTimeData bool
	flag.BoolVar(&analyzeReceiveTimeData, "analyze-event-receiveTimes", false, "Turn on an analysis of an LP's events by receiveTime")

	// for large event trace files, it is sometimes necessary to analyze only samples of the full event trace data.
	// this argument permits the user to define an alternate event trace data file for analysis.

	// this file can be large, so we provide an options to turn it off.
	var commSwitchOff bool
	flag.BoolVar(&commSwitchOff, "no-comm-matrix", false, "turn off generation of the file eventsExchanged.csv")

	// turns on a bunch of debug printing
	var debug bool
	flag.BoolVar(&debug, "debug", false, "turn on debugging.")
	
	// the default help out from the flag library doesn't include a way to include argument definitions; these
	// definitions permit us to define our own output from the -help flag.
	var help bool
	flag.BoolVar(&help, "help", false, "print out help.")
	flag.BoolVar(&help, "h", help, "print out help.")

	flag.Parse()
	
	printUsageAndExit := func() {
		fmt.Print("Usage: desAnalysis [options...] FILE \n Analyze the event trace data described by the json file FILE.\n\n")
		flag.PrintDefaults()
	}
	if help {
		printUsageAndExit()
		os.Exit(0)
	}

	if flag.NArg() != 1 {
		fmt.Printf("Invalid number of arguments (%v); only one expected.\n\n",flag.NArg())
		printUsageAndExit()
	}

	if debug {
		fmt.Printf("Command Line: commSwitchOff: %v, debug: %v, Args: %v.\n",
			commSwitchOff, debug, flag.Arg(0))
	}

	// --------------------------------------------------------------------------------
	// process the model json file

	// format of json file describing the model and location (csv file) of the event data
	var desTraceData struct {
		SimulatorName string `json:"simulator_name"`
		ModelName string `json:"model_name"`
		OriginalCaptureDate string `json:"original_capture_date"`
		CaptureHistory [] string `json:"capture_history"`
		TotalLPs int `json:"total_lps"`
		NumOfEvents int `json:"total_events"`
		NumInitialEvents int `json:"number_of_initial_events"`
		EventData struct {
			EventFile string `json:"file_name"`
			FileFormat []string `json:"format"`
			NumEvents int `json:"total_events"`
		} `json:"event_data"`
		DateAnalyzed string `json:"date_analyzed"`
	}
	
	// get a handle to the input file and import/parse the json file
//	traceDataFile, err := os.Open(os.Args[2])
	traceDataFile, err := os.Open(flag.Arg(0))
	defer traceDataFile.Close()
	if err != nil { panic(err) }
	fmt.Printf("Processing input json file: %v\n",flag.Arg(0))
	jsonDecoder := json.NewDecoder(traceDataFile)
	err = jsonDecoder.Decode(&desTraceData); 
	if err != nil { panic(err) }
	if debug { fmt.Printf("Json file parsed successfully.  Summary info:\n    Simulator Name: %s\n    Model Name: %s\n    Original Capture Date: %s\n    Capture history: %s\n    CSV File of Event Data: %s\n    Format of Event Data: %v\n", 
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
	desTraceData.NumOfEvents = 0
	desTraceData.NumInitialEvents = 0

	// so we need to map the csv fields from the event data file to the order we need for our internal data
	// structures (sLP, sTS, rLP, rTS).  the array eventDataOrderTable will indicate which csv entry corresponds;
	// thus (for example), eventDataOrderTable[0] will hold the index where  the sLP field lies in
	// desTraceData.EventData.FileFormat 

	eventDataOrderTable := [4]int{ -1, -1, -1, -1 }
	for i, entry := range desTraceData.EventData.FileFormat {
		switch entry {
		case "sLP":
			eventDataOrderTable[0] = i
		case "sTS":
			eventDataOrderTable[1] = i
		case "rLP":
			eventDataOrderTable[2] = i
		case "rTS":
			eventDataOrderTable[3] = i
		default:
			fmt.Printf("Ignoring unknown element %v from event_data->format field of the model json file.\n", entry)
		}
	}

	if debug { fmt.Printf("FileFormat: %v; eventDataOrderTable %v\n", desTraceData.EventData.FileFormat, eventDataOrderTable) }

	// sanity check; when error generated turn on debugging and look at -1 field in eventDataOrderTable to discover mssing entry
	for _, entry := range eventDataOrderTable {
		if entry == -1 { log.Fatal("Missing critcal field in event_data->format of model json file; run with --debug to view relevant data.\n") }
	}

	// --------------------------------------------------------------------------------
	// enable the use of all CPUs on the system
	numThreads := runtime.NumCPU()
//	runtime.GOMAXPROCS(numThreads)
	// temp solution for kvack....for some reason it doesn't fork to 32
	runtime.GOMAXPROCS(16)

	// function to print time as the program reports progress to stdout
	getTime := func () string {return time.Now().Format(time.RFC850)}

	fmt.Printf("%v: Parallelism setup to support up to %v threads.\n", getTime(), numThreads)

	// --------------------------------------------------------------------------------
	// function to connect to compressed (gz or bz2) and uncompressed eventData csv files

	openEventFile := func(fileName string) (*os.File, *csv.Reader) {
		eventFile, err := os.Open(fileName)
		if err != nil { panic(err) }
		var inFile *csv.Reader
		if strings.HasSuffix(fileName, ".gz") || strings.HasSuffix(fileName, ".gzip") {
			unpackRdr, err := gzip.NewReader(eventFile)
			if err != nil { panic(err) }
			inFile = csv.NewReader(unpackRdr)
		} else { if strings.HasSuffix(fileName, "bz2") || strings.HasSuffix(fileName, "bzip2") {
			unpackRdr := bzip2.NewReader(eventFile)
			inFile = csv.NewReader(unpackRdr)
			
		} else {
			inFile = csv.NewReader(eventFile)
		}}
		
		// tell the csv reader to skip lines beginning with a '#' character
		inFile.Comment = '#'
		// force a count check on the number of entries per row with each read
		inFile.FieldsPerRecord = len(desTraceData.EventData.FileFormat)
		
		return eventFile, inFile
	}

	// --------------------------------------------------------------------------------
	// function to compute running mean/variance for a stream of data values, 4 arguments
	//    currentAve: the ongoing mean
	//    varianceSum: the accumulated value for the variance (ultimately the actual variance = varianceSum / (numValues-1)
	//    newValue: the new value to add to the mean/variance computation
	//    numValues: the total number of values seen in the stream (including the new value)

	updateRunningMeanVariance := func(currentMean float64, varianceSum float64, newValue float64, numValues int) (float64, float64) {
		increment := newValue - currentMean
		currentMean = currentMean + (increment / float64(numValues))
		varianceSum = varianceSum + (increment * (newValue - currentMean))
		return currentMean, varianceSum
	}

	// --------------------------------------------------------------------------------
	// now on to the heavy lifting (the actual analysis)
	
	// memory is a big issue.  in order to minimize the size of our data structures, we will process the
	// input JSON file twice.  the first time we will build a map->int array for the LPs, count the total
	// number of LPs, and the total number of events.  we will use these counts to build the principle
	// data structures to hold events by LPs (by assuming that the events are more or less uniformly
	// distributed among the LPs.  in the second pass, we will store the events into our internal data
	// structures.  we will use the function processEvent to mahage each step.  between parses it will be
	// changed to point to the addEvent function.

	// variables to record the total number of LPs and events; numOfLPs will also be used to
	// enumerate unique integers for each LP recorded; set during the first pass over the JSON file.
	numOfLPs := 0
	numOfEvents := 0
	numOfInitialEvents := 0

	// record a unique int value for each LP and store the total number of sent and received events by that LP.
	type lpMap struct {
		toInt int
		sentEvents int					// CHECK THIS VALUE TO BE GREATER THAN 0
		receivedEvents int
	}

	lpNameMap := make(map[string]*lpMap)

	// ultimately we will use lps to hold event data; and lpIndex to record advancements of analysis among the lps
	var lps []lpData
	var lpIndex []int

	// if necessary, build lpNameMap for each new LP and then return pointer to lpMap of said LP
	defineLP := func(lp string) *lpMap {
		item, present := lpNameMap[lp]
		if !present {
			lpNameMap[lp] = new(lpMap)
			lpNameMap[lp].toInt = numOfLPs
			item = lpNameMap[lp]
			numOfLPs++
		}
		return item
	}

	// during the first pass over the JSON file: record all new LPs into the lpNameMap and count
	// the number of events sent/received by each LP; also count the total number of events in
	// the simulation 
	processEvent := func(sLP string, sTS float64, rLP string, rTS float64) {
		numOfEvents++
		lp := defineLP(sLP)
		lp.sentEvents++
		lp = defineLP(rLP)
		lp.receivedEvents++
		// we will count all events with a sending time stamp less or equal to zero as an initial event
		if sTS <= 0 {numOfInitialEvents++}
	}

	// this is the event processing function for the second pass over the JSON file; basically
	// fill in the information for the lps matrix that records events received by each LP
	addEvent := func(sLP string, sTS float64, rLP string, rTS float64) {
		rLPint := lpNameMap[rLP].toInt
		lpIndex[rLPint]++
		if lpIndex[rLPint] > cap(lps[rLPint].events) {log.Fatal("Something wrong, we should have computed the appropriate size on the first parse.\n")}
		lps[rLPint].events[lpIndex[rLPint]].companionLP = lpNameMap[sLP].toInt
		lps[rLPint].events[lpIndex[rLPint]].receiveTime = rTS
		lps[rLPint].events[lpIndex[rLPint]].sendTime = sTS
	}

	// process the desTraceData file; processEvent is redefined on the second pass to do the heavy lifting

	processEventDataFile := func() {

		eventFile, csvReader := openEventFile(desTraceData.EventData.EventFile)

		readLoop: for {

			eventRecord, err := csvReader.Read()
			if err != nil { if err == io.EOF {break readLoop} else { panic(err) }}
			
			// remove leading/trailing quote characters (double quotes are stripped automatically by the csvReader.Read() fn
			for i, _ := range eventRecord {
				eventRecord[i] = strings.Trim(eventRecord[i], "'")
				eventRecord[i] = strings.Trim(eventRecord[i], "`")
			}

			// convert timestamps from strings to floats
			sendTime, err := strconv.ParseFloat(eventRecord[eventDataOrderTable[1]],64)
			receiveTime, err := strconv.ParseFloat(eventRecord[eventDataOrderTable[3]],64)
			
			// we should require that the send time be strictly less than the receive time, but the ross
			// (airport model) data actually has data with the send/receive times (and other sequential
			// simulators are likely to have this as well), so we will weaken this constraint.
			if sendTime > receiveTime {
				log.Fatal("Event has send time greater than receive time: %v %v %v %v\n", 
					eventRecord[eventDataOrderTable[0]], sendTime, eventRecord[eventDataOrderTable[2]], receiveTime)
				log.Fatal("Aborting")
			}

			if debug {fmt.Printf("Event recorded: %v, %v, %v, %v\n", eventRecord[eventDataOrderTable[0]], sendTime, eventRecord[eventDataOrderTable[2]], receiveTime)}

			processEvent(eventRecord[eventDataOrderTable[0]], sendTime, eventRecord[eventDataOrderTable[2]], receiveTime)
		}
			err = eventFile.Close()
			if err != nil { panic(err) }
	}

	// --------------------------------------------------------------------------------
	// ok, now let's process the event data and populate our internal data structures


	// on the first pass, we will collect information on the number of events and the number of LPs

	fmt.Printf("%v: Processing %v to capture event and LP counts.\n", getTime(), desTraceData.EventData.EventFile)
	processEventDataFile()

	// lps is an array of the LPs; each LP entry will hold the events it received
	lps = make([]lpData, len(lpNameMap))

	// periodically we will need to walk through the LP arrays independently; we will use this array to do so
	lpIndex = make([]int, len(lps))
	// allocate entries in each LP to hold the number events it received AND SENT
	for _, i := range lpNameMap {
		lpIndex[i.toInt] = -1
		lps[i.toInt].lpId = i.toInt
		lps[i.toInt].events = make([]eventData, i.receivedEvents)
		lps[i.toInt].sentEvents = i.sentEvents//make([]eventSentData, i.sentEvents)			// construct sentEvents array in lp data struct
	}

	// on the second pass,  we will save the events

	fmt.Printf("%v: Processing %v to capture events.\n", getTime(), flag.Arg(0))
	processEvent = addEvent
	processEventDataFile()

	// build the reverse map: integers -> LP Names
	mapIntToLPName := make([]string, numOfLPs)
	for key, value := range lpNameMap {mapIntToLPName[value.toInt] = key}

	// Now we'll check to see if all LPs sent an event (can be a problem if LP sent one event in first 1 %
	// and did not receive anything afterwards) will record max num of events
	// we check this before received events, bc desAnalysis will crash if each LP doesn't send one
	fmt.Printf("%v: Verifying that all LPs sent at least one event. \n", getTime())
	maxLPSentArray := 0
	var zeroSentLPs = 0
	for i := range lps {
		if lps[i].sentEvents == 0 {
			zeroSentLPs++
		}
		if maxLPSentArray < lps[i].sentEvents {maxLPSentArray = lps[i].sentEvents}
	}
	fmt.Printf("    %v LP's sent ZERO messages.\n", zeroSentLPs)


	// let's check to see if all LPs received an event (not necessarily a huge problem, but something we
	// should probably be aware of.  also, record the max num of events received by an LP
	// fmt.Printf("%v: Verifying that all LPs received at least one event.\n", getTime())
	maxLPEventArray := 0
	var zeroReceivedLPs = 0
	for i := range lps {
		if len(lps[i].events) == 0 {
			zeroReceivedLPs++
		}
		if maxLPEventArray < len(lps[i].events) {maxLPEventArray = len(lps[i].events)}
	}
	fmt.Printf("    %v LP's received ZERO messages.\n", zeroReceivedLPs)
	
	// we now need to sort the event lists by receive time.  for this we'll use the sort package.
	fmt.Printf("%v: Sorting the events in each LP by receive time.\n", getTime())
	for i := range lps {sort.Sort(byReceiveTime(lps[i].events))}

	// on to analysis....

	// create the directory to store the resulting data files (if it is not already there)
	err =  os.MkdirAll ("analysisData", 0777)
	if err != nil {panic(err)}

// --------------------------------------------------------------------------------
// create a json formatted config file in the output directory

	outFile, err := os.Create("analysisData/modelSummary.json")
	if err != nil {panic(err)}

	desTraceData.TotalLPs = numOfLPs
	desTraceData.EventData.NumEvents = numOfEvents
	desTraceData.NumInitialEvents = numOfInitialEvents
	desTraceData.NumOfEvents = numOfEvents
	desTraceData.DateAnalyzed = getTime()

	jsonEncoder := json.NewEncoder(outFile)
	jsonEncoder.SetIndent("", "    ")
	err = jsonEncoder.Encode(&desTraceData)

//	os.Exit(0)

	err = outFile.Close()
	if err != nil {panic(err)}
// --------------------------------------------------------------------------------


	fmt.Printf("%v: Analysis (parallel) of events organized by receiving LP.\n", getTime())	

	// in this next step, we are going to compute event received summaries.  in this case we are going to
	// attack the problem by slicing the lps array and assigning each slice to a separate goroutine.  to
	// optimize performance we are going to have each goroutine perform all analyses and pass the results LP
	// by LP back to the main routine.  in each pass over an LP each goroutine will: (i) compute the
	// local/remote event data, (ii) record the LPs that send events to this LP and compute the coverage
	// statistics, and (iii) the (local, global, and linked) chain info for each LP.  while these
	// computations can technically be folded together and performed in one pass over each LP's event data, we
	// will keep them separate in order to keep the algorithm cleaner.  

	// in this next step, we are going to compute event received summaries.  in this case we are going to
	// compute two different results.  the first is a simple count of the (i) total events received, (ii)
	// the local events received local and, (iii) the remote events received.  in this case local means
	// events sent and received by the same LP, remote means events sent by some other LP.  the second
	// computation is the number of sending LPs to cover a certain percentage of the total events
	// received.  for example, how many LP (say X) were responsible for 75% of the events received by this
	// LP.  for this we will order the sending LP counts so that they are ordered by decreasing number of 
	// events sent to this LP.


	// this helper function will compute the number of LPs (sorted by most sending to least sending) that
	// result in coverage of the percent of events received (basically how many LPs send X% of the
	// received messages).
	numOfLPsToCover := func(total int, data []int, percent int) int {
		// add .99 because int() truncates and we actually want rounding
		numRequired := int(((float32(total) * float32(percent)) / float32(100)) + .99)
		// ok, we had a corner case where this test is needed
		if (numRequired > total) {numRequired = total}
		lpCount := 0
		eventCount := 0
		for i := range data {
			lpCount++
			eventCount = eventCount + data[i]
			if eventCount >= numRequired {
				return lpCount
			}
		}
		fmt.Printf("ERROR: something's amiss in this computation: %v, %v, %v, %v, %v\n", total, percent, numRequired, eventCount, len(data))
		panic("aborting.")
	}

	// data type to capture each LP's event summary data
	type lpEventSummary struct {
		lpId int
		local int
		remote int
		total int
		cover [5]int
		localChain []int
		linkedChain []int
		globalChain []int
	}

	// compute the local/remote events and cover statistics
	computeLPEventsProcessed := func(lp lpData) lpEventSummary {
		var es lpEventSummary
		lpEventSendCount := make([]int,numOfLPs)
		es.lpId = lp.lpId
		es.local = 0
		es.remote = 0
		for _, event := range lp.events {
			if lp.lpId != event.companionLP {
				es.remote++
				// we don't want to count self generated events in the sender set.
				lpEventSendCount[event.companionLP]++
			} else {
				es.local++
			}
		}
		sort.Sort(sort.Reverse(sort.IntSlice(lpEventSendCount)))
		es.total = es.remote + es.local
		// since we're tracking message transmissions, use only the remote event total
		es.cover[0] = numOfLPsToCover(es.remote, lpEventSendCount, 75)  //  75% cutoff
		es.cover[1] = numOfLPsToCover(es.remote, lpEventSendCount, 80)  //  80% cutoff
		es.cover[2] = numOfLPsToCover(es.remote, lpEventSendCount, 90)  //  90% cutoff
		es.cover[3] = numOfLPsToCover(es.remote, lpEventSendCount, 95)  //  95% cutoff
		es.cover[4] = numOfLPsToCover(es.remote, lpEventSendCount, 100) // 100% cutoff
		return es
	}

	// event chains are collections of events for execution at the LP that could potentially be executed
	// together.  thus when examining an event with receive timestamp t, the chain is formed by all future
	// (by receive time) events in that LP that have a send time < t.  local chains have the additional
	// constraint that they must also have been sent by the executing LP (self generated events).  global
	// chains can be sent from any LP.

	// a third form of event chains is the "linked event chain".  the linked chain is similar to the local
	// chain except that the constraint on the send time is relaxed so that any event sent within the time
	// window of the local event chain (so basically any event generated by the chain is also potentially
	// a member of the chain).

	// consider a locally linked chain computation.  that is, anything generated within the time frame of
	// the chain should also be included in the chain length computation

	// all chains longer than this will be summed as if they are this length.
	chainLength := 5

	accumulateChain := func(chain []int, maxChainLength int, chainLength int) {
		if chainLength >= maxChainLength {chain[maxChainLength - 1]++} else {chain[chainLength]++}
		return
	}

	// PAW: consider changing these to being strictly less than the receiveTime
	computeEventChains := func(lp lpData) ([]int, []int, []int) {

		local := make([]int,chainLength)
		linked := make([]int,chainLength)
		global := make([]int,chainLength)

		// LOCAL CHAINS: an event is part of the local chain if it is (i) generated by this LP (lpId) and (ii)
		// if it's send time is less than the receive time of the event at the head of the chain.
		i := 0
		for ; i < len(lp.events) ; {
			// skip over non local events
			for ; i < len(lp.events) && lp.events[i].companionLP != lp.lpId ; i++ {}
			j := i + 1
			// find end of chain
			for ; j < len(lp.events) && lp.events[j].companionLP == lp.lpId && lp.events[j].sendTime < lp.events[i].receiveTime ; {j++}
			accumulateChain(local, chainLength, j-i-1)
			i = j
		}
		
		// LINKED CHAINS: an event is part of the linked chain if it is (i) generated by this LP (lpId) and (ii)
		// if it's send time is less than or equal to the receive time (NOTE the change from local chains) of
		// any event currently in the chain (tested against the last event already in the chain)
		i = 0
		for ; i < len(lp.events) ; {
			// skip over non local events
			for ; i < len(lp.events) && lp.events[i].companionLP != lp.lpId ; i++ {}
			j := i + 1
			// find end of chain
			for ; j < len(lp.events) && lp.events[j].companionLP == lp.lpId && lp.events[j].sendTime <= lp.events[j-1].receiveTime ; {j++}
			accumulateChain(linked, chainLength, j-i-1)
			i = j
		}
		
		// GLOBAL CHAINS: an event is part of the global chain if it is has a send time that is less the receive
		// time of any event currently in the chain (tested against the last event already in the chain)
		i = 0
		for ; i < len(lp.events) ; {
			j := i + 1
			for ; j < len(lp.events) && lp.events[j].sendTime < lp.events[i].receiveTime ; {j++}
			accumulateChain(global, chainLength, j-i-1)
			i = j
		}
		return local, linked, global
	}

	// PAW: change the comment headers of chains/covers to for loops so the numbers will automatically grow
	// with the variable setting.... 
	localChainFile, err := os.Create("analysisData/localEventChainsByLP.csv")
	if err != nil {panic(err)}
	fmt.Fprintf(localChainFile,"# local event chains by LP\n")
	fmt.Fprintf(localChainFile,"# LP, local chains of length: 1, 2, 3, 4, ... , >= n\n")

	linkedChainFile, err := os.Create("analysisData/linkedEventChainsByLP.csv")
	if err != nil {panic(err)}
	fmt.Fprintf(linkedChainFile,"# linked event chains by LP\n")
	fmt.Fprintf(linkedChainFile,"# LP, linked chains of length: 1, 2, 3, 4, ... , >= n\n")

	globalChainFile, err := os.Create("analysisData/globalEventChainsByLP.csv")
	if err != nil {panic(err)}
	fmt.Fprintf(globalChainFile,"# global event chains by LP\n")
	fmt.Fprintf(globalChainFile,"# LP, global chains of length: 1, 2, 3, 4, ... , >= n\n")

	// location to write summaries of local and remote events received
	eventSummaries, err := os.Create("analysisData/eventsExecutedByLP.csv")
	if err != nil {panic(err)}
	fmt.Fprintf(eventSummaries, "# summary of local and remote events executed\n")
	fmt.Fprintf(eventSummaries, "# LP, local, remote, total\n")

	// location to write percentage of LPs to cover percentage of events received
	numToCover, err := os.Create("analysisData/numOfLPsToCoverPercentEventMessagesSent.csv")
	if err != nil {panic(err)}
	fmt.Fprintf(numToCover,"# number of destination LPs (sorted by largest messages sent to) to cover percentage of total events\n")
	fmt.Fprintf(numToCover,"# LP name, total events sent, num of LPs to cover: 75, 80, 90, 95, and 100 percent of the total events sent.\n")

	// this will be invoked as a gogroutine with LPs equally (nearly) partitioned among them
	analyzeReceivedEvents := func (lps []lpData, c chan<- lpEventSummary) {
		for _, lp := range(lps) {
			eventsProcessed := computeLPEventsProcessed(lp)
			eventsProcessed.localChain, eventsProcessed.linkedChain, eventsProcessed.globalChain = computeEventChains (lp)
			c <- eventsProcessed
		}
	}

	// defining how many LPs do we assign to each thread
	goroutineSliceSize := int((float32(len(lps))/float32(numThreads)) + .5)

	// each goroutine will compute event counts for one LP, send the results back over the channel and
	// continue. 
	c := make(chan lpEventSummary, numThreads * 4)
	if len(lps) < numThreads * 2 {numThreads = numThreads/2}
	for i := 0; i < numThreads; i++ {
		low := i * goroutineSliceSize
		high := low + goroutineSliceSize
		if i == numThreads - 1 {high = len(lps)}
		go analyzeReceivedEvents(lps[low:high], c)
	}

	localEventChainSummary := make([]int,chainLength)
	linkedEventChainSummary := make([]int,chainLength)
	globalEventChainSummary := make([]int,chainLength)

	// process all of the data in the channel and output the results
	for _ = range lps {
		eventsProcessed := <- c
		// capture event chain summaries
		for i := 0; i < chainLength; i++ {
			localEventChainSummary[i] = localEventChainSummary[i] + eventsProcessed.localChain[i]
			linkedEventChainSummary[i] = linkedEventChainSummary[i] + eventsProcessed.linkedChain[i]
			globalEventChainSummary[i] = globalEventChainSummary[i] + eventsProcessed.globalChain[i]
		}
		fmt.Fprintf(eventSummaries,"%v, %v, %v, %v\n", 
			mapIntToLPName[eventsProcessed.lpId], eventsProcessed.local, eventsProcessed.remote, eventsProcessed.local + eventsProcessed.remote)
		// PAW: turn this into a for loop so the variable will actually control
		fmt.Fprintf(numToCover,"%v, %v", mapIntToLPName[eventsProcessed.lpId], eventsProcessed.remote)
		for _, i := range eventsProcessed.cover {fmt.Fprintf(numToCover,", %v", i)}
		fmt.Fprintf(numToCover,"\n")
		fmt.Fprintf(localChainFile,"%v",mapIntToLPName[eventsProcessed.lpId])
		fmt.Fprintf(linkedChainFile,"%v",mapIntToLPName[eventsProcessed.lpId])
		fmt.Fprintf(globalChainFile,"%v",mapIntToLPName[eventsProcessed.lpId])
		for i := range eventsProcessed.localChain {
			fmt.Fprintf(localChainFile,", %v", eventsProcessed.localChain[i])
			fmt.Fprintf(linkedChainFile,", %v", eventsProcessed.linkedChain[i])
			fmt.Fprintf(globalChainFile,", %v", eventsProcessed.globalChain[i])
		}
        fmt.Fprintf(localChainFile,"\n")
        fmt.Fprintf(linkedChainFile,"\n")
        fmt.Fprintf(globalChainFile,"\n")
	}
	close(c)

	err = eventSummaries.Close()
	if err != nil {panic(err)}
	err = numToCover.Close()
	if err != nil {panic(err)}
	err = localChainFile.Close()
	if err != nil {panic(err)}
	err = linkedChainFile.Close()
	if err != nil {panic(err)}
	err = globalChainFile.Close()
	if err != nil {panic(err)}

	// number of LPs with n event chains of length X number of LPs with average event chains of length X
	
	// not sure this will be useful or not, but let's save totals of the local and global event chains.
	// specifically we will sum the local/global event chains for all of the LPs in the system

	outFile, err = os.Create("analysisData/eventChainsSummary.csv")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# number of event chains of length X\n")
	fmt.Fprintf(outFile,"# chain length, num of local chains, num of linked chains, num of global chains\n")
	for i := 0; i < chainLength; i++ {
		fmt.Fprintf(outFile,"%v, %v, %v, %v\n", i+1,
			localEventChainSummary[i],linkedEventChainSummary[i],globalEventChainSummary[i])
	}
	err = outFile.Close()
	if err != nil {panic(err)}

	//  now we will compute and print a summary matrix of the number of events exchanged between two LPs; while we
	// do this, we will also capture the minimum, maximum, and average delta between the send and receive time (for
	// lookahead analysis).  this matrix can be quite large and can actually be too large to create.  in order to
	// save space, we will print only the non-zero entries.  in the cases where it is too large to print, the
	// command line argument "--no-comm-matrix" can be used to suppress it's creation.

	if commSwitchOff == false {
		outFile, err = os.Create("analysisData/eventsExchanged-remote.csv")
		if err != nil {panic(err)}
		
		outFile2, err := os.Create("analysisData/eventsExchanged-local.csv")
		if err != nil {panic(err)}
		
		fmt.Fprintf(outFile,"# event exchanged matrix data (remote)\n")
		fmt.Fprintf(outFile,"# receiving LP, sending LP, num of events sent, minimum timestamp delta, maximum timestamp delta, average timestamp delta, variance of ave delta\n")
		
		fmt.Fprintf(outFile2,"# event exchanged matrix data (local)\n")
		fmt.Fprintf(outFile2,"# receiving LP, sending LP, num of events sent, minimum timestamp delta, maximum timestamp delta, average timestamp delta, variance of ave delta\n")
		
		lpEventSendCount := make([]int,numOfLPs)
		minTimeDelta := make([]float64,numOfLPs)
		maxTimeDelta := make([]float64,numOfLPs)
		aveSendTimeDelta := make([]float64,numOfLPs)
		varianceOfTimeDelta := make([]float64, numOfLPs)
		for j, lp := range(lps) {
			for i := range lpEventSendCount {
				lpEventSendCount[i] = 0
				aveSendTimeDelta[i] = 0
				minTimeDelta[i] = math.MaxFloat64
				maxTimeDelta[i] = 0
				varianceOfTimeDelta[i] = 0
			}
			for _, event := range lp.events {
				lpEventSendCount[event.companionLP]++
				delta := event.receiveTime - event.sendTime
				aveSendTimeDelta[event.companionLP] = aveSendTimeDelta[event.companionLP] + delta
				// update the minimum delta only if the event is not part of the initial event
				// pool.  we will be conservative on this and define an event as part of the
				// initial pool if it's receive time is 0.0 or if it's send time is <= 0.0 
				if (minTimeDelta[event.companionLP] > delta) && (event.receiveTime != 0.0) && (event.sendTime > 0.0) {minTimeDelta[event.companionLP] = delta}
				if maxTimeDelta[event.companionLP] < delta {maxTimeDelta[event.companionLP] = delta} 
				aveSendTimeDelta[event.companionLP], varianceOfTimeDelta[event.companionLP] =
					updateRunningMeanVariance(aveSendTimeDelta[event.companionLP], varianceOfTimeDelta[event.companionLP], delta, lpEventSendCount[event.companionLP])
			}
			for i := range lpEventSendCount {
				if lpEventSendCount[i] != 0 {
					if i != j {
						// remote event
						fmt.Fprintf(outFile,"%v,%v,%v,%v,%v,%v,%v\n", j, i, lpEventSendCount[i],minTimeDelta[i],maxTimeDelta[i],aveSendTimeDelta[i],varianceOfTimeDelta[i]/(float64(lpEventSendCount[i])-1))
					} else {
						// local event
						fmt.Fprintf(outFile2,"%v,%v,%v,%v,%v,%v,%v\n", j, i, lpEventSendCount[i],minTimeDelta[i],maxTimeDelta[i],aveSendTimeDelta[i],varianceOfTimeDelta[i]/(float64(lpEventSendCount[i])-1))
					}
				}
			}
		}
		err = outFile.Close()
		if err != nil {panic(err)}
	}

	// collect analysis data on the events processed by an LP.  most of this will be relating to the
	// intervals of receive time between adjacent in an LP.

	analyzeEventsByReceiveTime := func() {

		fmt.Printf("%v: Analyzing the events in an LP organized by their receive time.\n", getTime())
		
		outFile, err = os.Create("analysisData/eventReceiveTimeDeltasByLP.csv")
		if err != nil {panic(err)}
		
		fmt.Fprintf(outFile, "# Record of the timestamp deltas (receiveTime[LP_i+1] - receiveTime[LP_i]) of events in each LP\n")
		fmt.Fprintf(outFile, "# receiving LP, number of events processed in unique timesteps, min time delta, max time delta, mean time delta (of all but initial events), variance of mean, num times events share a timestamp, num of sampling windows, length of time interval sampled, [,num of sampling windows] events falling in said window\n")
		
		for _, lp := range(lps) {
			
			//reset counters for the next LP
			lastEventTime := 0.0
			numUniqueEvents := 0
			minTimeDelta := math.MaxFloat64
			maxTimeDelta := 0.0
			mean := 0.0
			variance := 0.0
			frequencyOfSharedReceiveTime := 0
			firstEventSeen := false
			
			// if this LP has zero events, move to the next LP.
			if len(lp.events) == 0 {continue}
			
			for _, event := range(lp.events) {
				
				// skip all events at time zero (assumption: they are part of the inital event pool)
				if event.receiveTime == 0.0 {continue}
				
				if firstEventSeen == false {
					firstEventSeen = true
					numUniqueEvents++
					lastEventTime = event.receiveTime
					continue
				}
				if event.receiveTime == lastEventTime {
					frequencyOfSharedReceiveTime++
				} else {
					delta := event.receiveTime - lastEventTime
					if delta <= 0.0 {log.Fatal("Something is amiss; it should not be possible to have a receive time delta (%v) of zero or less.\n", delta)}
					mean, variance = updateRunningMeanVariance(mean, variance, delta, numUniqueEvents)
					if (delta < minTimeDelta) {minTimeDelta = delta}
					if (delta > maxTimeDelta) {maxTimeDelta = delta}
					lastEventTime = event.receiveTime
					numUniqueEvents++
				}
			}
			variance = variance / float64(numUniqueEvents - 1)
			
			// now its a simple matter of writing the results
			fmt.Fprintf(outFile, "%v, %v, %v, %v, %v, %v, %v,", mapIntToLPName[lp.lpId], numUniqueEvents, minTimeDelta, maxTimeDelta, mean, variance, frequencyOfSharedReceiveTime)
			
			// next we will examine the number of events executed in a fixed time interval at random
			// locations of the LP's executed events.  the idea is to see if the LP executes events at a
			// mostly stationary rate throughout the execution or if there are hot/cold spots during the
			// simulation run where the LP executes significantly more (less) events.  we will capture
			// numEventDeltaSamples (initially set at 10, but the code will support changing this value to
			// any integer).
			
			// given that we have no common base timeline for an input simulation model, we will first
			// compute the mean taken from several random sequences of numSamleEvents (initially set to
			// 100, but the code will support changing this value to any integer) of events from the LP's
			// list of events.
			
			// the outcome of this will be the number of samples and then the number of events counted in
			// each sampled locations in the event list.  if a sample finishes the list of events in the
			// LP before reaching the end of the time interval, it is rejected and a new count is started
			// at a different location.
			
			numEventDeltaSamples := 10
			numSampleEvents := 100

			// ok, this analysis requires that the LP have processed at least numSampleEvents; if it
			// doesn't, what do we do??  of course we wouldn't normally expect to encounter this, but i've
			// already run into it with testing (fortunately).  to my mind, we have two choices, either
			// (i) reduce the numSampleEvents or (ii) skip this part of the analysis and have the
			// visualization tools deal with the missing data.  for now, i'm gonna take the second option
			// and simply output -1's for the results; actually i'm gonna require that the LP have at
			// least numEventDeltaSamples times the numSampleEvents before the real computation
			// actually occurs....why i suppose smaller would still suffice, but let's take a more
			// conservative stance
			if len(lp.events) < (numEventDeltaSamples * numSampleEvents) {
				fmt.Fprintf(outFile, "%v, %v", numEventDeltaSamples, -1.0)
				for i := 0; i < numEventDeltaSamples; i++ {fmt.Fprintf(outFile, ", %v", -1.0)}
				fmt.Fprintf(outFile, "\n")
				continue
			}

			// ok, so let's get the mean of the time interval for numSampleEvents at various points of the
			// executed events
			mean = 0.0
			variance = 0.0
			for i := 0; i < numEventDeltaSamples; i++ {
				// give us a random number to start our index into the event list of the LP; but let's
				// skip the first event as it sometimes holds an event from an init state
				start := int(rand.Int31n(int32(len(lp.events) - (numSampleEvents + 2)))) + 1
				mean, variance = updateRunningMeanVariance(mean, variance,
					lp.events[start + numSampleEvents].receiveTime - lp.events[start].receiveTime, i+1)
			}				
			variance = variance / float64(numEventDeltaSamples - 1)
			
			fmt.Fprintf(outFile, "%v, %v", numEventDeltaSamples, mean)
			sampleCount := 0
			allSamplesLoop: for {
				// have we captured the necessary samples?
				if sampleCount == numEventDeltaSamples {break}
				// again, let's ensure that all samples are taken after the first event and try to
				// avoid using initial state events in this computation
				start := int(rand.Int31n(int32(len(lp.events) - 1))) + 1
				startTime := lp.events[start].receiveTime
				count := 1
				oneSampleLoop: for {
					// we've walked off the end of the event list, reject this sample
					if ((start + count) > (len(lp.events) - 1)) {continue allSamplesLoop}
					if (lp.events[start + count].receiveTime <= (startTime + mean)) {
						// accept event as a member of this sample
						count++
					} else {
						// we've found the end of a complete sample
						break oneSampleLoop
					}
				}
				// record this sample
				fmt.Fprintf(outFile, ", %v", count)
				sampleCount++
			}
			fmt.Fprintf(outFile, "\n")
		}
		
		err = outFile.Close()
		if err != nil {panic(err)}

		// now we will create a total events processed file which will keep track of receiving LP, num events
		// processed, min timestamp delta, max timestamp delta, average timestamp delta, standard deviation
		// (of the mean).
		
		outFile, err = os.Create("analysisData/totalEventsProcessed.csv")
		if err != nil {panic(err)}
		
		fmt.Fprintf(outFile, "# Total Events Processed Data (per LP)\n")
		fmt.Fprintf(outFile, "# receiving LP, number of events processed, min timestamp delta, max timestamp delta, average timestamp delta, standard deviation.\n")
		
		lpEventReceivedCount := make([]int, numOfLPs)
		numEventsProcessed := make([]int, numOfLPs)
		minTimeDelta := make([]float64, numOfLPs)
		maxTimeDelta := make([]float64, numOfLPs)
		aveTimeDelta := make([]float64, numOfLPs)
		stanDeviation := make([]float64, numOfLPs)
		//currLP := -1                  // This shows where the process is at for analysis.
		for j, lp:= range(lps) {
			// j is the received LP
			
			// set all values to zero
			for i := range lpEventReceivedCount {
				lpEventReceivedCount[i] = 0
				numEventsProcessed[i] = 0
				minTimeDelta[i] = math.MaxFloat64
				maxTimeDelta[i] = 0
				aveTimeDelta[i] = 0
				stanDeviation[i] = 0
			}
			for _, event := range lp.events {
				lpEventReceivedCount[j]++
				delta := event.receiveTime - event.sendTime
				aveTimeDelta[j] += delta
				if minTimeDelta[j] > delta {
					minTimeDelta[j] = delta
				}
				if maxTimeDelta[j] < delta {
					maxTimeDelta[j] = delta
				}
			}
			for _, event := range lp.events {
				stanDeviation[j] += math.Pow(((event.receiveTime - event.sendTime) - (aveTimeDelta[j]/float64(lpEventReceivedCount[j]))), 2)
			}
			
			for i := range lpEventReceivedCount {
				//              fmt.Println("Here")
				if lpEventReceivedCount[i] != 0 {
					fmt.Fprintf(outFile, "%v,%v,%v,%v,%v,%v\n", j, lpEventReceivedCount[j], minTimeDelta[j], maxTimeDelta[j], aveTimeDelta[j]/float64(lpEventReceivedCount[j]), math.Sqrt(stanDeviation[j]/float64(lpEventReceivedCount[j])))
				}
			}
			//err = outFile.Close()
			//if err != nil {panic(err)}
			//fmt.Println(err)
		}
	}

	// events available for execution: here we will assume all events execute in unit time and evaluate the
	// events as executable by simulation cycle.  basically we will advance the simulation time to the lowest
	// receive time of events in all of the LPs and count the number of LPs that could potentially be executed
	// at that time.  the general algorithm is outlined in the indexTemplate.md file for this project.

	fmt.Printf("%v: Analysis (parallel) of event parallelism available by simulation cycle (potential parallelism).\n", getTime())	

	// so we are going to do two parts of the simulation cycle analysis in each pass, namely: (i) count the
	// numbber of events available for execution, and (ii) find the lowest timestamp for the next simulation
	// cycle.  we are also going to run this analysis in parallel (and the two steps simultaneously) by
	// partitioning the LPs among the threads. 

	// using this data structure to hold cycle by cycle analysis results.  
	type simCycleAnalysisResults struct {
		definingLP int
		timeStamp float64
		numAvailable int
		eventsExhausted bool
	}

	// this function is setup to simultaneously (i) find the number of events that are available for execution
	// at a fixed schedule time and (ii) find the next (minimum) time for the next simulation cycle (by
	// assuming that all available events would be executed).
	analyzeSimCycle := func(lps []lpData, scheduleTime float64, definingLP int) simCycleAnalysisResults {
		var results simCycleAnalysisResults
		results.timeStamp = math.MaxFloat64
		results.eventsExhausted = true
		for _, lp := range lps {
			if lpIndex[lp.lpId] < len(lp.events) {
				// accumulate events available at this simulation time
				if (lp.events[lpIndex[lp.lpId]].sendTime < scheduleTime || definingLP == lp.lpId) {
					results.numAvailable++
					lpIndex[lp.lpId]++
				}
				// search for the time for the next simulation cycle
				if lpIndex[lp.lpId] < len(lp.events) { // since we potentially incremented the lpIndex ptr
					if results.timeStamp > lp.events[lpIndex[lp.lpId]].receiveTime {
						results.timeStamp = lp.events[lpIndex[lp.lpId]].receiveTime
						results.definingLP = lp.lpId
					}
				}
				results.eventsExhausted = false
			}
		}
		return results
	}
	
	// this function serves as the parallel thread function to communicate back and forth with the main thread.
	analyzeSimulationCycle := func(lps []lpData, c1 <-chan simCycleAnalysisResults, c2 chan<- simCycleAnalysisResults) {
		for ; ; {
			nextCycle :=<- c1
			if nextCycle.eventsExhausted == true {close(c2); return} // done
			results := analyzeSimCycle(lps, nextCycle.timeStamp, nextCycle.definingLP)
			c2 <- results
		}
	}
	
	outFile, err = os.Create("analysisData/eventsAvailableBySimCycle.csv")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# events available by simulation cycle\n")
	fmt.Fprintf(outFile,"# num of events ready for execution\n")

	// setup/start the goroutines for simulation cycle analysis
	in := make([]chan simCycleAnalysisResults, numThreads)
	out := make([]chan simCycleAnalysisResults, numThreads)
	for i := 0; i < numThreads; i++ {
		in[i] = make(chan simCycleAnalysisResults)
		out[i] = make(chan simCycleAnalysisResults)
	}

	// start the threads for simulation cycle analysis
	for i := 0; i < numThreads; i++ {
		low := i * goroutineSliceSize
		high := low + goroutineSliceSize
		if i == numThreads - 1 {high = len(lps)}
		go analyzeSimulationCycle(lps[low:high], in[i], out[i])
	}

	// initialize the first data to send into the simulation cycle goroutines
	var nextCycle simCycleAnalysisResults
	nextCycle.timeStamp = math.MaxFloat64
	for _, lp := range lps {
		// if lp has no events, do nothing
		if len(lp.events) > 0 {
			if lp.events[0].receiveTime < nextCycle.timeStamp {
				nextCycle.timeStamp = lp.events[0].receiveTime
				nextCycle.definingLP = lp.lpId
			}
		}
		lpIndex[lp.lpId] = 0 // reset these pointers
	}

	// process data to/from the simulation cycle analysis threads
	simCycle := 0
	maxEventsAvailable := 0
	timesXEventsAvailable := make([]int, numOfEvents)
	for ; ; {
		for i := 0; i < numThreads; i++ {in[i] <- nextCycle}
		nextCycle.timeStamp = math.MaxFloat64
		nextCycle.eventsExhausted = true
		nextCycle.numAvailable = 0
		for i := 0; i < numThreads; i++ {
			results := <- out[i]
			if results.timeStamp < nextCycle.timeStamp {
				nextCycle.timeStamp = results.timeStamp
				nextCycle.definingLP = results.definingLP
			}
			if results.eventsExhausted == false {
				nextCycle.eventsExhausted = false
				nextCycle.numAvailable = nextCycle.numAvailable + results.numAvailable
			}
		}
		if nextCycle.eventsExhausted == true {break}
		fmt.Fprintf(outFile,"%v\n",nextCycle.numAvailable)
		timesXEventsAvailable[nextCycle.numAvailable]++
		if maxEventsAvailable < nextCycle.numAvailable {maxEventsAvailable = nextCycle.numAvailable}
		simCycle++
	}

	err = outFile.Close()
	if err != nil {panic(err)}

	// write out summary of events available
	outFile, err = os.Create("analysisData/timesXeventsAvailable.csv")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# times X events are available for execution\n")
	fmt.Fprintf(outFile,"# X, num of occurrences\n")
	for i := 0; i < maxEventsAvailable; i++ {fmt.Fprintf(outFile,"%v,%v\n",i+1,timesXEventsAvailable[i+1])}
	err = outFile.Close()
	if err != nil {panic(err)}

	if analyzeAllData || analyzeReceiveTimeData {analyzeEventsByReceiveTime()}

	fmt.Printf("%v: Finished.\n", getTime())
	return

} 
