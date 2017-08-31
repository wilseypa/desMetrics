
// this program performs the analysis for the desMetrics project at UC
// (http://github.com/wilseypa/desMetrics).  this program inputs a json file containing profile data of the
// events processed by a discrete event simulation engine and outputs cvs/data files containing various
// reports on the characteristics of the events from that profile data.  this project is developed from a
// parallel simulation (PDES) perspective and so much of the jargon and analysis is related to that field.  to
// understand the documentation, familiarity with PDES terminology is essential.  the input json file and
// overall project perspective is available from the project website.

// operationally, this program parses the input json file twice, the first pass captures the general
// characteristics of the file such as number of LPs, total number of events and so on.  the second pass
// inputs and stores the event data into internal data structures for processing.  this approach is followed
// to maintain the memory footprint as these files tend to be quite large.  memory and time are issues so the
// program is organized accordingly.  in particular, whenever possible, the analysis is partitioned and
// performed in parallel threads.  the json parse is done sequentially (while it could be probably done in
// parallel, it's run time does not currently merit the need).  the program is setup to use all cores on the
// host processor so plan accordingly.

package main

import "os"
import "fmt"
import "sort"
//import "math"
import "strconv"
import "time"
import "runtime"
import "flag"

// setup a data structure for events.  internally we're going to store LP names with their integer map value.
// since we're storing events into an array indexed by the LP in question (sender or receiver), we will only
// store the other "companion" LP internally.
type eventData struct {
	companionLP int
	sendTime float64
	receiveTime float64
}

// setup a data structure for LPs; each LP has a unique id and a list of events it generates.
type lpData struct {
	lpId int
	events []eventData
}

// functions to support sorting of the events by their receive time
type byReceiveTime []eventData
func (a byReceiveTime) Len() int           {return len(a)}
func (a byReceiveTime) Swap(i, j int)      {a[i], a[j] = a[j], a[i]}
func (a byReceiveTime) Less(i, j int) bool {return a[i].receiveTime < a[j].receiveTime}

func main() {

	var commSwitchOff bool
	var debug bool
	flag.BoolVar(&commSwitchOff, "no-comm-matrix", false, "turn off generation of the file eventsExchanged.csv")
	flag.BoolVar(&debug, "debug", false, "turn on debugging.")
	flag.Parse()

	// enable the use of all CPUs on the system
	numThreads := runtime.NumCPU()
	runtime.GOMAXPROCS(numThreads)

	// function to print time as the program reports progress to stdout
	printTime := func () string {return time.Now().Format(time.RFC850)}

	fmt.Printf("%v: Parallelism setup to support up to %v threads.\n", printTime(), numThreads)

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

	// record a unique int value for each LP and store the total number of sent and received events by that LP.
	type lpMap struct {
		toInt int
		sentEvents int
		receivedEvents int
	}

	lpNameMap := make(map[string]*lpMap)

	// ultimately we will use lps to hold event data; and lpIndex to record advancements of analysis among the lps
	var lps []lpData
	var lpIndex []int

	
	// make a directory storing files generated from streaming the data
	var err error
	err =  os.MkdirAll ("streamedData", 0777)
	if err != nil {panic(err)}
	
	// during the first pass over the JSON file: record all new LPs into the lpNameMap and count
	// the number of events sent/received by each LP; also count the total number of events in
	// the simulation 
	//processEvent := func(sLP string, sTS float64, rLP string, rTS float64) {
		//numOfEvents++
		//lp := defineLP(sLP)
		//lp.sentEvents++
		//lp = defineLP(rLP)
		//lp.receivedEvents++
	//}

	// temporary data structure for semi-streaming on first pass
	//procEvent := func(sendingLP string, sendTime string, receivingLP string, receiveTime string){
	//	sLP := sendingLP
	//	sT := sentTime
	//	rLP := receivingLP
	//	rT := receiveTime
	//}

	printInfo := func (format string, a ...interface{}) (n int, err error) {
		n, err = fmt.Printf(format, a)
		return n, err
	}

	simulatorName := ""
	modelName := ""
	captureDate := ""
	commandLineArgs := ""
	// this is the parser we will use
	parseJsonFile := func(inputFile *os.File) {

		// initialize the scanner
		ScanInit(inputFile)
		
		var token int
		var tokenText []byte
		
		// helper function
		scanAssume:= func(expectedToken int) {
			if expectedToken != token {
				fmt.Printf("Mal formed json file at line: %v\n", fileLineNo)
				panic("Aborting")
			}
			token, tokenText = Scan()
		}
		
		outFile, err := os.Create("streamedData/eventInfo.csv")
		if err != nil {panic(err)}
		
		// parse the json data file, this is hugely fragile but functional
		token, tokenText = Scan()
		scanAssume(L_CURLY)
		parsingLoop: for ; token != EOF; {
			switch token {
			case SIMULATOR_NAME:
				token, tokenText = Scan()
				scanAssume(COLON)
				printInfo("    Simulator Name: %v\n", string(tokenText))
				simulatorName = string(tokenText)
				token, tokenText = Scan()
				scanAssume(COMMA)
			case MODEL_NAME:
				token, tokenText = Scan()
				scanAssume(COLON)
				printInfo("    Model Name: %v\n", string(tokenText))
				modelName = string(tokenText)
				token, tokenText = Scan()
				scanAssume(COMMA)
			case CAPTURE_DATE:
				token, tokenText = Scan()
				scanAssume(COLON)
				printInfo("    Capture date: %v\n", string(tokenText))
				captureDate = string(tokenText)
				token, tokenText = Scan()
				scanAssume(COMMA)
			case COMMAND_LINE_ARGS :
				token, tokenText = Scan()
				scanAssume(COLON)
				printInfo("    Command line arguments: %v\n", string(tokenText))
				commandLineArgs = string(tokenText)
				token, tokenText = Scan()
				scanAssume(COMMA)
			case EVENTS:
				token, tokenText = Scan()
				scanAssume(COLON)
				scanAssume(L_BRACKET)
				
				for ; token != EOF ; {
					var sendingLP, receivingLP string
					var sendTime, receiveTime float64
					var err error

					scanAssume(L_BRACKET) // sets token, tokenText of next token
					sendingLP = string(tokenText)
					token, tokenText = Scan()
					scanAssume(COMMA)
					if tokenText[0] == 34 { // quick and dirty removal of surrounding "" if they exist
						tokenText = append(tokenText[:0], tokenText[1:]...)
						tokenText = tokenText[:len(tokenText)-1]
					}
					sendTime, err = strconv.ParseFloat(string(tokenText),64)
					if err != nil {panic(err)}
					token, tokenText = Scan()
					scanAssume(COMMA)
					receivingLP = string(tokenText)
					token, tokenText = Scan()
					scanAssume(COMMA)
					if tokenText[0] == 34 { // quick and dirty removal of surrounding "" if they exist
						tokenText = append(tokenText[:0], tokenText[1:]...)
						tokenText = tokenText[:len(tokenText)-1]
					}
					receiveTime, err = strconv.ParseFloat(string(tokenText),64)
					if err != nil {panic(err)}
					token, tokenText = Scan()
					scanAssume(R_BRACKET)
					// we should require that the send time be strictly less than the
					// receive time, but the ross (airport model) data actually has data
					// with the send/receive times (and other sequential simulators are
					// likely to have this as well), so we will weaken this constraint.
					if sendTime > receiveTime {
						fmt.Printf("Event has send time greater than receive time: %v %v %v %v\n", 
							sendingLP, sendTime, receivingLP, receiveTime)
						panic("Aborting")
					}
					if debug {fmt.Printf("Event recorded: %v, %v, %v, %v\n", sendingLP, sendTime, receivingLP, receiveTime)}
					
					numOfEvents++
					
					fmt.Fprintf(outFile,"%v, %v, %v, %v\n", sendingLP, sendTime, receivingLP, receiveTime)
					
					if token == R_BRACKET {break parsingLoop}
					scanAssume(COMMA)
				}
			default:
				// this should never happen
				fmt.Printf("Mal formed json file at %v\n",fileLineNo)
				panic("Aborting")
			}
		}
		err = outFile.Close()
		if err != nil {panic(err)}
	}


	// this time we will collect information on the number of events and the number of LPs
	fmt.Printf("%v: Processing %v to capture event and LP counts.\n", 
		printTime(), flag.Args())
	inputFile, err := os.Open(flag.Arg(0))
	if err != nil { panic(err) }
	parseJsonFile(inputFile)
	err = inputFile.Close()

	//exec.Command("sort","k 4","streamedData/eventInfo.csv").Output()

	// reset this function so we don't print the model information in the second parse
	printInfo = func (format string, a ...interface{}) (n int, err error) {return 0, nil}

	// lps is an array of the LPs; each LP entry will hold the events it received
	lps = make([]lpData, len(lpNameMap))

	// periodically we will need to walk through the LP arrays independently; we will use this array to do so
	lpIndex = make([]int, len(lps))
	// allocate entries in each LP to hold the number events it received
	for _, i := range lpNameMap {
		lpIndex[i.toInt] = -1
		lps[i.toInt].lpId = i.toInt
		lps[i.toInt].events = make([]eventData, i.receivedEvents)
	}

	// this time we will save the events
	fmt.Printf("%v: Processing %v to capture events.\n", printTime(), flag.Arg(0))
	//processEvent = addEvent
	inputFile, err = os.Open(flag.Arg(0))
	if err != nil { panic(err) }
	parseJsonFile(inputFile)
	err = inputFile.Close()

	// build the reverse map: integers -> LP Names
	mapIntToLPName := make([]string, numOfLPs)
	for key, value := range lpNameMap {mapIntToLPName[value.toInt] = key}

	// let's check to see if all LPs received an event (not necessarily a huge problem, but something we
	// should probably be aware of.  also, record the max num of events received by an LP
	fmt.Printf("%v: Verifying that all LPs recieved at least one event.\n", printTime())
	maxLPEventArray := 0
	for i := range lps {
		if len(lps[i].events) == 0 {
			fmt.Printf("WARNING: LP %v recived zero messages.\n", mapIntToLPName[i])
		}
		if maxLPEventArray < len(lps[i].events) {maxLPEventArray = len(lps[i].events)}
	}

	// we now need to sort the event lists by receive time.  for this we'll use the sort package.
	fmt.Printf("%v: Sorting the events in each LP by receive time.\n", printTime())
	for i := range lps {sort.Sort(byReceiveTime(lps[i].events))}

	// on to analysis....

	// create the directory to store the resulting data files (if it is not already there)
	err =  os.MkdirAll ("analysisData", 0777)
	if err != nil {panic(err)}

	outFile, err := os.Create("analysisData/modelSummary.json")
	if err != nil {panic(err)}

        fmt.Fprintf(outFile,"{\n  \"simulator_name\" : %v,\n", simulatorName)
        fmt.Fprintf(outFile,"  \"model_name\" : %v,\n", modelName)
        fmt.Fprintf(outFile,"  \"capture_date\" : %v,\n", captureDate)
        fmt.Fprintf(outFile,"  \"command_line_args\" : %v,\n", commandLineArgs)
        fmt.Fprintf(outFile,"  \"total_lps\" : %v,\n",numOfLPs)
        fmt.Fprintf(outFile,"  \"total_events\" : %v\n}",numOfEvents)

	err = outFile.Close()
	if err != nil {panic(err)}

	fmt.Printf("%v: Analysis (parallel) of events organized by receiving LP.\n", printTime())	
	return
}

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
		if lp.events[0].receiveTime < nextCycle.timeStamp {
			nextCycle.timeStamp = lp.events[0].receiveTime
			nextCycle.definingLP = lp.lpId
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

	// write out summary of events available
	outFile, err = os.Create("analysisData/timesXeventsAvailable.csv")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# times X events are available for execution\n")
	fmt.Fprintf(outFile,"# X, num of occurrences\n")
	for i := 0; i < maxEventsAvailable; i++ {fmt.Fprintf(outFile,"%v,%v\n",i+1,timesXEventsAvailable[i+1])}
	err = outFile.Close()
	if err != nil {panic(err)}

	fmt.Printf("%v: Finished.\n", printTime())
	return
