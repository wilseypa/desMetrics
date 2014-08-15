
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
import "math"
import "strconv"
import "time"
import "runtime"

// setup a data structure for events.  internally we're going to store LP names with their integer map value.
// since we're storing events into an array indexed by the LP in question (sender or receiver), we will only
// store the other "companion" LP internally
type eventData struct {
	companionLP int
	sendTime float64
	receiveTime float64
}

// setup a data structure for LPs
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

	// enable the use of all CPUs on the system
	numThreads := runtime.NumCPU()
	runtime.GOMAXPROCS(numThreads)

	// function to print time as the program reports progress to stdout
	printTime := func () string {return time.Now().Format(time.RFC850)}

	// memory is a big issue.  in order to minimize the size of our data structures, we will process the
	// input JSON file twice.  the first time we will build a map->int array for the LPs, count the total
	// number of LPs, and the total number of events.  we will use these counts to build the principle
	// data structures to hold events by LPs (by assuming that the events are more or less uniformly
	// distributed among the LPs.  in the second pass, we will store the events into our internal data
	// structures.  we will use the function processEvent to mahage each step.  between parses it will be
	// changed to point to the addEvent function.
	numOfLPs := 0
	numOfEvents := 0

	type lpMap struct {
		toInt int
		sentEvents int
		receivedEvents int
	}

	lpNameMap :=make(map[string]*lpMap)

	// ultimately we will use these to hold event data
	var lps []lpData
	var lpIndex []int

	addLP := func(lp string) *lpMap {
		item, present := lpNameMap[lp]
		if !present {
			lpNameMap[lp] = new(lpMap)
			lpNameMap[lp].toInt = numOfLPs
			item = lpNameMap[lp]
			numOfLPs++
		}
		return item
	}
	// this is the event processing function for the first pass
	processEvent := func(sLP string, sTS float64, rLP string, rTS float64) {
		// count the events
		numOfEvents++
		// build a map of LPs to ints and record the number of sent and received events at each
		lp := addLP(sLP)
		lp.sentEvents++
		lp = addLP(rLP)
		lp.receivedEvents++
	}
	// this is the event processing function for the second pass
	addEvent := func(sLP string, sTS float64, rLP string, rTS float64) {
		rLPint := lpNameMap[rLP].toInt
		lpIndex[rLPint]++
		if lpIndex[rLPint] > cap(lps[rLPint].events) {panic("Something wrong, we should have computed the appropriate size on the first parse.\n")}
		lps[rLPint].events[lpIndex[rLPint]].companionLP = lpNameMap[sLP].toInt
		lps[rLPint].events[lpIndex[rLPint]].receiveTime = rTS
		lps[rLPint].events[lpIndex[rLPint]].sendTime = sTS
	}

	printInfo := func (format string, a ...interface{}) (n int, err error) {
		n, err = fmt.Printf(format, a)
		return n, err
	}

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
		
		// parse the json data file, this is hugely fragile but functional
		token, tokenText = Scan()
		scanAssume(L_CURLY)
		parsingLoop: for ; token != EOF; {
			switch token {
			case SIMULATOR_NAME:
				token, tokenText = Scan()
				scanAssume(COLON)
				printInfo("    Simulator Name: %v\n", string(tokenText))
				token, tokenText = Scan()
				scanAssume(COMMA)
			case MODEL_NAME:
				token, tokenText = Scan()
				scanAssume(COLON)
				printInfo("    Model Name: %v\n", string(tokenText))
				token, tokenText = Scan()
				scanAssume(COMMA)
			case CAPTURE_DATE:
				token, tokenText = Scan()
				scanAssume(COLON)
				printInfo("    Capture date: %v\n", string(tokenText))
				token, tokenText = Scan()
				scanAssume(COMMA)
			case COMMAND_LINE_ARGS :
				token, tokenText = Scan()
				scanAssume(COLON)
				printInfo("    Command line arguments: %v\n", string(tokenText))
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
					scanAssume(L_CURLY)
					for i := 0; i < 4; i++ {
						if i > 0 {scanAssume(COMMA)}
						switch token {
						case SEND_LP:
							token, tokenText = Scan()
							scanAssume(COLON)
							sendingLP = string(tokenText)
							token, tokenText = Scan()
						case SEND_TIME:
							token, tokenText = Scan()
							scanAssume(COLON)
							sendTime, err = strconv.ParseFloat(string(tokenText),64)
							if err != nil {panic(err)}
							token, tokenText = Scan()
						case RECEIVE_LP:
							token, tokenText = Scan()
							scanAssume(COLON)
							receivingLP = string(tokenText)
							token, tokenText = Scan()
						case RECEIVE_TIME:
							token, tokenText = Scan()
							scanAssume(COLON)
							receiveTime, err = strconv.ParseFloat(string(tokenText),64)
							if err != nil {panic(err)}
							token, tokenText = Scan()
						}
					}
					scanAssume(R_CURLY)
					// we should require that the send time be strictly less than the
					// receive time, but the ross (airport model) data actually has data
					// with the send/receive times (and other sequential simulators are
					// likely to have this as well), so we will weaken this constraint.
					if sendTime > receiveTime {
						fmt.Printf("Event has send time greater than receive time: %v %v %v %v\n", 
							sendingLP, sendTime, receivingLP, receiveTime)
						panic("Aborting")
					}
					processEvent(sendingLP, sendTime, receivingLP, receiveTime)
					if token == R_BRACKET {break parsingLoop}
					scanAssume(COMMA)
				}
			default:
				// this should never happen
				fmt.Printf("Mal formed json file at %v\n",fileLineNo)
				panic("Aborting")
			}
		}
	}

	// this time we will collect information on the number of events and the number of LPs
	fmt.Printf("%v: Processing %v to capture event and LP counts.\n", 
		printTime(), os.Args[1])
	inputFile, err := os.Open(os.Args[1])
	if err != nil { panic(err) }
	parseJsonFile(inputFile)
	err = inputFile.Close()

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
	fmt.Printf("%v: Processing %v to capture events.\n", printTime(), os.Args[1])
	processEvent = addEvent
	inputFile, err = os.Open(os.Args[1])
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
	fmt.Printf("%v: Analysis (parallel) of events organized by receiving LP.\n", printTime())	

	// create the directory to store the resulting data files (if it is not already there)
	err =  os.MkdirAll ("analysisData", 0777)
	if err != nil {panic(err)}

	// in this next step, we are going to compute event received summaries.  in this case we are going to
	// attack the problem by slicine the lps array and assigning each slice to a separate goroutine.  to
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
	// result in coverage of the number of events received (yes, this is confusing)
	numOfLPsToCover := func(total int, data []int, percent int) int {
		// add .99 because int() truncates and we actually want rounding
		numRequired := int(((float32(total) * float32(percent)) / float32(100)) + .99) 
		lpCount := 0
		eventCount := 0
		for i := range data {
			lpCount++
			eventCount = eventCount + data[i]
			if eventCount >= numRequired {return lpCount}
		}
		panic("ERROR: something's amiss in this computation\n")
	}

	// data type to capture each LP's event summary data
	type lpEventSummary struct {
		lpId int
		local int
		remote int
		total int
		cover [5]int
	}

	// compute the local/remote events and cover statistics
	computeLPEventsProcessed := func(lp lpData) lpEventSummary {
		var es lpEventSummary
		lpSenders := make([]int,numOfLPs)
		es.lpId = lp.lpId
		es.local = 0
		es.remote = 0
		for _, event := range lp.events {
			lpSenders[event.companionLP]++
			if lp.lpId != event.companionLP {es.remote++} else {es.local++}
		}
		sort.Sort(sort.Reverse(sort.IntSlice(lpSenders)))
		es.total = es.remote + es.local
		es.cover[0] = numOfLPsToCover(es.total, lpSenders, 75)  //  75% cutoff
		es.cover[1] = numOfLPsToCover(es.total, lpSenders, 80)  //  80% cutoff
		es.cover[2] = numOfLPsToCover(es.total, lpSenders, 90)  //  90% cutoff
		es.cover[3] = numOfLPsToCover(es.total, lpSenders, 95)  //  95% cutoff
		es.cover[4] = numOfLPsToCover(es.total, lpSenders, 100) // 100% cutoff
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

	type lpChainSummary struct {
		lpId int
		local []int
		linked []int
		global []int
	}
	chainLength := 5

	accumulateChain := func(chain []int, maxChainLength int, chainLength int) {
		if chainLength >= maxChainLength {chain[maxChainLength - 1]++} else {chain[chainLength]++}
		return
	}

	// PAW: consider changing these to being strictly less than the receiveTime
	computeEventChains := func(lp lpData) lpChainSummary {

		var lpChain lpChainSummary

		// all chains lengths >= chainLength will be counted together
		lpChain.lpId = lp.lpId
		lpChain.local = make([]int,chainLength)
		lpChain.linked = make([]int,chainLength)
		lpChain.global = make([]int,chainLength)

		// LOCAL CHAINS: an event is part of the local chain if it is (i) generated by this LP (lpId) and (ii)
		// if it's send time is less than or equal to the receive time of the event at the head of the chain.
		i := 0
		for ; i < len(lp.events) ; {
			for ; i < len(lp.events) && lp.events[i].companionLP != lp.lpId ; i++ {}
			if i < len(lp.events) && lp.events[i].companionLP == lp.lpId {
				j := i + 1
				for ; j < len(lp.events) && lp.events[j].companionLP == lp.lpId && lp.events[j].sendTime <= lp.events[i].receiveTime ; {j++}
				accumulateChain(lpChain.local, chainLength, j-i-1)
				i = i + j
			}
		}
		
		// LINKED CHAINS: an event is part of the linked chain if it is (i) generated by this LP (lpId) and
		// (ii) if it's send time is less than or equal to the receive time of any event currently in the
		// chain (tested against the last event already in the chain)
		i = 0
		for ; i < len(lp.events) ; {
			for ; i < len(lp.events) && lp.events[i].companionLP != lp.lpId ; i++ {}
			if i < len(lp.events) && lp.events[i].companionLP == lp.lpId {
				j := i + 1
				for ; j < len(lp.events) && lp.events[j].companionLP == lp.lpId && lp.events[j].sendTime <= lp.events[j-1].receiveTime ; {j++}
				accumulateChain(lpChain.linked, chainLength, j-i-1)
				i = i + j
			}
		}
		
		// GLOBAL CHAINS: an event is part of the global chain if it is has a send time that is less than or
		// equal to the receive time of any event currently in the chain (tested against the last event
		// already in the chain)
		i = 0
		for ; i < len(lp.events) ; {
			j := i + 1
			for ; j < len(lp.events) && lp.events[j].sendTime <= lp.events[i].receiveTime ; {j++}
			accumulateChain(lpChain.global, chainLength, j-i-1)
			i = i + j
		}
		return lpChain
	}

	// PAW: change the comment headers of chains/covers to for loops so the numbers will automatically grow
	// with the variable setting.... 
	localChainFile, err := os.Create("analysisData/localEventChainsByLP.dat")
	fmt.Fprintf(localChainFile,"# local event chains by LP\n")
	fmt.Fprintf(localChainFile,"# LP, local chains of length: 1, 2, 3, 4, >= 5\n")

	linkedChainFile, err := os.Create("analysisData/linkedEventChainsByLP.dat")
	fmt.Fprintf(linkedChainFile,"# linked event chains by LP\n")
	fmt.Fprintf(linkedChainFile,"# LP, linked chains of length: 1, 2, 3, 4, >= 5\n")

	globalChainFile, err := os.Create("analysisData/globalEventChainsByLP.dat")
	fmt.Fprintf(globalChainFile,"# global event chains by LP\n")
	fmt.Fprintf(globalChainFile,"# LP, global chains of length: 1, 2, 3, 4, >= 5\n")

	// location to write summaries of local and remote events received
	eventSummaries, err := os.Create("analysisData/eventsExecutedByLP.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(eventSummaries, "# summary of local and remote events executed\n")
	fmt.Fprintf(eventSummaries, "# LP, local, remote\n")

	// location to write percentage of LPs to cover percentage of events received
	numToCover, err := os.Create("analysisData/numOfLPsToCoverPercentTotalMessages.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(numToCover,"# number of destination LPs (sorted by largest messages sent to) to cover percentage of total events\n")
	fmt.Fprintf(numToCover,"# LP name, total events sent, num of LPs to cover: 75, 80, 90, 95, and 100 percent of the total events sent.\n")

	// PAW, if you want to combine these results into one large csv file you will have to
	// combine c1/c2 so that the data pops out in a chennel together for each LP.

	// this will be invoked as a gogroutine with LPs equally (nearly) partitioned among them
	analyzeReceivedEvents := func (lps []lpData, c1 chan<- lpEventSummary, c2 chan <- lpChainSummary) {
		for _, lp := range(lps) {
			eventsProcessed := computeLPEventsProcessed(lp)
			chains := computeEventChains (lp)
			c1 <- eventsProcessed
			c2 <- chains
		}
	}

	// defining how many LPs do we assign to each thread
	goroutineSliceSize := int((float32(len(lps))/float32(numThreads)) + .5)

	// each goroutine will compute event counts for one LP, send the results back over the channel and
	// continue. 
	c1 := make(chan lpEventSummary, numThreads * 4)
	c2 := make(chan lpChainSummary, numThreads * 4)
	for i := 0; i < numThreads; i++ {
		low := i * goroutineSliceSize
		high := low + goroutineSliceSize
		if i == numThreads - 1 {high = len(lps)}
		go analyzeReceivedEvents(lps[low:high], c1, c2)
	}

	localEventChainSummary := make([]int,chainLength)
	linkedEventChainSummary := make([]int,chainLength)
	globalEventChainSummary := make([]int,chainLength)

	// process all of the data in the channel
	for _ = range lps {
		eventsProcessed := <- c1
		chains := <- c2
		// capture event chain summaries
		for i := 0; i < chainLength; i++ {
			localEventChainSummary[i] = localEventChainSummary[i] + chains.local[i]
			linkedEventChainSummary[i] = linkedEventChainSummary[i] + chains.linked[i]
			globalEventChainSummary[i] = globalEventChainSummary[i] + chains.global[i]
		}
		fmt.Fprintf(eventSummaries,"%v, %v, %v\n", 
			mapIntToLPName[eventsProcessed.lpId], eventsProcessed.local, eventsProcessed.remote)
		// PAW: turn this into a for loop so the variable will actually control
		fmt.Fprintf(numToCover,"%v, %v", mapIntToLPName[eventsProcessed.lpId], eventsProcessed.total)
		for _, i := range eventsProcessed.cover {fmt.Fprintf(numToCover,", %v", i)}
		fmt.Fprintf(numToCover,"\n")
		fmt.Fprintf(localChainFile,"%v",mapIntToLPName[chains.lpId])
		fmt.Fprintf(linkedChainFile,"%v",mapIntToLPName[chains.lpId])
		fmt.Fprintf(globalChainFile,"%v",mapIntToLPName[chains.lpId])
		for i := range chains.local {
			fmt.Fprintf(localChainFile,", %v", chains.local[i])
			fmt.Fprintf(linkedChainFile,", %v", chains.linked[i])
			fmt.Fprintf(globalChainFile,", %v", chains.global[i])
		}
        fmt.Fprintf(localChainFile,"\n")
        fmt.Fprintf(linkedChainFile,"\n")
        fmt.Fprintf(globalChainFile,"\n")
	}
	close(c1)
	close(c2)

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
	outFile, err := os.Create("analysisData/eventChainsSummary.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# number of event chains of length X\n")
	fmt.Fprintf(outFile,"# chain length, num of local chains, num of linked chains, num of global chains\n")
	for i := 0; i < chainLength; i++ {
		fmt.Fprintf(outFile,"%v, %v, %v, %v\n", i+1,
			localEventChainSummary[i],linkedEventChainSummary[i],globalEventChainSummary[i])
	}
	err = outFile.Close()
	if err != nil {panic(err)}

	// events available for execution: here we will assume all events execute in unit time and evaluate the
	// events as executable by simulation cycle.  basically we will advance the simulation time to the lowest
	// receive time of events in all of the LPs and count the number of LPs that could potentially be executed
	// at that time.  the general algorithm is outlined in the indexTemplate.md file for this project.

	fmt.Printf("%v: Computing event parallelism statistics.\n", printTime())	

	type lowestTimeStampFound struct {
		definingLP int
		minTS float64
		found bool
	}

	// find the lowest timestamped event at the head of each LP slice we are given 
	findLowestTS := func(lps []lpData) lowestTimeStampFound {
		var findResults lowestTimeStampFound
		findResults.definingLP = 0
		findResults.minTS = math.MaxFloat64
		findResults.found = false
		for _, lp := range lps {
			if lpIndex[lp.lpId] < len(lp.events) {
				if findResults.minTS > lp.events[lpIndex[lp.lpId]].receiveTime {
					findResults.minTS = lp.events[lpIndex[lp.lpId]].receiveTime
					findResults.definingLP = lp.lpId
					findResults.found = true
				}
			}
		}
			return findResults
	}

	// given a timestamp (scheduleTime), fine the number of events that are available for execution (that is,
	// count the number of LPs such that their head event has a receiveTime at or above scheduleTime and a
	// sendTime at or below scheduleTime (this second check should be strictly below, but some of our input
	// data sets have events with the same send/receive time so we have to weaken this constraint..
	findEventsAvailable := func(lps []lpData, scheduleTime float64, definingLP int) int {
		eventsAvailable := 0
		for i, lp := range lps {
			if lpIndex[i] < len(lp.events) && (lp.events[lpIndex[i]].sendTime < scheduleTime || definingLP == lp.lpId) {
				eventsAvailable++
				lpIndex[i]++
			}
		}
		return eventsAvailable
	}

	outFile, err = os.Create("analysisData/eventsAvailableBySimCycle.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# events available by simulation cycle\n")
	fmt.Fprintf(outFile,"# sim cycle, num of events\n")
	for i := range lps {lpIndex[i] = 0}
	simCycle := 0
	maxEventsAvailable := 0
	eventsExhausted := false
	timesXEventsAvailable := make([]int, numOfEvents)
	for ; eventsExhausted != true ; {
		eventsExhausted = true
		findResults := findLowestTS(lps)
		if findResults.found == true {
			eventsAvailable := findEventsAvailable(lps, findResults.minTS, findResults.definingLP)
			fmt.Fprintf(outFile,"%v %v\n",simCycle + 1,eventsAvailable)
			timesXEventsAvailable[eventsAvailable]++
			if maxEventsAvailable < eventsAvailable {maxEventsAvailable = eventsAvailable}
			simCycle++
			eventsExhausted = false
		}
	}
	err = outFile.Close()
	if err != nil {panic(err)}
				
	// write out summary of events available
	outFile, err = os.Create("analysisData/timesXeventsAvailable.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# times X events are available for execution\n")
	fmt.Fprintf(outFile,"# X, num of occurrences\n")
	for i := 0; i < maxEventsAvailable; i++ {fmt.Fprintf(outFile,"%v %v\n",i+1,timesXEventsAvailable[i+1])}
	err = outFile.Close()
	if err != nil {panic(err)}


	fmt.Printf("%v: Finished.\n", printTime())
	return
}	
