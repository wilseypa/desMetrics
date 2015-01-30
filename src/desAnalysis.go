
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
import "flag"

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

	var commSwitchOff bool
	flag.BoolVar(&commSwitchOff, "no-comm-matrix", false, "turn off generation of the file eventsExchanged.csv")
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
		printTime(), flag.Args())
	inputFile, err := os.Open(flag.Arg(0))
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
	fmt.Printf("%v: Processing %v to capture events.\n", printTime(), flag.Arg(0))
	processEvent = addEvent
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
        fmt.Fprintf(outFile,"  \"total_events\" : %v,\n}",numOfEvents)

	err = outFile.Close()
	if err != nil {panic(err)}

	fmt.Printf("%v: Analysis (parallel) of events organized by receiving LP.\n", printTime())	

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
		lpCount := 0
		eventCount := 0
		for i := range data {
			lpCount++
			eventCount = eventCount + data[i]
			if eventCount >= numRequired {
				return lpCount
			}
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
		// if it's send time is less than or equal to the receive time of the event at the head of the chain.
		i := 0
		for ; i < len(lp.events) ; {
			for ; i < len(lp.events) && lp.events[i].companionLP != lp.lpId ; i++ {}
			if i < len(lp.events) && lp.events[i].companionLP == lp.lpId {
				j := i + 1
				for ; j < len(lp.events) && lp.events[j].companionLP == lp.lpId && lp.events[j].sendTime <= lp.events[i].receiveTime ; {j++}
				accumulateChain(local, chainLength, j-i-1)
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
				accumulateChain(linked, chainLength, j-i-1)
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
			accumulateChain(global, chainLength, j-i-1)
			i = i + j
		}
		return local, linked, global
	}

	// PAW: change the comment headers of chains/covers to for loops so the numbers will automatically grow
	// with the variable setting.... 
	localChainFile, err := os.Create("analysisData/localEventChainsByLP.csv")
	if err != nil {panic(err)}
	fmt.Fprintf(localChainFile,"# local event chains by LP\n")
	fmt.Fprintf(localChainFile,"# LP, local chains of length: 1, 2, 3, 4, >= 5\n")

	linkedChainFile, err := os.Create("analysisData/linkedEventChainsByLP.csv")
	if err != nil {panic(err)}
	fmt.Fprintf(linkedChainFile,"# linked event chains by LP\n")
	fmt.Fprintf(linkedChainFile,"# LP, linked chains of length: 1, 2, 3, 4, >= 5\n")

	globalChainFile, err := os.Create("analysisData/globalEventChainsByLP.csv")
	if err != nil {panic(err)}
	fmt.Fprintf(globalChainFile,"# global event chains by LP\n")
	fmt.Fprintf(globalChainFile,"# LP, global chains of length: 1, 2, 3, 4, >= 5\n")

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

	// now we will compute and print a summary matrix of the number of events exchanged between two LPs.
	// this matrix can be quite large and can actually be too large to create.  in order to save space, we
	// will print only the non-zero entries.  in the cases where it is too large to print, the command line
	// argument "--no-comm-matrix" can be used to suppress it's creation.

	if commSwitchOff == false {
		outFile, err = os.Create("analysisData/eventsExchanged.csv")
		if err != nil {panic(err)}

		fmt.Fprintf(outFile,"# event exchanged matrix data\n")
		fmt.Fprintf(outFile,"# receiving LP, sending LP, number of events sent\n")
		
		lpEventSendCount := make([]int,numOfLPs)
		for j, lp := range(lps) {
			for i := range lpEventSendCount {lpEventSendCount[i] = 0}
			for _, event := range lp.events {lpEventSendCount[event.companionLP]++}
			for i := range lpEventSendCount {
				if lpEventSendCount[i] != 0 {
					fmt.Fprintf(outFile,"%v,%v,%v\n", j, i, lpEventSendCount[i])
				}
			}
		}
		err = outFile.Close()
		if err != nil {panic(err)}
	}

	// events available for execution: here we will assume all events execute in unit time and evaluate the
	// events as executable by simulation cycle.  basically we will advance the simulation time to the lowest
	// receive time of events in all of the LPs and count the number of LPs that could potentially be executed
	// at that time.  the general algorithm is outlined in the indexTemplate.md file for this project.

	fmt.Printf("%v: Analysis (parallel) of event parallelism available by simulation cycle (potential parallelism).\n", printTime())	

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

	fmt.Printf("%v: Finished.\n", printTime())
	return
}	
