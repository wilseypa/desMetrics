package main

import "os"
import "fmt"
import "sort"
import "math"
import "strconv"
import "runtime"

// setup a data structure for events.  internally we're going to store LP names with their integer map value.
// since we're storing events into an array indexed by the LP in question (sender or receiver), we will only
// store the other "companion" LP internally
type eventData struct {
	companionLP int
	sendTime float64
	receiveTime float64
}

// i'm sure there is a way to combine these sorts with a common setup, but i don't know go well enough to
// achieve it; we'll use this approach for now

// functions to support sorting of the events by their receive time
type byReceiveTime []eventData
func (a byReceiveTime) Len() int           {return len(a)}
func (a byReceiveTime) Swap(i, j int)      {a[i], a[j] = a[j], a[i]}
func (a byReceiveTime) Less(i, j int) bool {return a[i].receiveTime < a[j].receiveTime}

func main() {

	// enable the use of all CPUs on the system
	numThreads := runtime.NumCPU()
	runtime.GOMAXPROCS(numThreads)

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

//	mapLPNameToInt := make(map[string]lpMap)
	lpNameMap :=make(map[string]*lpMap)

	// ultimately we will use these to hold event data
	var lps [][]eventData
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
		if lpIndex[rLPint] > cap(lps[rLPint]) {panic("Something wrong, we should have computed the appropriate size on the first parse.\n")}
		lps[rLPint][lpIndex[rLPint]].companionLP = lpNameMap[sLP].toInt
		lps[rLPint][lpIndex[rLPint]].receiveTime = rTS
		lps[rLPint][lpIndex[rLPint]].sendTime = sTS
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
	fmt.Printf("Processing %v to capture event and LP counts.\n", os.Args[1])
	inputFile, err := os.Open(os.Args[1])
	if err != nil { panic(err) }
	parseJsonFile(inputFile)
	err = inputFile.Close()

	// reset this function so we don't print the model information in the second parse
	printInfo = func (format string, a ...interface{}) (n int, err error) {return 0, nil}

	// lps is an array of the LPs; each LP entry will hold the events it received
	lps = make([][]eventData, len(lpNameMap))

	// periodically we will need to walk through the LP arrays independently; we will use this array to do so
	lpIndex = make([]int, len(lps))
	// allocate entries in each LP to hold the number events it received
	for _, i := range lpNameMap {
		lpIndex[i.toInt] = -1
		lps[i.toInt] = make([]eventData, i.receivedEvents)
	}

	// this time we will save the events
	fmt.Printf("Processing %v to capture events.\n", os.Args[1])
	processEvent = addEvent
	inputFile, err = os.Open(os.Args[1])
	if err != nil { panic(err) }
	parseJsonFile(inputFile)
	err = inputFile.Close()

	// build the reverse map: integers -> LP Names
	mapIntToLPName := make([]string, numOfLPs)
	for key, value := range lpNameMap {mapIntToLPName[value.toInt] = key}

	// let's check to see if all LPs received an event (not necessarily a huge problem, but something we
	// should probably be aware of. 
	fmt.Printf("Verifying that all LPs recieved at least one event.\n")
	for i := range lps {
		if len(lps[i]) == 0 {
			fmt.Printf("WARNING: LP %v recived zero messages.\n", mapIntToLPName[i])
		}
	}

	// we now need to sort the event lists by receive time.  for this we'll use the sort package.
	fmt.Printf("Sorting the events in each LP by receive time.\n")
	for i := range lps {sort.Sort(byReceiveTime(lps[i]))}

	// on to analysis: we first count local and remote events received by each LP.
	fmt.Printf("ANALYSIS of events by receiving LP.\n")	

	// create the directory to store the resulting data files (if it is not already there)
	err =  os.MkdirAll ("analysisData", 0777)
	if err != nil {panic(err)}

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

	// this is the channel type to communication event summary results
	type lpEventSummary struct {
		lpInt int
		local int
		remote int
		total int
		cover [5]int
	}

	// this helper function (goroutine) will compute the number of local and remote events received for a
	// subset of the LP (defined as those from lps[low] to lps[high]; as well as computing the number of
	// sending events to cover the target percentages (75, 80, 90, 95, and 100%)
	countLocalRemoteEvents := func(low int, high int, c chan<- lpEventSummary)  {
		var es lpEventSummary
		lpData := make([]int,len(lps))
		for i := low; i <= high; i++ {

			es.lpInt = i
			es.local = 0
			es.remote = 0
			for j := range lps {lpData[j] = 0}
			for j := range lps[i] {
				lpData[lps[i][j].companionLP]++
				if i != lps[i][j].companionLP {
					es.remote++
				} else {
					es.local++
				}	
			}
			sort.Sort(sort.Reverse(sort.IntSlice(lpData)))
			es.total = es.remote + es.local
			es.cover[0] = numOfLPsToCover(es.total, lpData, 75)  //  75% cutoff
			es.cover[1] = numOfLPsToCover(es.total, lpData, 80)  //  80% cutoff
			es.cover[2] = numOfLPsToCover(es.total, lpData, 90)  //  90% cutoff
			es.cover[3] = numOfLPsToCover(es.total, lpData, 95)  //  95% cutoff
			es.cover[4] = numOfLPsToCover(es.total, lpData, 100) // 100% cutoff
			c <- es
		}
		return
	}


	// location to write summaries of local and remote events received
	eventSummaries, err := os.Create("analysisData/eventsExecutedByLP.dat")
	if err != nil {panic(err)}
	fmt.Printf("Computing: the number local/remote events executed by each LP.\n")
	fmt.Fprintf(eventSummaries, "# summary of local and remote events executed\n")
	fmt.Fprintf(eventSummaries, "# LP, local, remote\n")

	// location to write percentage of LPs to cover percentage of events received
	numToCover, err := os.Create("analysisData/numOfLPsToCoverPercentTotalMessages.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(numToCover,"# number of destination LPs (sorted by largest messages sent to) to cover percentage of total events\n")
	fmt.Fprintf(numToCover,"# LP name, total events sent, num of LPs to cover: 75, 80, 90, 95, and 100 percent of the total events sent.\n")

	c := make(chan lpEventSummary)
	goroutineSliceSize := len(lps)/numThreads
	for i := 0; i < numThreads; i++ {
		low := i * goroutineSliceSize
		high := low + goroutineSliceSize - 1
		if i == numThreads - 1 {high = len(lps)-1}
		go countLocalRemoteEvents(low, high, c)
	}
	
	// process all of the data in the channel
	for _ = range lps {
		info := <- c
		fmt.Fprintf(eventSummaries,"%v, %v, %v\n", mapIntToLPName[info.lpInt], info.local, info.remote)
		fmt.Fprintf(numToCover,"%v, %v, %v, %v, %v, %v, %v\n", mapIntToLPName[info.lpInt], info.total,
			info.cover[0], info.cover[1], info.cover[2], info.cover[3], info.cover[4])
	}
	close(c)

	err = eventSummaries.Close()
	if err != nil {panic(err)}
	err = numToCover.Close()
	if err != nil {panic(err)}

	// events available for execution: here we will assume all events execute in unit time and evaluate
	// the events as executable by simulation cycle.  basically we will advance the simulation time to the
	// lowest receive time of events in all of the LPs and count the number of LPs that could potentially
	// be executed at that time.  the general algorithm is outlined in the indexTemplate.md file for this
	// project.

	// we will use lpIndex to point to the current event not yet processed in a simulation cycle at each
	// LP

	fmt.Printf("Computing: event parallelism statistics.\n")
	for i := range lps {lpIndex[i] = 0}
	simCycle := 0
	eventsAvailable := make([]int, numOfEvents)
	maxEventsAvailable := 0
	done := false
	lpIndexToFixSameSendReceiveTime := 0
	for ; done != true ; {
		scheduleTime := math.MaxFloat64
		done = true
		// pickup the minimum receive time not yet processed
		for i, lp := range lps {
			if lpIndex[i] < len(lp) &&  lp[lpIndex[i]].receiveTime < scheduleTime {
				scheduleTime = lp[lpIndex[i]].receiveTime
				done = false
				// because the ross data has events with the same send and receive times, we
				// have to have an exception for that case of this algorithm doesn't
				// terminate.  here we will record the LP index defining the schedule time and
				// use that as an exception for the next part.....what a drag
				lpIndexToFixSameSendReceiveTime =  i
			}
		}
		// walk through and count the number of events available
		if done == false {
			for i, lp := range lps {
				if lpIndex[i] < len(lp) &&  (lp[lpIndex[i]].sendTime < scheduleTime || 
					// this is our special case
					i ==  lpIndexToFixSameSendReceiveTime) {
					eventsAvailable[simCycle]++
					lpIndex[i]++
				}
			}
			if maxEventsAvailable < eventsAvailable[simCycle] {maxEventsAvailable = eventsAvailable[simCycle]}
			simCycle++
		}
	}

	// while printing results let's also capture the number of simulation cycles with X events available
	timesXeventsAvailable := make([]int,maxEventsAvailable)
	outFile, err := os.Create("analysisData/eventsAvailableBySimCycle.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# events available by simulation cycle\n")
	fmt.Fprintf(outFile,"# sim cycle, num of events\n")
	for i := 0; i < simCycle; i++ {
		fmt.Fprintf(outFile,"%v %v\n",i+1,eventsAvailable[i])
		timesXeventsAvailable[eventsAvailable[i] - 1]++
	}
	err = outFile.Close()
	if err != nil {panic(err)}

	// write out summary of events available
	outFile, err = os.Create("analysisData/timesXeventsAvailable.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# times X events are available for execution\n")
	fmt.Fprintf(outFile,"# X, num of occurrences\n")
	for i := range timesXeventsAvailable {fmt.Fprintf(outFile,"%v %v\n",i+1,timesXeventsAvailable[i])}
	err = outFile.Close()
	if err != nil {panic(err)}

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

	// compute the local and global event chains for each LP
	fmt.Printf("Computing: local, linked, and global event chains by LP.\n")
	eventChainLength := 5
	localEventChain := make([][]int,len(lps))
	for i := range localEventChain {localEventChain[i] = make([]int,eventChainLength)}
	linkedEventChain := make([][]int,len(lps))
	for i := range linkedEventChain {linkedEventChain[i] = make([]int,eventChainLength)}
	globalEventChain := make([][]int,len(lps))
	for i := range globalEventChain {globalEventChain[i] = make([]int,eventChainLength)}
	for i, lp := range lps {
		// local chain analysis
		j := 0
		for ; j < len(lp) ; {
			for ; j < len(lp) && lp[j].companionLP != i ; j++ {}
			if j < len(lp) && lp[j].companionLP == i {
				k := j + 1
				for ; k < len(lp) && lp[k].sendTime <= lp[j].receiveTime && lp[k].companionLP == i ; {k++}
				if k-j-1 >= eventChainLength {
					localEventChain[i][eventChainLength - 1]++
				} else {
					localEventChain[i][k-j-1]++
				}
				j = j + k
			}
		}
		// lined chain analysis
		j = 0
		for ; j < len(lp) ; {
			for ; j < len(lp) && lp[j].companionLP != i ; j++ {}
			if j < len(lp) && lp[j].companionLP == i {
				k := j + 1
				for ; k < len(lp) && lp[k].sendTime <= lp[k-1].receiveTime && lp[k].companionLP == i ; {k++}
				if k-j-1 >= eventChainLength {
					linkedEventChain[i][eventChainLength - 1]++
				} else {
					linkedEventChain[i][k-j-1]++
				}
				j = j + k
			}
		}
		// global chain analysis
		j = 0
		for ; j < len(lp) ; {
			k := j + 1
			for ; k < len(lp) && lp[k].sendTime <= lp[j].receiveTime ; {
				k++
			}
			if k-j-1 >= eventChainLength {
				globalEventChain[i][eventChainLength - 1]++
			} else {
				globalEventChain[i][k-j-1]++
			}
			j = j + k
		}
	}
	
	// write local event chains for each LP
	outFile, err = os.Create("analysisData/localEventChainsByLP.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# local event chains by LP\n")
	fmt.Fprintf(outFile,"# LP, local chains of length: 1, 2, 3, 4, >= 5\n")
	for i := range lps {
		fmt.Fprintf(outFile,"%v",mapIntToLPName[i])
		for j := 0; j < eventChainLength; j++ {fmt.Fprintf(outFile,", %v",localEventChain[i][j])}
		fmt.Fprintf(outFile,"\n")
	}
	err = outFile.Close()
	if err != nil {panic(err)}

	// write linked event chains for each LP
	outFile, err = os.Create("analysisData/linkedEventChainsByLP.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# linked event chains by LP\n")
	fmt.Fprintf(outFile,"# LP, linked chains of length: 1, 2, 3, 4, >= 5\n")
	for i := range lps {
		fmt.Fprintf(outFile,"%v",mapIntToLPName[i])
		for j := 0; j < eventChainLength; j++ {fmt.Fprintf(outFile,", %v",linkedEventChain[i][j])}
		fmt.Fprintf(outFile,"\n")
	}
	err = outFile.Close()
	if err != nil {panic(err)}

	// write global event chains for each LP
	outFile, err = os.Create("analysisData/globalEventChainsByLP.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# global event chains by LP\n")
	fmt.Fprintf(outFile,"# LP, global chains of length: 1, 2, 3, 4, >= 5\n")
	for i := range lps {
		fmt.Fprintf(outFile,"%v",mapIntToLPName[i])
		for j := 0; j < eventChainLength; j++ {fmt.Fprintf(outFile,", %v",globalEventChain[i][j])}
		fmt.Fprintf(outFile,"\n")
	}
	err = outFile.Close()
	if err != nil {panic(err)}

// number of LPs with n event chains of length X number of LPs with average event chains of length X

	// not sure this will be useful or not, but let's save totals of the local and global event chains.
	// specifically we will sum the local/global event chains for all of the LPs in the system
	eventChainSummary := make([][]int,3)
	// store local event chain summary data here
	eventChainSummary[0] = make([]int,eventChainLength)
	// store linked event chain summary data here
	eventChainSummary[1] = make([]int,eventChainLength)
	// store global event chain summary data here
	eventChainSummary[2] = make([]int,eventChainLength)
	for i := range lps {
		for j := 0; j < eventChainLength; j++ {
			eventChainSummary[0][j] = eventChainSummary[0][j] + localEventChain[i][j]
			eventChainSummary[1][j] = eventChainSummary[1][j] + linkedEventChain[i][j]
			eventChainSummary[2][j] = eventChainSummary[2][j] + globalEventChain[i][j]
		}
	}
	outFile, err = os.Create("analysisData/eventChainsSummary.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# number of event chains of length X\n")
	fmt.Fprintf(outFile,"# chain length, num of local chains, num of linked chains, num of global chains\n")
	for i := 0; i < eventChainLength; i++ {fmt.Fprintf(outFile,"%v, %v, %v, %v\n",i+1,eventChainSummary[0][i],eventChainSummary[1][i],eventChainSummary[2][i])}
	err = outFile.Close()
	if err != nil {panic(err)}

	return
}	
