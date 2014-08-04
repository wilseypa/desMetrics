package main

import "os"
import "fmt"
import "log"
import "sort"
import "math"
import "strconv"

var desTraceData struct {
	SimulatorName string `json:"simulator_name"`
	ModelName string `json:"model_name"`
	CaptureDate string `json:"capture_date"`
	CommandLineArgs string `json:"command_line_arguments"`
	Events []struct {
		SendLP string  `json:"sLP"`
		SendTime float64  `json:"sTS"`
		ReceiveLP string  `json:"rLP"`
		ReceiveTime float64  `json:"rTS"`
	} `json:"events"`
}

// now we need to setup a data structure for events.  internally we're going to store LP
// names with their integer map value.  since we're storing events into an array indexed
// by the LP in question (sender or receiver), we will only store the other "companion" LP
// internally
type eventData struct {
	companionLP int
	sendTime float64
	receiveTime float64
}

// i'm sure there is a way to combine these sorts with a common setup, but i don't know go
// well enough to achieve it; we'll use this approach for now

// functions to support sorting of the events by their receive time
type byReceiveTime []eventData
func (a byReceiveTime) Len() int           {return len(a)}
func (a byReceiveTime) Swap(i, j int)      {a[i], a[j] = a[j], a[i]}
func (a byReceiveTime) Less(i, j int) bool {return a[i].receiveTime < a[j].receiveTime}

// functions to support sorting of the events by their send time
type bySendTime []eventData
func (a bySendTime) Len() int           {return len(a)}
func (a bySendTime) Swap(i, j int)      {a[i], a[j] = a[j], a[i]}
func (a bySendTime) Less(i, j int) bool {return a[i].sendTime < a[j].sendTime}

func main() {

	inputFile, err := os.Open(os.Args[1])
	if err != nil { panic(err) }

	fileInfo, err := os.Stat(os.Args[1])
	if err != nil { panic(err) }
	fmt.Printf("Length of file %v\n",fileInfo.Size())


	// now we need to partition the events by LP for our analysis.  lps is an slice
	// (of slices) that we will use to store the events associated with each LP.  we
	// have to define initial size and capacity limits to the go slices.  as a
	// starting point, we will assume that the events are approximately equally
	// distributed among the LPs.  if this is not large enough, we will grow the slice
	// in increments of 2048 additional elements.

	// since we have no real idea of how many LPs or even the total events processed,
	// we will have to begin with wild guesses.  i suppose we could make this a
	// configurable parameter on the argument list, but for now i'm just going to punt
	// and begin with 4k LPs and with 10,000 events/LP.


	// we'll initialize then entries for each element in lps as we discover new LPs in
	// the input file (and grow the size of lps as necessary as well)
	lps := make([][]eventData, 4096)
	// we will use lpIndex throughout the program to keep a pointer for each LP to the
	// event of interest for that LP
	lpIndex := make([]int, len(lps))
	for i := range lps {lpIndex[i] = -1}
	numOfLPs := 0
	numOfEvents := 0
	mapLPNameToInt := make(map[string]int)

	addLP := func(lp string) int {
		index, present := mapLPNameToInt[lp]
		if !present {
			mapLPNameToInt[lp] = numOfLPs
			lps[numOfLPs] = make([]eventData, 20000)
			index = numOfLPs
			numOfLPs++
		}
		if numOfLPs > cap(lps) {
			sizeToGrow := 1024
			fmt.Printf("Enlarging slices to hold LPs.  Prev size: %v, New size: %v\n",
				cap(lps),cap(lps)+sizeToGrow)
			newLpIndex := make([]int, cap(lps)+sizeToGrow)
			for i := range newLpIndex {newLpIndex[i] = -1}
			copy(newLpIndex, lpIndex)
			lpIndex = newLpIndex
			newLps := make([][]eventData, cap(lps)+sizeToGrow)
			copy(newLps, lps)
			lps = newLps
		}
		return index
	}

	addEvent := func(sLP string, sTS float64, rLP string, rTS float64) {
		sLPindex := addLP(sLP)
		rLPindex := addLP(rLP)
		lpIndex[rLPindex]++
		if lpIndex[rLPindex] > cap(lps[rLPindex]) {
			fmt.Printf("Enlarging slice for receiving LP: %v.  Prev size: %v, New size: %v\n",
				rLP,cap(lps[rLPindex]),cap(lps[rLPindex])+2048)
			newSlice := make([]eventData,cap(lps[rLPindex])+2048)
			copy(newSlice,lps[rLPindex])
			lps[rLPindex] = newSlice
		}
		lps[rLPindex][lpIndex[rLPindex]].companionLP = sLPindex
		lps[rLPindex][lpIndex[rLPindex]].receiveTime = rTS
		lps[rLPindex][lpIndex[rLPindex]].sendTime = sTS
		numOfEvents++
	}

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

	fmt.Printf("Processing JSON file: %v\n", os.Args[1])
		
	// parse the json data file, this is hugely fragile but functional
	token, tokenText = Scan()
	scanAssume(L_CURLY)
	parsingLoop: for ; token != EOF; {
		switch token {
		case SIMULATOR_NAME:
			token, tokenText = Scan()
			scanAssume(COLON)
			fmt.Printf("    Simulator Name: %v\n", string(tokenText))
			token, tokenText = Scan()
			scanAssume(COMMA)
		case MODEL_NAME:
			token, tokenText = Scan()
			scanAssume(COLON)
			fmt.Printf("   Model Name: %v\n", string(tokenText))
			token, tokenText = Scan()
			scanAssume(COMMA)
		case CAPTURE_DATE:
			token, tokenText = Scan()
			scanAssume(COLON)
			fmt.Printf("    Capture date: %v\n", string(tokenText))
			token, tokenText = Scan()
			scanAssume(COMMA)
		case COMMAND_LINE_ARGS :
			token, tokenText = Scan()
			scanAssume(COLON)
			fmt.Printf("    Command line arguments: %v\n", string(tokenText))
			token, tokenText = Scan()
			scanAssume(COMMA)
		case EVENTS:
			token, tokenText = Scan()
			scanAssume(COLON)
			scanAssume(L_BRACKET)
			fmt.Printf("Processing Events.\n")
			for ; token != EOF ; {
				var sendingLP, receivingLP string
				var sendTime, receiveTime float64
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
				// we should require that the send time be strictly less than the receive
				// time, but the ross (airport model) data actually has data with the
				// send/receive times (and other sequential simulators are likely to have
				// this as well), so we will weaken this constraint.
				if sendTime > receiveTime {log.Panic("Event has send time greater than receive time: %v %v %v %v\n", 
					sendingLP, sendTime, receivingLP, receiveTime)}
				addEvent(sendingLP, sendTime, receivingLP, receiveTime)
				if token == R_BRACKET {break parsingLoop}
				scanAssume(COMMA)
			}
		default:
			// this should never happen
			fmt.Printf("Mal formed json file at %v\n",fileLineNo)
			panic("Aborting")
		}
	}

	// build the reverse map: integers -> LP Names
	mapIntToLPName := make([]string, numOfLPs)
	for key, value := range mapLPNameToInt {mapIntToLPName[value] = key}

	// now we need to set the lengths of the slices in the LP data so we can use the
	// go builtin len() function on them
	fmt.Printf("Sizing the slices of the LP events.\n")
	for i := 0; i < numOfLPs; i++ {
		if lpIndex[i] != -1 {
			lps[i] = lps[i][:lpIndex[i]+1]
		} else {
			lps[i] = lps[i][:0]
			fmt.Printf("WARNING: LP \"%v\" recived zero messages.\n", mapIntToLPName[i])
		}
	}
	lps = lps[:numOfLPs]

	// we now need to sort the event lists by receive time.  for this we'll use the sort package.
	fmt.Printf("Sorting the events in each LP by receive time.\n")
	for i := range lps {sort.Sort(byReceiveTime(lps[i]))}

	// on to analysis: we first consider local and remote events.  for this purpose,
	// we'll compute a (square) matrix of ints where the row index is the receiving LP
	// and the column index is the sending LP and the entries are the number of events
	// exchanged between them.  from this matrix, we will print out summary files.
	lpMatrix := make([][]int, len(lps))
	for i := range lps {
		lpMatrix[i] = make([]int,len(lps))
		for j := range lps[i] {
			lpMatrix[i][lps[i][j].companionLP]++
		}
	}

	// create the directory to store results if it is not already there
	err =  os.MkdirAll ("analysisData", 0777)
	if err != nil {panic(err)}

	fmt.Printf("Computing the number local/remote events executed by each LP.\n")
	// write summaries of local and remote events received
	outFile, err := os.Create("analysisData/eventsExecutedByLP.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile, "# summary of local and remote events executed\n")
	fmt.Fprintf(outFile, "# LP, local, remote\n")
	for i := range lps {
		fmt.Fprintf(outFile,"%v, ",mapIntToLPName[i])
		rCount := 0
		for j := range lps {if i != j {rCount = rCount + lpMatrix[i][j]}}
		fmt.Fprintf(outFile,"%v, %v\n",lpMatrix[i][i],rCount)
	}
	err = outFile.Close()
	if err != nil {panic(err)}

	// let's take a look at the frequency of communication with others.  how many
	// different LPs does each LP send messages to?
	fmt.Printf("Computing the number of LPs sent to to cover a percentage of total events sent.\n")
	// first we need to transpose the matrix so we have senders in rows and receivers
	// in columns
	for i := range lps {
		for j := i; j < len(lps); j++ {
			tmp := lpMatrix[i][j]
			lpMatrix[i][j] = lpMatrix[j][i]
			lpMatrix[j][i] = tmp
		}
	}
	// now we define a helper function to compute the number of LPs that an LP sends
	// messages to to cover the number required
	numOfLPsToCover := func(lpMatrix []int, numRequired int) int {
		lpCount := 0
		eventCount := 0
		for i := range lpMatrix {
			lpCount++
			eventCount = eventCount + lpMatrix[i]
			if eventCount >= numRequired {return lpCount}
		}
		panic("ERROR: something's amiss in this computation\n")
	}
	// next we have to sort the events sent by each sending LP (this means we no
	// longer know the receiving LP); count the number of events sent by each LP; and
	// dump the results
	outFile, err = os.Create("analysisData/numOfLPsToCoverPercentTotalMessages.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile,"# number of destination LPs (sorted by largest messages sent to) to cover percentage of total events\n")
	fmt.Fprintf(outFile,"# LP name, total events sent, num of LPs to cover: 75, 80, 90, 95, and 100 percent of the total events sent.\n")
	for i := range lps {
		sort.Sort(sort.Reverse(sort.IntSlice(lpMatrix[i])))
		totalEventsSent := 0 
		j := 0
		for ; j < len(lps) && lpMatrix[i][j] != 0; j++ {
			totalEventsSent = totalEventsSent + lpMatrix[i][j]
		}
		if totalEventsSent != 0 {
			fmt.Fprintf(outFile,"%v, %v, %v, %v, %v, %v, %v\n",mapIntToLPName[i],totalEventsSent,
				numOfLPsToCover(lpMatrix[i], (totalEventsSent * 75) / 100), // 75% cutoff
				numOfLPsToCover(lpMatrix[i], (totalEventsSent * 80) / 100), // 80% cutoff
				numOfLPsToCover(lpMatrix[i], (totalEventsSent * 90) / 100), // 90% cutoff
				numOfLPsToCover(lpMatrix[i], (totalEventsSent * 95) / 100), // 95% cutoff
				j+1) // 100% cutoff
		} else {
			fmt.Fprintf(outFile,"%v, %v, %v, %v, %v, %v, %v\n",mapIntToLPName[i],totalEventsSent,0,0,0,0,0)
		}
	}
	err = outFile.Close()
	if err != nil {panic(err)}

	// events available for execution: here we will assume all events execute in unit
	// time and evaluate the events as executable by simulation cycle.  basically we
	// will advance the simulation time to the lowest receive time of events in all
	// of the LPs and count the number of LPs that could potentially be executed at
	// that time.  the general algorithm is outlined in the indexTemplate.md file for
	// this project. 

	// we will use lpIndex to point to the current event not yet processed in a
	// simulation cycle at each LP

	fmt.Printf("Computing event parallelism statistics.\n")
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
				// because the ross data has events with the same send and
				// receive times, we have to have an exception for that
				// case of this algorithm doesn't terminate.  here we will
				// record the LP index defining the schedule time and use
				// that as an exception for the next part.....what a drag
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

	// while printing results let's also capture the number of simulation cycles with
	// X events available
	timesXeventsAvailable := make([]int,maxEventsAvailable)
	outFile, err = os.Create("analysisData/eventsAvailableBySimCycle.dat")
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

	// event chains are collections of events for execution at the LP that could
	// potentially be executed together.  thus when examining an event with receive
	// timestamp t, the chain is formed by all future (by receive time) events in that
	// LP that have a send time < t.  local chains have the additional constraint that
	// they must also have been sent by the executing LP (self generated events).
	// global chains can be sent from any LP.

	// a third form of event chains is the "linked event chain".  the linked chain is
	// similar to the local chain except that the constraint on the send time is
	// relaxed so that any event sent within the time window of the local event chain
	// (so basically any event generated by the chain is also potentially a member of
	// the chain).


	// consider a locally linked chain computation.  that is, anything generated
	// within the time frame of the chain should also be included in the chain length
	// computation

	// compute the local and global event chains for each LP
	fmt.Printf("Computing local, linked, and global event chains by LP.\n")
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

// number of LPs with n event chains of length X
// number of LPs with average event chains of length X

	// not sure this will be useful or not, but let's save totals of the local and
	// global event chains.  specifically we will sum the local/global event chains
	// for all of the LPs in the system
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

/*
	// in this step we will be looking at events seen at the sending LPsd.  the first
	// step it to store the event data into the lps by the sending LP id.

	// reset the lpIndex pointers and the length of slices to their capacity
	for i := range lps {lpIndex[i] = -1}
	for i := range lps {lps[i] = lps[i][:cap(lps[i])]}

	fmt.Printf("Organizing data by sending LPs.\n")
	for _, traceEvent := range desTraceData.Events {
		sLP := mapLPNameToInt[traceEvent.SendLP]

		lpIndex[sLP]++
		if lpIndex[sLP] >= cap(lps[sLP]) {
			fmt.Printf("Enlarging slice for sending LP: %v.  Prev size: %v, New size: %v\n",
				traceEvent.ReceiveLP,cap(lps[sLP]),cap(lps[sLP])+2048)
			newSlice := make([]eventData,cap(lps[sLP])+2048)
			for i := range lps[sLP] {newSlice[i] = lps[sLP][i]}
			lps[sLP] = newSlice
		}
		lps[sLP][lpIndex[sLP]].companionLP = mapLPNameToInt[traceEvent.ReceiveLP]
		lps[sLP][lpIndex[sLP]].receiveTime = traceEvent.ReceiveTime
		lps[sLP][lpIndex[sLP]].sendTime = traceEvent.SendTime
	}

	// now we need to set the lengths of the slices in the LP data so we can use the
	// go builtin len() function on them
	fmt.Printf("Sizing the slices of the LP events.\n")
	for i := range lps {
		if lpIndex[i] != -1 {
			lps[i] = lps[i][:lpIndex[i]+1]
		} else {
			lps[i] = lps[i][:0]
			fmt.Printf("WARNING: LP \"%v\" sent zero messages.\n", mapIntToLPName[i])
		}
		//fmt.Printf("%v\n",lps[i])
	}
	// we now need to sort the event lists by receive time.  for this we'll use the sort package.
	fmt.Printf("Sorting the events in each LP by send time.\n")
	for i := range lps {sort.Sort(bySendTime(lps[i]))}
*/
	return
}	
