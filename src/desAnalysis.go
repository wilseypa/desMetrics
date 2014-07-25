package main

import "fmt"
import "os"
import "log"
import "sort"
import "math"
import "encoding/json"

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

// assumes one argument that is the name of the trace file to use
func main() {
	
	// get a handle to the input file and import the json file
	// get a handle to the input file and import the json file
	traceDataFile, err := os.Open(os.Args[1])
	if err != nil { panic(err) }
	fmt.Printf("Parsing input json file: %v\n",os.Args[1])
	jsonParser := json.NewDecoder(traceDataFile)
	err = jsonParser.Decode(&desTraceData); 
	if err != nil { panic(err) }
	fmt.Printf("Json file parsed successfully.  Summary info:\n    Simulator Name: %s\n    Model Name: %s\n    Capture Date: %s\n    Command line used for capture: %s\n",
		desTraceData.SimulatorName, 
		desTraceData.ModelName, 
		desTraceData.CaptureDate, 
		desTraceData.CommandLineArgs)

	// ok, so let's create a map of the LP names -> integers so we can setup
	// arrays/slices of LPs; while we're running through the event list, let's do what
	// we can to verify the integrity of the data.  at this point all we can do is
	// ensure that the send time is less than the receive time.  

	fmt.Printf("Building LP maps and validating data.\n")
	numOfLPs := 0
	mapLPNameToInt := make(map[string]int)
	for _,eventData := range desTraceData.Events {
		_,present := mapLPNameToInt[eventData.SendLP]
		if !present {
			mapLPNameToInt[eventData.SendLP] = numOfLPs
			numOfLPs = numOfLPs + 1
		}
		_,present = mapLPNameToInt[eventData.ReceiveLP]
		if !present {
			mapLPNameToInt[eventData.ReceiveLP] = numOfLPs
			numOfLPs = numOfLPs + 1
		}
		if eventData.SendTime > eventData.ReceiveTime {log.Panic("Event has send time greater than receive time: ", eventData)}
	}
	// build the reverse map: integers -> LP Names
	mapIntToLPName := make([]string, numOfLPs)
	for key, value := range mapLPNameToInt {mapIntToLPName[value] = key}

	// now we need to partition the events by LP for our first analysis.  lps is an
	// slice (of slices) that we will use to store the events associated with each LP.
	// we have to define initial size and capacity limits to the go slices.  while we
	// will include code to resize the slices, since they are large, let's try to
	// start out with capacities that are sufficient.  as a starting point, we will
	// assume that the events are approximately equally distributed among the LPs.
	// however, we will assume that the distribution of events may be off by as much
	// as 10%.  if this is not large enough, we will grow the slice in increments of
	// 2048 additional elements.
		
	capOfLPEventSlice := int(len(desTraceData.Events)/numOfLPs + int(.1*float64((len(desTraceData.Events)/(numOfLPs)))))
	if capOfLPEventSlice < 2048 {capOfLPEventSlice = 2048}
	lps := make([][]eventData, numOfLPs)
	for i := range lps {
		lps[i] = make([]eventData,capOfLPEventSlice)
	}

	// we will use lpIndex throughout the program to keep a pointer for each LP to the
	// event of interest for that LP
	lpIndex := make([]int, numOfLPs)
	for i := 0; i < numOfLPs; i++ {lpIndex[i] = -1}

	// in this step we will be looking at events seen at the receiving LP.  the first
	// step it to store the event data into the lps by the receiving LP id.

	fmt.Printf("Organizing data by receiving LPs.\n")
	for _, traceEvent := range desTraceData.Events {
		rLP := mapLPNameToInt[traceEvent.ReceiveLP]
		lpIndex[rLP]++
		if lpIndex[rLP] >= cap(lps[rLP]) {
			fmt.Printf("Enlarging slice for receiving LP: %v.  Prev size: %v, New size: %v\n",
				traceEvent.ReceiveLP,cap(lps[rLP]),cap(lps[rLP])+2048)
			newSlice := make([]eventData,cap(lps[rLP])+2048)
			for i := range lps[rLP] {newSlice[i] = lps[rLP][i]}
			lps[rLP] = newSlice
		}
		lps[rLP][lpIndex[rLP]].companionLP = mapLPNameToInt[traceEvent.SendLP]
		lps[rLP][lpIndex[rLP]].receiveTime = traceEvent.ReceiveTime
		lps[rLP][lpIndex[rLP]].sendTime = traceEvent.SendTime
	}
	// now we need to set the lengths of the slices in the LP data so we can use the
	// go builtin len() function on them
	fmt.Printf("Sizing the slices of the LP events.\n")
	for i := range lps {
		if lpIndex[i] != -1 {
			lps[i] = lps[i][:lpIndex[i]+1]
		} else {
			lps[i] = lps[i][:0]
			fmt.Printf("WARNING: LP \"%v\" recived zero messages.\n", mapIntToLPName[i])
		}
		//fmt.Printf("%v\n",lps[i])
	}

	// we now need to sort the event lists by receive time.  for this we'll use the sort package.
	fmt.Printf("Sorting the events in each LP by receive time.\n")
	for i := range lps {sort.Sort(byReceiveTime(lps[i]))}

	// on to analysis: we first consider local and remote events.  for this purpose,
	// we'll compute a matrix of ints where the row index is the receiving LP and the
	// column index is the sending LP and the entries are the number of events
	// exchanged between them.  from this matrix, we will print out summary files.

	fmt.Printf("Computing local/remote events executed by LP.\n")
	lpMatrix := make([][]int, numOfLPs)
	for i := range lpMatrix {
		lpMatrix[i] = make([]int,numOfLPs)
		for j := range lps[i] {
			lpMatrix[i][lps[i][j].companionLP]++
		}
	}

	// create the directory to store results if it is not already there
	err =  os.MkdirAll ("analysisData", 0777)
	if err != nil {panic(err)}

	// write summaries of local and remote events received
	outFile, err := os.Create("analysisData/eventsExecutedByLP.dat")
	if err != nil {panic(err)}
	fmt.Fprintf(outFile, "# summary of local and remote events executed\n")
	fmt.Fprintf(outFile, "# LP, local, remote\n")
	for i := 0; i < numOfLPs; i++ {
		fmt.Fprintf(outFile,"\"%v\", ",mapIntToLPName[i])
		rCount := 0
		for j := 0; j < numOfLPs; j++ {if i != j {rCount = rCount + lpMatrix[i][j]}}
		fmt.Fprintf(outFile,"%v, %v\n",lpMatrix[i][i],rCount)
	}
	err = outFile.Close()
	if err != nil {panic(err)}

	// events available for execution: here we will assume all events execute in unit
	// time and evaluate the events as executable by simulation cycle.  basically we
	// will advance the simulation time to the lowest receive time of events in all
	// of the LPs and count the number of LPs that could potentially be executed at
	// that time.  the general algorithm is outlined in the indexTemplate.md file for
	// this project. 

	// we will use lpIndex to point to the 

	fmt.Printf("Computing event parallelism statistics.\n")
	for i := range lps {lpIndex[i] = 0}
	simCycle := 0
	eventsAvailable := make([]int, len(desTraceData.Events))
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
	localEventChain := make([][]int,numOfLPs)
	for i := range localEventChain {localEventChain[i] = make([]int,eventChainLength)}
	linkedEventChain := make([][]int,numOfLPs)
	for i := range linkedEventChain {linkedEventChain[i] = make([]int,eventChainLength)}
	globalEventChain := make([][]int,numOfLPs)
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
		fmt.Fprintf(outFile,"\"%v\"",mapIntToLPName[i])
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
		fmt.Fprintf(outFile,"\"%v\"",mapIntToLPName[i])
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
		fmt.Fprintf(outFile,"\"%v\"",mapIntToLPName[i])
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


	// in this step we will be looking at events seen at the sending LPsd.  the first
	// step it to store the event data into the lps by the sending LP id.

	// reset the lpIndex pointers and the length of slices to their capacity
	for i := 0; i < numOfLPs; i++ {lpIndex[i] = -1}
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

	return
}
