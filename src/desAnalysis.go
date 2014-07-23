package main

import "fmt"
import "os"
import "log"
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

// assumes one argument that is the name of the trace file to use
func main() {
	
	// get a handle to the input file and import the json file
	// get a handle to the input file and import the json file
	traceDataFile, err := os.Open(os.Args[1])
	if err != nil { panic(err) }
	jsonParser := json.NewDecoder(traceDataFile)
	err = jsonParser.Decode(&desTraceData); 
	if err != nil { panic(err) }
	fmt.Printf("%s\n %s\n %s\n %s\n", 
		desTraceData.SimulatorName, 
		desTraceData.ModelName, 
		desTraceData.CaptureDate, 
		desTraceData.CommandLineArgs)

	// ok, so let's create a map of the LP names -> integers so we can setup
	// arrays/slices of LPs; while we're running through the event list, let's do what
	// we can to verify the integrity of the data.  at this point all we can do is
	// ensure that the send time is less than the receive time.  
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
		if eventData.SendTime > eventData.ReceiveTime {log.Panic("Event %v has send time greater than receive time", eventData)}
	}
	// build the reverse map: integers -> LP Names
	mapIntToLPName := make([]string, numOfLPs)
	for key, value := range mapLPNameToInt {mapIntToLPName[value] = key}

	// verification prints
	fmt.Printf("LP2Int: %v\n", mapLPNameToInt)
//	fmt.Printf("Int2LP: %v\n", mapIntToLPName)

	// now we need to setup a data structure for events.  internally we're going to
	// store LP names with their integer map value.

	// since we're storing events into an array indexed by the LP in question (sender
	// or receiver), we will only store the other "companion" LP internally

	type eventData struct {
		companionLP int
		sendTime float64
		receiveTime float64
	}
	// now we need to partition the events by LP for our first analysis.  lps is an
	// slice (of slices) that we will use to store the events associated with each LP.
	// we have to define initial size and capacity limits to the go slices.  while we
	// will include code to resize the slices, since they are large, let's try to
	// start out with capacities that are sufficient.  as a starting point, we will
	// assume that the events are approximately equally distributed among the LPs.
	// however, we will assume that the distribution of events may be off by as
	// much as 10%.
		
	capOfLPEventSlice := int(len(desTraceData.Events)/numOfLPs + int(.1*float64((len(desTraceData.Events)/(numOfLPs)))))
	if capOfLPEventSlice < 2048 {capOfLPEventSlice = 2048}
	lps := make([][]eventData, numOfLPs)
	for i := range lps {
		lps[i] = make([]eventData,2048,capOfLPEventSlice)
	}

	lpIndex := make([]int, numOfLPs)

	for i := 0; i < numOfLPs; i++ {lpIndex[i] = -1}

	// in this step we will be looking at events seen at the receiving LP.  the first
	// step it to store the event data into the lps by the receiving LP id.

	for _, traceEvent := range desTraceData.Events {
		rLP := mapLPNameToInt[traceEvent.ReceiveLP]

		lpIndex[rLP]++
		lps[rLP][lpIndex[rLP]].companionLP = mapLPNameToInt[traceEvent.SendLP]
		lps[rLP][lpIndex[rLP]].receiveTime = traceEvent.ReceiveTime
		lps[rLP][lpIndex[rLP]].sendTime = traceEvent.SendTime
	}
	/* for debugging
	for i := range lps {
		fmt.Printf("rLP %v, ",mapIntToLPName[i])
		for j := 0; j <= lpIndex[i]; j++ {fmt.Printf("sLP: %v, sTS: %v, rTS %v ",
			mapIntToLPName[lps[i][j].companionLP],
			lps[i][j].sendTime,
			lps[i][j].receiveTime)
		}
		fmt.Printf("\n")
	}
	*/

	// in this step we will be looking at events seen at the sending LPsd.  the first
	// step it to store the event data into the lps by the sending LP id.

	// reset the lpIndex pointers
	for i := 0; i < numOfLPs; i++ {lpIndex[i] = -1}

	for _, traceEvent := range desTraceData.Events {
		sLP := mapLPNameToInt[traceEvent.SendLP]

		lpIndex[sLP]++
		lps[sLP][lpIndex[sLP]].companionLP = mapLPNameToInt[traceEvent.ReceiveLP]
		lps[sLP][lpIndex[sLP]].receiveTime = traceEvent.ReceiveTime
		lps[sLP][lpIndex[sLP]].sendTime = traceEvent.SendTime
	}
	/* for debugging
	for i := range lps {
		fmt.Printf("sLP %v, ",mapIntToLPName[i])
		for j := 0; j <= lpIndex[i]; j++ {fmt.Printf("rLP: %v, sTS: %v, rTS %v ",
			mapIntToLPName[lps[i][j].companionLP],
			lps[i][j].sendTime,
			lps[i][j].receiveTime)
		}
		fmt.Printf("\n")
	}
	*/
	return
}
