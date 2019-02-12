// This program performs the analysis for the desMetrics project at UC. This file does
// the desAnalysis out-of-kernel as a way of working with larger files. The program 
// inputs a JSON file containing profile data of the simulation model for which the event
// trace was captured. The events are stored in a seperate (compressed or not) CSV file
// The name of this file is captured in the json file. This project is developed from a 
// parallel simulation (PDES) perspective and the jargon and analysis is related to this
// field. To understand the documenation, familiarity with PDES terminology is essential. 
// The input JSON file and overall project perspective is availble from the project website. 

// Operationally, this program parses the event trace data file twice, the first pass 
// captures the general characteristics of the file such as number of LPs, total number of 
// and so on. The second pass inputs and stores the event data into internal data structures
// for processing. This approach is followed to maintain the memory footprint as these files
// tend to be large. Memory and time are issues so the program is organized accordingly. In 
// particular, whenever possible, the analysis is partitioned and performed in parallel 
// threads. The program is setup to for a number of threads equal to the processor cores.
// These threads have minimal communication and the program can place a heavy load on the 
// host processor, so plan accordingly.


package main

import (
	"fmt"
	"flag"
	"os"
	"log"
	"strings"
	"encoding/json"
	"encoding/csv"
	"compress/gzip"
	"compress/bzip2"
	"runtime"
	"time"
	"io"
	"strconv"
)
/****************************** GLOBAL VARIABLES ******************************/

// Moved out of main, global variables are initialized to 0 or 0.0 if not declared.
var numOfLPs 							int
var numOfEvents 					int
var numOfInitialEvents 		int
var lpNameMap = make(map[string]*lpMap)
var lps 									[]lpData
var lpIndex 							[]int


// numOfLPs = 0 
// numOfEvents := 0
// numOfInitialEvents := 0
// lpNameMap := make(map[string]*lpMap)
// var lps 			[]lpData
// var lpIndex		[]int


/****************************** DATA STRUCTURES ******************************/

// DS for events. Internally store LP names w/ integer map value. 
// Storing events into an array by LP in question (sender or receiver), we will only store
// the other "companion" LP internally.
type eventData struct {
	companionLP		int
	sendTime			float64
	receiveTime		float64
}

// DS for sent events, stored w/ integer map value
type eventSentData struct {
	companionLP 	int
	sendTime			float64
	receiveTime		float64
}

// DS for LPs; each LP has unique ID and list of events it generates.
type lpData struct {
	lpId					int
	events				[]eventData
	sentEvents		int
}

// Moved to main. Data structure maps integer to LP, holds number of sent/received events
type lpMap struct {
	toInt							int
	sentEvents				int
	receivedEvents 		int
}

// Format of JSON file describing model and location (CSV file) of event data
var desTraceData struct {
	SimulatorName 					string `json:"simulator_name"`
	ModelName 							string `json:"model_name"`
	OriginalCaptureDate 		string `json:"original_capture_date"`
	CaptureHistory 					[]string `json:"capture_history"`
	TotalLPs 								int `json:"total_lps"`
	NumInitialEvents 				int `json:"number_of_initial_events"`
	EventData struct {			
		EventFile 						string `json:"file_name"`
		FileFormat 						[]string `json:"format"`
		NumEvents 						int `json:"total_events"`
	} 											`json:"event_data"`
	DateAnalyzed 						string `json:"date_analyzed"`
}


type byReceiveTime []eventData
func (a byReceiveTime) Len() int 						{return len(a)}
func (a byReceiveTime) Swap(i, j int)				{a[i], a[j] = a[j], a[i]}
func (a byReceiveTime) Less(i, j int) bool	{return a[i].receiveTime < a[j].receiveTime}

/****************************** GLOBAL FUNCTIONS ******************************/

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func getTime() string {
	return time.Now().Format(time.RFC850)
}

// Compute running mean and variance for a stream of data values
func UpdateRunningMeanVariance(currentMean float64, varianceSum float64, newValue float64, numValues int) (float64, float64) {
	increment := newValue - currentMean
	currentMean += (increment / float64(numValues))
	varianceSum += (increment * increment)
	return currentMean, varianceSum
}

// Connect to compressed (gz or bz2) and uncompressed eventData csv files
func openEventFile(fileName string) (*os.File, *csv.Reader) {
	eventFile, err := os.Open(fileName)
	check(err)
	var inFile *csv.Reader
	if strings.HasSuffix(fileName, ".gz") || strings.HasSuffix(fileName, ".gzip") {
		unpackRdr, err := gzip.NewReader(eventFile)
		check(err)
		inFile = csv.NewReader(unpackRdr)
	} else {
		if strings.HasSuffix(fileName, "bz2") || strings.HasSuffix(fileName, "bzip2") {
			unpackRdr := bzip2.NewReader(eventFile)
			inFile = csv.NewReader(unpackRdr)
		} else {
			inFile = csv.NewReader(eventFile)
		}
	}

	// '#' is a comment
	inFile.Comment = '#'
	inFile.FieldsPerRecord = len(desTraceData.EventData.FileFormat)
	
	return eventFile, inFile
}

// Move this out of main
func processEvent(sLP string, sTS float64, rLP string, rTS float64) {
	numOfEvents++	
	lp := defineLP(sLP)
	lp.sentEvents++
	lp = defineLP(rLP)
	lp.receivedEvents++

	if sTS <= 0 {
		numOfInitialEvents++
	}
}

	// Move this out of main
func defineLP(lp string) *lpMap {
	item, present := lpNameMap[lp]
	if !present {
		lpNameMap[lp] = new(lpMap)
		lpNameMap[lp].toInt = numOfLPs
		item = lpNameMap[lp]
		numOfLPs++
	}
	return item
}


func main() {
	fmt.Println("OUT OF KERNEL.")

	/**************************************************************************************************/
	// Process command line

	var analyzeAllData bool
	flag.BoolVar(&analyzeAllData, "analyze-everything", false, "Turn on all analysis capabilities")

	var analyzeReceiveTimeData bool
	flag.BoolVar(&analyzeReceiveTimeData, "analyze-event-receiveTimes", false, "Turn on analysis of an LP's events by receive time")

	// This file can be large, option to turn it off
	var commSwitchOff bool
	flag.BoolVar(&commSwitchOff, "no-comm-matrix", false, "Turn off generation of the file eventsExchanged.csv")

	// Turns on debugging print statements
	var debug bool
	flag.BoolVar(&debug, "debug", false, "Turn on debugging")
	
	// Default help out from flag library doesn't include a way to include argument definitions. 
	// The -help flag defines our own output
	var help bool
	flag.BoolVar(&help, "help", false, "Print out help")
	flag.BoolVar(&help, "h", help, "Print out help.")

	flag.Parse()

	if help {
		fmt.Print("Usage: desAnalysis [options...] FILE \nAnalyze the event trace data described by the JSON file FILE.\n\n")
		flag.PrintDefaults()
		os.Exit(0)
	}

	if flag.NArg() != 1 {
		fmt.Printf("Invalid number of arguments (%v); only one expected.\n\n", flag.NArg())
		fmt.Print("Usage: desAnalysis [options...] FILE \nAnalyze the event trace data described by the JSON file FILE.\n\n")
		flag.PrintDefaults()
		os.Exit(0)
	}
	
	if debug {
		fmt.Printf("Command Line: commSwitchOff: %v, debug: %v, Args: %v.\n", commSwitchOff, debug, flag.Arg(0))
	}

	/**************************************************************************************************/
	// Process the JSON file

	traceDataFile, err := os.Open(flag.Arg(0))
	defer traceDataFile.Close()
	check(err)
	fmt.Printf("Processing input JSON file: %v\n", flag.Arg(0))
	jsonDecoder := json.NewDecoder(traceDataFile)
	err = jsonDecoder.Decode(&desTraceData)
	check(err)
	if debug {
		fmt.Printf("JSON file parsed successfully. Summary info: \n Simulator Name: %s\n Model Name: %s\n Oiginal Capture Date: %s\n Capture History: %s \n CSV File of Event Data: %s\n Format of Event Data: %v\n",
			desTraceData.SimulatorName,
			desTraceData.ModelName,
			desTraceData.OriginalCaptureDate,
			desTraceData.CaptureHistory,
			desTraceData.EventData.EventFile,
			desTraceData.EventData.FileFormat)
	}

	desTraceData.TotalLPs = -1
	desTraceData.EventData.NumEvents = 0
	desTraceData.DateAnalyzed = ""
	desTraceData.NumInitialEvents = 0

	// Map the CSV fields from event data file to order we need for our internal DS's (sLP, sTS, rLP, rTS).
	// eventDataOrderTable indicates which CSV entry corresponds. For example, eventDataOrderTable[0] will 
	// hold the index where the sLP field lies in desTraceData.EventData.FileFormat
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

	if debug {
		fmt.Printf("File Format: %v\n eventDataOrderTable %v\n", desTraceData.EventData.FileFormat, eventDataOrderTable) 
	}
	for _, entry := range eventDataOrderTable {
		if entry == -1 {
			log.Fatal("Missing critical field in event_data->format of model JSON file. Run with --debug flag to view relevant data.\n")
		}
	}

	/**************************************************************************************************/
	
	numThreads := runtime.NumCPU()

	// for kvack
	runtime.GOMAXPROCS(16)
	
	fmt.Printf("%v: Parallelism setup to support up to %v threads.\n", getTime(), numThreads)


	// Memory is the big issue, to minimize the size of our data structures, we process the JSON file twice.
	// 1st: map -> int array for the LPs, count total number of LPs, and total number of events.
	// Used to build principle DS to hold events by LPs (by assuming that the events are more or less 
	// uniformly distributed among LPs). 
	// 2nd: Store events in our internal DS. use processEvent function to manage each step. Between parses
	// it will be changed to point to the addEvent function

	// lpNameMap := make(map[string]*lpMap)




	processEventDataFile := func() {
		eventFile, csvReader := openEventFile(desTraceData.EventData.EventFile)
		fmt.Println(eventFile)
		readLoop: for {
			eventRecord, err := csvReader.Read()
			fmt.Println(eventRecord)
			if err != nil { if err == io.EOF { break readLoop } else {panic(err)}	}

			for i := range eventRecord {
				eventRecord[i] = strings.Trim(eventRecord[i], "'")
				eventRecord[i] = strings.Trim(eventRecord[i], "`")
			}

			sendTime, _ := strconv.ParseFloat(eventRecord[eventDataOrderTable[1]], 64)
			receiveTime, _ := strconv.ParseFloat(eventRecord[eventDataOrderTable[3]], 64)

			if sendTime > receiveTime {
				log.Fatal("Event has send time greater than receive time: %v %v %v %v\n", 
							eventRecord[eventDataOrderTable[0]], sendTime, eventRecord[eventDataOrderTable[2]], receiveTime)
				log.Fatal("Aborting")
			}

			if debug {
				fmt.Printf("Event recorded: %v, %v, %v, %v\n", eventRecord[eventDataOrderTable[0]], sendTime, eventRecord[eventDataOrderTable[2]], receiveTime)
			}

			processEvent(eventRecord[eventDataOrderTable[2]], sendTime, eventRecord[eventDataOrderTable[2]], receiveTime)

			
		}
		err = eventFile.Close()
		check(err)
	}

	/**************************************************************************************************/
	// Process event data and populate our internal data structures

	// 1st Pass: Collect info on number of events and number of LPs

	fmt.Printf("%v: Processing %v to capture event and LP counts.\n", getTime(), desTraceData.EventData.EventFile)
	processEventDataFile()




}