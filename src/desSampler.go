
// this program will extract event trace samples from an event trace set.  the resulting sample event traces
// (and their corresponding json file will be dumped into the subdirectory "./eventTraceSamples"

package main

import "io"
import "os"
import "fmt"
import "strconv"
import "runtime"
import "flag"
import "log"
import "strings"
import "encoding/json"
import "compress/gzip"
import "compress/bzip2"
import "encoding/csv"

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

	// --------------------------------------------------------------------------------
	// process the command line

	invocation := fmt.Sprintf("%v", os.Args)

        var sampleDir string
        flag.StringVar(&sampleDir, "out-dir", "sampleDir",
                "subdirectory where the samples will be written (default: sampleDir/)")

        var trimPercent float64
        flag.Float64Var(&trimPercent, "trim-percent", -1.0,
                "trim the first/last events as a percentage of the total events")

        var trimNum int
        flag.IntVar(&trimNum, "trim-count", -1,
                "trim the first/last (num-of-events * #LPs) events")

        var trimUntil bool
        flag.BoolVar(&trimUntil, "trim-until-all-lps-have-events", false,
                "trim the first/last events until all LPs have events")

        var trimOnlyHead bool
        flag.BoolVar(&trimOnlyHead, "trim-only-head", false,
                "trim only the head events from the file")

	var numSamples int
	flag.IntVar(&numSamples, "num-samples", -1,
		"prepare N samples (after trimming) of events at evenly distributed including head/tail")

	var sampleSize int
	flag.IntVar(&sampleSize, "sample-size", -1,
		"number of events (sample-size * #LPs) to include in each sample")

        // turns on a bunch of debug printing
        var debug bool
        flag.BoolVar(&debug, "debug", false,
                "turn on debugging.")
        
        // the default help out from the flag library doesn't include a way to include argument definitions;
        // these definitions permit us to define our own output from the -help flag.
        var help bool
        flag.BoolVar(&help, "help", false,
                "print out help.")
        flag.BoolVar(&help, "h", help,
                "print out help.")

        flag.Parse()

	trimPercent = trimPercent / 100.0
	err :=  os.MkdirAll(sampleDir, 0777)
	if err != nil {panic(err)}

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
		fmt.Printf("Command Line: sampleDir: %v, trimPercent: %v, trimNum: %v, trimUntil: %v, trimOnlyHead: %v, numSamples: %v, sampleSize: %v, flag.Arg(0): %v\n",
			sampleDir, trimPercent * 100.0, trimNum, trimUntil, trimOnlyHead, numSamples, sampleSize, flag.Arg(0))
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
	log.Printf("Processing input json file: %v\n",flag.Arg(0))
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
			log.Printf("Ignoring unknown element %v from event_data->format field of the model json file.\n", entry)
		}
	}

	if debug { log.Printf("FileFormat: %v; eventDataOrderTable %v\n", desTraceData.EventData.FileFormat, eventDataOrderTable) }

	// sanity check; when error generated turn on debugging and look at -1 field in eventDataOrderTable to discover mssing entry
	for _, entry := range eventDataOrderTable {
		if entry == -1 { log.Fatal("Missing critcal field in event_data->format of model json file; run with --debug to view relevant data.\n") }
	}

	// --------------------------------------------------------------------------------
	// enable the use of all CPUs on the system
	numThreads := runtime.NumCPU()
	runtime.GOMAXPROCS(numThreads)

	log.Printf("Parallelism setup to support up to %v threads.\n", numThreads)

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

	// record a unique int value for each LP and store the total number of sent and received events by that LP.
	type lpMap struct {
		toInt int
		sentEvents int
		receivedEvents int
		// first/lastTime denotes the timestamp of the first/last event in the file for this LP
		// first/lastOccurrence denotes the ordinal number of the first/last event in the file for this LP
		sentEventRecords struct {
			firstTime float64
			firstOccurrence int
			lastTime float64
			lastOccurrence int
			total int
		}
		receivedEventRecords struct {
			firstTime float64
			firstOccurrence int
			lastTime float64
			lastOccurrence int
			total int
		}
	}

	// record the global ranks
	lastFirstSendRecord := -1
        lastFirstSendTime := -1.0
        lastFirstReceiveRecord := -1
        lastFirstReceiveTime := -1.0

	lpNameMap := make(map[string]*lpMap)

	// ultimately we will use lps to hold event data; and lpIndex to record advancements of analysis among the lps
	var lps []lpData
	var lpIndex []int

	// if necessary, build lpNameMap for each new LP and then return pointer to lpMap of said LP
	defineLP := func(lp string) *lpMap {
		item, present := lpNameMap[lp]
		if !present {
			// on first occurrence of an LP, create a record for it.
			lpNameMap[lp] = new(lpMap)

			item = lpNameMap[lp]
			item.toInt = numOfLPs
			item.sentEvents = 0
			item.receivedEvents = 0

			item.sentEventRecords.firstTime = -1.0
			item.sentEventRecords.firstOccurrence = -1
			item.sentEventRecords.lastTime = -1.0
			item.sentEventRecords.lastOccurrence = -1
			item.sentEventRecords.total = 0

			item.receivedEventRecords.firstTime = -1.0
			item.receivedEventRecords.firstOccurrence = -1
			item.receivedEventRecords.lastTime = -1.0
			item.receivedEventRecords.lastOccurrence = -1
			item.receivedEventRecords.total = 0
			
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
		if lp.sentEventRecords.firstTime == -1.0 {
			lp.sentEventRecords.firstTime = sTS
			lp.sentEventRecords.firstOccurrence = numOfEvents
			lastFirstSendTime = sTS
			lastFirstSendRecord = numOfEvents
		}
		lp.sentEventRecords.lastTime = sTS
		lp.sentEventRecords.lastOccurrence = numOfEvents
		lp.sentEventRecords.total++
		lp = defineLP(rLP)
		if lp.receivedEventRecords.firstTime == -1.0 {
			lp.receivedEventRecords.firstTime = rTS
			lp.receivedEventRecords.firstOccurrence = numOfEvents
			lastFirstReceiveTime = sTS
			lastFirstReceiveRecord = numOfEvents
		}
		lp.receivedEventRecords.lastTime = rTS
		lp.receivedEventRecords.lastOccurrence = numOfEvents
		lp.receivedEventRecords.total++
	}

	// process the desTraceData file; processEvent is redefined on the second pass to do the heavy lifting
	profileEventDataFile := func() {

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
			
			if debug {log.Printf("Event recorded: %v, %v, %v, %v\n", eventRecord[eventDataOrderTable[0]], sendTime, eventRecord[eventDataOrderTable[2]], receiveTime)}
			
			processEvent(eventRecord[eventDataOrderTable[0]], sendTime, eventRecord[eventDataOrderTable[2]], receiveTime)
		}
		err = eventFile.Close()
		if err != nil { panic(err) }
	}
	
	// --------------------------------------------------------------------------------
	// ok, now let's process the event data and populate our internal data structures


	// on the first pass, we will collect information on the number of events and the number of LPs

	log.Printf("Processing %v to capture event and LP counts.\n", desTraceData.EventData.EventFile)
	profileEventDataFile()
	log.Printf("Found %v total LPs and %v total Events.\n", numOfLPs, numOfEvents)

	// lps is an array of the LPs; each LP entry will hold the events it received
	lps = make([]lpData, len(lpNameMap))

	// we may need to walk through the LP arrays independently; we will use this array to do so
	lpIndex = make([]int, len(lps))
	// allocate entries in each LP to hold the number events it received
	for _, i := range lpNameMap {
		lpIndex[i.toInt] = -1
		lps[i.toInt].lpId = i.toInt
		lps[i.toInt].events = make([]eventData, i.receivedEvents)
	}

	log.Printf("Processing %v to extract and save samples in %v.\n", flag.Arg(0), sampleDir)
	
	// for now we're just going to trim by percent and get N samples.

	var numEventsToSkip int
	numEventsInSample := sampleSize * numOfLPs
	if trimPercent != -1.0 {
		numEventsToSkip = int(float64(numOfEvents) * trimPercent)
		log.Printf("Skipping %v percent of the events at head/tail of file.\n", trimPercent * 100.0)
 		log.Printf("    Total events: %v, skipping first/last: %v.\n", numOfEvents, numEventsToSkip) 
	}

	type sampleRangeType struct {
		start int
		stop int
	}
	
	var sampleRanges []sampleRangeType
	if numSamples != -1 {
		sampleRanges = make([]sampleRangeType, numSamples)
		log.Printf("Setting the bounds to extract %v samples.\n", numSamples)
		// find the size of events in the (trimmed) source file for each prospective sample
		if (trimOnlyHead) {
			regionWidth := int((float64(numofEvents) - float64(numEventsToSkip)) / float64(numSamples))
		} else {		
			regionWidth := int((float64(numOfEvents) - (2.0 * float64(numEventsToSkip))) / float64(numSamples))
		}
		// let's verify that our samples don't overlap
		if ((numEventsToSkip * 2) + (numEventsInSample * numSamples)) > numOfEvents {
			log.Fatal("Overlapping Samples, choose fewer or smaller samples.  Num Samples: %v, Num Events: %v, Sample size: %v, Num Events to Skip: %v\n", numSamples, numOfEvents, numEventsInSample, numEventsToSkip)
		}

		// location to take first sample
		sampleStart :=
			numEventsToSkip +
			int(float64(regionWidth) / 2.0) -
			int(float64(numEventsInSample) / 2.0)
		

		for i := 0; i < numSamples; i++ {
			sampleRanges[i].start = sampleStart
			sampleRanges[i].stop = sampleEnd
			sampleStart = sampleStart + regionWidth
			sampleEnd = sampleEnd + regionWidth
		}
	} else {
		log.Fatal("Program does not yet support other sampling styles, please specify the number of samples to extract.\n")
	}

	eventFile, csvReader := openEventFile(desTraceData.EventData.EventFile)
	
	// add this sampling to the capture history;
	// can't figure out how to make work as a prepend operation....
	desTraceData.TotalLPs = numOfLPs
	desTraceData.CaptureHistory = append(desTraceData.CaptureHistory, invocation)
	desTraceData.EventData.EventFile = "desMetrics.csv"
	desTraceData.EventData.NumEvents = numOfEvents

	// record where we are at in the original file
	fileLocation := 0
	samplingLoop: for i := 0; i < numSamples; i++ {

		log.Printf("Writing sample %v with events in range %v-%v.\n",
			i+1, sampleRanges[i].start, sampleRanges[i].stop)

		// ok, write out the json file description
		sampleDir := fmt.Sprintf("%v/%v-%v", sampleDir, sampleRanges[i].start, sampleRanges[i].stop)
		err =  os.MkdirAll(sampleDir, 0777)
		if err != nil {panic(err)}
		
		outFile, err := os.Create(fmt.Sprintf("%v/modelSummary.json", sampleDir))
		if err != nil {panic(err)}
		
		jsonEncoder := json.NewEncoder(outFile)
		jsonEncoder.SetIndent("", "    ")
		err = jsonEncoder.Encode(&desTraceData)
		if err != nil {panic(err)}
		err = outFile.Close()
		if err != nil {panic(err)}
		
		// skip to the starting point
		for ; fileLocation < sampleRanges[i].start; fileLocation++ {
			_, err := csvReader.Read()
			if err != nil { if err == io.EOF {break samplingLoop} else { panic(err) }}
		}
		if err != nil { panic(err) }
		

		// writing uncompressded as the bzip2 golang library doesn't yet support writing....bummer
		// for some reason the csv writer was truncating lines; resorting to manual writing
		//		newEventFile, err := os.Create(fmt.Sprintf("%v/desMetrics.csv", sampleDir))
		//		if err != nil {panic(err)}
		//		sampleFile := csv.NewWriter(newEventFile)

		sampleFile, err := os.Create(fmt.Sprintf("%v/desMetrics.csv", sampleDir))
		if err != nil {panic(err)}

		for ; fileLocation < sampleRanges[i].stop; fileLocation ++ {
			eventRecord, err := csvReader.Read()
			if err != nil { if err == io.EOF {break samplingLoop} else { panic(err) }}
			//			err = sampleFile.Write(eventRecord)
			//			if err != nil { panic(err) }
			separator := ""
			for _, field := range(eventRecord) {
				fmt.Fprintf(sampleFile, "%v%v", separator, field)
				separator = ","
			}
			fmt.Fprintf(sampleFile, "\n")
		}
		//		err = newEventFile.Close()
		err = sampleFile.Close()
		if err != nil {panic(err)}
	}
	err = eventFile.Close()
	if err != nil {panic(err)}
	
	log.Printf("Finished.\n")
	return
}	
