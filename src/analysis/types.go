package main

// Data structure for events.  internally we're going to store LP names with their integer map value.
// since we're storing events into an array indexed by the LP in question (sender or receiver), we will only
// store the other "companion" LP internally.
type eventData struct {
	companionLP int
	sendTime    float64
	receiveTime float64
}

// Data structure for sent events, stored with their integer map value.
type eventSentData struct {
	companionLP int
	sendTime    float64
	receiveTime float64
}

// Data structure for LPs; each LP has a unique id and a list of events it generates.
type lpData struct {
	lpId       int
	events     []eventData
	sentEvents int
}

// functions to support sorting of the events by their receive time
type byReceiveTime []eventData

func (a byReceiveTime) Len() int           { return len(a) }
func (a byReceiveTime) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byReceiveTime) Less(i, j int) bool { return a[i].receiveTime < a[j].receiveTime }

// record a unique int value for each LP and store the total number of sent and received events by that LP.
type lpMap struct {
	toInt          int
	sentEvents     int // CHECK THIS VALUE TO BE GREATER THAN 0
	receivedEvents int
}

// data type to capture each LP's event summary data
type lpEventSummary struct {
	lpId        int
	local       int
	remote      int
	total       int
	cover       [5]int
	localChain  []int
	linkedChain []int
	globalChain []int
}

// using this data structure to hold cycle by cycle analysis results.
type simCycleAnalysisResults struct {
	definingLP      int
	timeStamp       float64
	numAvailable    int
	eventsExhausted bool
}
