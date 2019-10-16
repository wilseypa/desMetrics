package main

// variables to record the total number of LPs and events; numOfLPs will also be used to
// enumerate unique integers for each LP recorded; set during the first pass over the JSON file.
var numOfEvents int
var numOfInitialEvents int
var numOfLPs int

var lpNameMap map[string]*lpMap

// ultimately we will use lps to hold event data; and lpIndex to record advancements of analysis among the lps
var lps []lpData
var lpIndex []int

// format of json file describing the model and location (csv file) of the event data
var desTraceData struct {
	SimulatorName       string   `json:"simulator_name"`
	ModelName           string   `json:"model_name"`
	OriginalCaptureDate string   `json:"original_capture_date"`
	CaptureHistory      []string `json:"capture_history"`
	TotalLPs            int      `json:"total_lps"`
	NumInitialEvents    int      `json:"number_of_initial_events"`
	EventData           struct {
		EventFile  string   `json:"file_name"`
		FileFormat []string `json:"format"`
		NumEvents  int      `json:"total_events"`
	} `json:"event_data"`
	DateAnalyzed string `json:"date_analyzed"`
}

var eventDataOrderTable [4]int

var maxLPSentArray int
var zeroSentLPs int

var maxLPEventArray int
var zeroReceivedLPs int

var chainLength int

var mapIntToLPName []string
