package main

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

// DesTraceData : Format of the modelSummary.json file
type DesTraceData struct {
	SimulatorName       string   `json:"simulator_name"`
	ModelName           string   `json:"model_name"`
	OriginalCaptureDate string   `json:"original_capture_date"`
	CaptureHistory      []string `json:"capture_history"`
	TotalLPs            int      `json:"total_lps"`
	NumInitialEvents    int      `json:"number_of_initial_events"`
	DateAnalyzed        string   `json:"date_analyzed"`
	EventData           struct {
		EventFile  string   `json:"file_name"`
		FileFormat []string `json:"format"`
		NumEvents  int      `json:"total_events"`
	} `json:"event_data"`
}

// ReadInputFile : store into a struct and return
func ReadInputFile(fileName string) (d DesTraceData) {
	traceDataFile, err := os.Open(fileName)
	check(err)
	jsonDecoder := json.NewDecoder(traceDataFile)
	err = jsonDecoder.Decode(&d)
	check(err)
	return
}

// ReadDataOrder : returns the order of the four fields: "sLP", "rLP", "sTS", "rTS"
func ReadDataOrder(d *DesTraceData) []int {
	var ret = [4]int{-1, -1, -1, -1}
	for i, entry := range d.EventData.FileFormat {
		switch entry {
		case "sLP":
			ret[0] = i
		case "sTS":
			ret[1] = i
		case "rLP":
			ret[2] = i
		case "rTS":
			ret[3] = i
		default:
			fmt.Printf("Ignoring unknown element %v from EventData->Format of input JSON file.\n", entry)
		}
	}
	return ret
}

// GetTime : string of the current time for logging
func GetTime() string {
	return time.Now().Format(time.RFC850)
}
