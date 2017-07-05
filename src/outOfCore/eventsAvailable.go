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
