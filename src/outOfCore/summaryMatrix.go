	//  now we will compute and print a summary matrix of the number of events exchanged between two LPs; while we
	// do this, we will also capture the minimum, maximum, and average delta between the send and receive time (for
	// lookahead analysis).  this matrix can be quite large and can actually be too large to create.  in order to
	// save space, we will print only the non-zero entries.  in the cases where it is too large to print, the
	// command line argument "--no-comm-matrix" can be used to suppress it's creation.

	if commSwitchOff == false {
		outFile, err = os.Create("analysisData/eventsExchanged-remote.csv")
		if err != nil {panic(err)}

		outFile2, err := os.Create("analysisData/eventsExchanged-local.csv")
		if err != nil {panic(err)}

		fmt.Fprintf(outFile,"# event exchanged matrix data (remote)\n")
		fmt.Fprintf(outFile,"# receiving LP, sending LP, num of events sent, minimum timestamp delta, maximum timestamp delta, average timestamp delta\n")

		fmt.Fprintf(outFile2,"# event exchanged matrix data (local)\n")
		fmt.Fprintf(outFile2,"# receiving LP, sending LP, num of events sent, minimum timestamp delta, maximum timestamp delta, average timestamp delta\n")
		
		lpEventSendCount := make([]int,numOfLPs)
		aveSendTimeDelta := make([]float64,numOfLPs)
		minTimeDelta := make([]float64,numOfLPs)
		maxTimeDelta := make([]float64,numOfLPs)
		for j, lp := range(lps) {
			for i := range lpEventSendCount {
				lpEventSendCount[i] = 0
				aveSendTimeDelta[i] = 0
				minTimeDelta[i] = math.MaxFloat64
				maxTimeDelta[i] = 0
			}
			for _, event := range lp.events {
				lpEventSendCount[event.companionLP]++
				delta := event.receiveTime - event.sendTime
				aveSendTimeDelta[event.companionLP] = aveSendTimeDelta[event.companionLP] + delta
				if minTimeDelta[event.companionLP] > delta {minTimeDelta[event.companionLP] = delta}
				if maxTimeDelta[event.companionLP] < delta {maxTimeDelta[event.companionLP] = delta}
			}
			for i := range lpEventSendCount {
				if lpEventSendCount[i] != 0 {
					if i != j {
						fmt.Fprintf(outFile,"%v,%v,%v,%v,%v,%v\n", j, i, lpEventSendCount[i],minTimeDelta[i],maxTimeDelta[i],aveSendTimeDelta[i]/float64(lpEventSendCount[i]))
					} else {
						fmt.Fprintf(outFile2,"%v,%v,%v,%v,%v,%v\n", j, i, lpEventSendCount[i],minTimeDelta[i],maxTimeDelta[i],aveSendTimeDelta[i]/float64(lpEventSendCount[i]))
					}
				}
			}
		}
		err = outFile.Close()
		if err != nil {panic(err)}
	}
