package main

import (
	"math/rand"
	"readFile"

	"github.com/SamuelCarroll/DataTypes"
	"github.com/SamuelCarroll/DecisionForest"
	"github.com/SamuelCarroll/DecisionTree"
)

//CLASSES is the number of classes we have for a particular dataset
const CLASSES = 2
const TREES = 1000

func main() {

	allData := readFile.Read("/home/ritadev/Documents/Thesis_Work/Decision-Tree/wine.data")
	allData = genSynthetic(allData)

	DecisionForest.GenMatrix(allData, CLASSES, TREES)

}

//The way we generate Synthetic data will heavily influence our clustering algorithm
func genSynthetic(observed []*dataTypes.Data) []*dataTypes.Data {
	//Find averages and standard deviation to generate synthetic data
	mins := getMins(observed)
	maxs := getMaxs(observed)

	// ensure we don't have something that is too short may have problems if we do
	if len(observed) == 0 || len(mins) == 0 || len(maxs) == 0 {
		return nil
	}

	//Label observed data as zero
	for _, ob := range observed {
		ob.Class = 1
	}

	//Add a length of synthetic data that is equal the length of the observed data
	numSyn := len(observed)
	for loop := 0; loop < numSyn; loop++ {
		newSyn := new(dataTypes.Data)
		newSyn.Class = 2

		//loop over the number of features each observation has
		for i := range mins {
			min := DecisionTree.GetFloatReflectVal(mins[i])
			max := DecisionTree.GetFloatReflectVal(maxs[i])

			tempSD := (max - min) / float64(len(observed))

			synVal := rand.NormFloat64()*tempSD + min
			newSyn.FeatureSlice = append(newSyn.FeatureSlice, synVal)
		}

		//Append the new synthetic data point to the data we have
		observed = append(observed, newSyn)
	}

	return observed
}

func getMins(observations []*dataTypes.Data) []interface{} {
	var mins []interface{}

	for _, ob := range observations {
		for i := range ob.FeatureSlice {
			if len(mins)-1 < i {
				mins = append(mins, 100000000.000)
			}

			switch val := ob.FeatureSlice[i].(type) {
			case float64:
				temp := float64(val)
				if temp < DecisionTree.GetFloatReflectVal(mins[i]) {
					mins[i] = temp
				}
			case bool:
				if val == false {
					mins[i] = 0.0
				}
			}
		}
	}

	return mins
}

func getMaxs(observations []*dataTypes.Data) []interface{} {
	var maxs []interface{}

	for _, ob := range observations {
		for i := range ob.FeatureSlice {
			if len(maxs)-1 < i {
				maxs = append(maxs, -100000000.000)
			}

			switch val := ob.FeatureSlice[i].(type) {
			case float64:
				temp := float64(val)
				if temp > DecisionTree.GetFloatReflectVal(maxs[i]) {
					maxs[i] = temp
				}
			case bool:
				if val == true {
					maxs[i] = 1.0
				}
			}
		}
	}

	return maxs
}
