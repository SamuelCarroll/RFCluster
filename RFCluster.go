package main

import (
	"fmt"
	"math/rand"
	"readFile"

	"github.com/SamuelCarroll/DataTypes"
	"github.com/SamuelCarroll/DecisionForest"
	"github.com/SamuelCarroll/DecisionTree"
)

//TREECLASSES is the number of classes we have for training the Forest (2 observed and synthetic)
const TREECLASSES = 2

//CLUSTERCLASSES is the number of classes we will use in the RF Cluster
const CLUSTERCLASSES = 3

//TREES is the number of trees we should have in a forest
const TREES = 1000

func main() {

	origData := readFile.Read("/home/ritadev/Documents/Thesis_Work/Decision-Tree/wine.data")
	allData := genSynthetic(origData)

	_, dissSlice, numData := DecisionForest.GenMatrix(allData, TREECLASSES, TREES)

	if numData != len(dissSlice)/numData {
		fmt.Println("OH NO!!!")
	}
}

//This is a basic dot product function, will take two vectors and return a scalar value
func dotProduct(x, y dataTypes.Data) float64 {
	featLen := len(x.FeatureSlice)

	if featLen != len(y.FeatureSlice) {
		fmt.Println("something funny happened with these variables")
	}

	sum := 0.0
	for i := 0; i < featLen; i++ {
		xFloat := DecisionTree.GetFloatReflectVal(x.FeatureSlice[i])
		yFloat := DecisionTree.GetFloatReflectVal(y.FeatureSlice[i])
		sum += xFloat * yFloat
	}

	return sum
}

//This is a helper function for generating the cluster centers
func initializeS(data []*dataTypes.Data, dissMatrix []float64, numObs, clusters int) ([]*dataTypes.Data, []int, []bool) {
	var medroids []*dataTypes.Data
	var indices []int
	sumDiss := make([]float64, numObs)

	for i := range sumDiss {
		for j := 0; j < numObs; j++ {
			sumDiss[i] += dissMatrix[i*numObs+j]
		}
	}

	used := make([]bool, numObs)
	// for i := 0; i < clusters; i++ {
	minDiss := 0
	for j := range data {
		if sumDiss[j] < sumDiss[minDiss] {
			minDiss = j
		}
	}
	medroids = append(medroids, data[minDiss])
	indices = append(indices, minDiss)
	used[minDiss] = true

	return medroids, indices, used
}

//This is a helper function for the buildPhase function
func findContribution(i, j, numObs int, dissMatrix []float64, indices []int) float64 {
	//For an object j in U (not including the i we are checking) find the dissimilarity between i and j
	Dji := dissMatrix[i*numObs+j]
	var Cji float64

	//Find the dissimilarity between j and the closes object in S
	//NOTE: we are tracking S with the indices matrix
	Dj := 100000000.0
	for _, index := range indices {
		if dissMatrix[index*numObs+j] < Dj {
			Dj = dissMatrix[index*numObs+j]
		}
	}

	if Dj > Dji {
		Cji = Dj - Dji

		if Cji < 0 {
			Cji = 0.0
		}
	}

	return Cji
}

//Details of this function can be found at https://www.cs.umb.edu/cs738/pam1.pdf
func buildPhase(data []*dataTypes.Data, dissMatrix []float64, numObs, clusters int) ([]*dataTypes.Data, []int, []bool) {
	//We have two sets of objects S (medroids) and U (All non-medroid objects)
	//Initialize S by adding to it an object for which the sum of the distances to all other objects is minimal.
	medroids, indices, used := initializeS(data, dissMatrix, numObs, clusters)
	gains := make([]float64, numObs)

	//repeat the following steps until we have a number of medroids equal to the number of clusters we want
	for k := 1; k < clusters; k++ {
		//Consider an object as a candidate for inclusion into the set of selected objects
		for i := 0; i < numObs; i++ {
			//check if we have used this index, only continue if it's a part of U
			if used[i] == false {
				for j := 0; j < numObs; j++ {
					//Find the total gain obtaind by adding i to S call this gain gi
					if used[j] == false && i != j {
						gains[i] += findContribution(i, j, numObs, dissMatrix, indices)
					}
				}
			}
		}

		//Choose the object i that maximizes gi
		newMax := -1000000.0
		maxIndex := 0
		for j := 0; j < numObs; j++ {
			if used[j] == false {
				if gains[j] > newMax {
					newMax = gains[j]
					maxIndex = j
				}
			}
		}

		medroids = append(medroids, data[maxIndex])
		used[maxIndex] = true
	}

	return medroids, indices, used
}

//this will be a helper function to the swapPhase function
func getDiss(dissMatrix []float64, numObs, object int, indices []int) (float64, float64) {
	//we want to find the dissimilarty between an object in U and the two closest objects in S
	Dj := 100000000.00
	Ej := 100000000.00

	for _, index := range indices {
		Dij := dissMatrix[index*numObs+object]

		if Dij < Dj {
			Ej = Dj
			Dj = Dij
		}
	}

	return Dj, Ej
}

//this will be a helper function to the swapPhase function
func getK(dissMatrix []float64, i, h, j, numObs int, indices []int) float64 {
	var Kjih float64
	Dj, Ej := getDiss(dissMatrix, numObs, j, indices)

	//Two cases can occur, the new dissimilarty is either greater than or equal to the old dissimilarty
	if dissMatrix[j*numObs+i] > Dj {
		//if we have a value greater than the current dissimilarity we want to return the
		//minimum between 0 and the difference between new dissimilarity and old dissimilarity
		Kjih = dissMatrix[j*numObs+h] - Dj
		if Kjih > 0 {
			Kjih = 0.0
		}
	} else {
		//if we have a value equal to the old dissimilarity
		//we want to return the minimum of the difference between the old and new dissimilarity and the dissimilarity
		//between our current object and the second closest medroid
		Kjih = dissMatrix[j*numObs+h]
		if Kjih > Ej {
			Kjih = Ej
		}
		Kjih = Kjih - Dj
	}

	return Kjih
}

//details of this function are under the Swap phase at https://www.cs.umb.edu/cs738/pam1.pdf
func swapPhase(data, medroids []*dataTypes.Data, dissMatrix []float64, numObs, clusters int, indices []int, used []bool) ([]*dataTypes.Data, []int) {
	//for each pair (i, h) in S x U comput the effect of Tih on the sum of dissimilarity caused by swapping i from S to U and h from U to S
	//The computation of Tih involves the computation of the contribution Kjih of each object j in U (without h)

	continueSwap := true

	for continueSwap {
		//Since we only make a swap if the smallest value of Tih is less than 0 I set this to zero
		//I also initialized other variables, some are impossible, but dependent on the swap bool
		minTih := 0.0
		continueSwap = false
		swapI := len(indices)
		swapH := numObs

		//index in this case will be i
		for i, index := range indices {
			//for evey value in U
			for h := 0; h < numObs; h++ {
				if used[h] == false {
					//find the contribution of every value remaining in U to swap i and h
					Tih := 0.0
					for j := 0; j < numObs; j++ {
						if j != h && used[j] == false {
							//getK returns the contribution of a variable to a proposed swap
							Tih += getK(dissMatrix, index, h, j, numObs, indices)
						}
					}
					//if we have a value less than zero we will use the one with the greatest contribution to the change
					if Tih < minTih {
						minTih = Tih
						swapI = i
						swapH = h
						continueSwap = true
					}
				}
			}
		}

		// if we want to continue, make the swap
		if continueSwap {
			used[indices[swapI]] = false
			used[swapH] = true
			medroids[swapI] = data[swapH]
			indices[swapI] = swapH
		}
	}

	return medroids, indices
}

func getMedroids(data []*dataTypes.Data, dissMatrix []float64, numObs int, clusters int) ([]*dataTypes.Data, []int) {
	//details found at https://www.cs.umb.edu/cs738/pam1.pdf
	medroids, indices, used := buildPhase(data, dissMatrix, numObs, clusters)

	medroids, indices = swapPhase(data, medroids, dissMatrix, numObs, clusters, indices, used)

	return medroids, indices
}

func extendedJaccard(x, y dataTypes.Data) float64 {
	xdoty := dotProduct(x, y)
	xLen := dotProduct(x, x)
	yLen := dotProduct(y, y)

	ej := xdoty / (xLen + yLen - xdoty)

	return ej
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
