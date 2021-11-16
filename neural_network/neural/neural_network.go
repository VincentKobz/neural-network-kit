package neural

import (
	"fmt"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

type Network struct {
	inputs  int
	outputs int

	hiddens        int
	hiddenInput    []float64
	hiddenWeights  [][]float64
	hiddenGradient []float64

	outputWeights  [][]float64
	outputResult   []float64
	outputGradient []float64
	outputInput    []float64
	rate           float64
}

func UpdateWeightsHidden(net Network) {
	for i := 0; i < net.inputs; i++ {
		for j := 0; j < net.hiddens; j++ {
			net.hiddenWeights[i][j] += net.hiddenGradient[j] * net.hiddenInput[i] * net.rate
		}
	}
}

func UpdateWeightsOutput(net Network) {
	for i := 0; i < net.hiddens; i++ {
		for j := 0; j < net.outputs; j++ {
			net.outputWeights[i][j] += net.outputGradient[j] * net.outputInput[i] * net.rate
		}
	}
}

func CalculateHidden(net Network) {
	for i := 0; i < net.hiddens; i++ {
		var x float64
		for j := 0; j < net.inputs; j++ {
			x += net.hiddenWeights[j][i] * net.hiddenInput[j]
		}
		net.outputInput[i] = sigmoide(x)
	}
}

func CalculateOutput(net Network) {
	for i := 0; i < net.outputs; i++ {
		var x float64
		for j := 0; j < net.hiddens; j++ {
			x += net.outputWeights[j][i] * net.outputInput[j]
		}
		net.outputResult[i] = sigmoide(x)
	}
}

func ProcessBackPropagation(net Network, target []float64) {

	for i := 0; i < net.outputs; i++ {
		net.outputGradient[i] = (target[i] - net.outputResult[i]) * sigmoideDerivative(net.outputResult[i])
	}

	for i := 0; i < net.hiddens; i++ {
		var temp float64
		for j := 0; j < net.outputs; j++ {
			temp += net.outputGradient[j] * net.outputWeights[i][j]
		}
		net.hiddenGradient[i] = sigmoideDerivative(net.outputInput[i]) * temp
	}

	UpdateWeightsOutput(net)
	UpdateWeightsHidden(net)
}

func CreateNetwork(nb_input, nb_hiden, nb_output int, rate float64) (net Network) {
	new_net := Network{
		inputs:  nb_input,
		outputs: nb_output,
		hiddens: nb_hiden,
		rate:    rate,
	}
	new_net = InitializeNetwork(new_net)
	return new_net
}

func InitializeNetwork(net Network) Network {

	// Initialize all the array
	net.hiddenInput = make([]float64, net.inputs)
	net.outputInput = make([]float64, net.outputs)
	net.outputResult = make([]float64, net.outputs)
	net.hiddenGradient = make([]float64, net.hiddens)
	net.outputGradient = make([]float64, net.outputs)

	// Initialize random weights for hidden and output layer
	for i := 0; i < net.inputs; i++ {
		for j := 0; j < net.hiddens; j++ {
			net.hiddenWeights[i][j] = rand.Float64()
		}
	}
	for i := 0; i < net.hiddens; i++ {
		for j := 0; j < net.outputs; j++ {
			net.outputWeights[i][j] = rand.Float64()
		}
	}

	return net
}

func Training(data, target []float64, net Network) {

	// Fill hiddenInput with Input Data
	for i := 0; i < net.inputs; i++ {
		net.hiddenInput[i] = data[i]
	}

	// Calculate output of each layer
	CalculateHidden(net)
	CalculateOutput(net)

	// [ BackPropagation ]
	// Calculate Gradient
	ProcessBackPropagation(net, target)
}

func ShowResult(data []float64, net Network) {

	// Fill hiddenInput with Input Data
	for i := 0; i < net.inputs; i++ {
		net.hiddenInput[i] = data[i]
	}

	// Calculate output of each layer
	CalculateHidden(net)
	CalculateOutput(net)

	fmt.Println(net.outputResult[0])
}
