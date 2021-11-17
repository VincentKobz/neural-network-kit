package neural

import (
	"fmt"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

// Network struct: inputs, hiddens outputs : Number of intputs, hiddens and outputs layers
type Network struct {
	inputs, hiddens, outputs       int
	networkInput                   []float64
	hiddenGradient, outputGradient []float64

	outputWeights, hiddenWeights [][]float64
	outputResult, hiddenResult   []float64
	rate                         float64
}

// UpdateWeights: Update the network weights after calculating the gradient
func UpdateWeights(net Network) {
	for i := 0; i < net.hiddens; i++ {
		for j := 0; j < net.outputs; j++ {
			net.outputWeights[i][j] += net.outputGradient[j] * net.hiddenResult[i] * net.rate
		}
	}

	for i := 0; i < net.inputs; i++ {
		for j := 0; j < net.hiddens; j++ {
			net.hiddenWeights[i][j] += net.hiddenGradient[j] * net.networkInput[i] * net.rate
		}
	}
}

// UpdateNetwork: Calculate all network outputs
func UpdateNetwork(net Network) {
	// Calculate output for hidden layer
	for i := 0; i < net.hiddens; i++ {
		var x float64
		for j := 0; j < net.inputs; j++ {
			x += net.hiddenWeights[j][i] * net.networkInput[j]
		}
		net.hiddenResult[i] = sigmoide(x)
	}

	// Calculate output for output layer
	for i := 0; i < net.outputs; i++ {
		var x float64
		for j := 0; j < net.hiddens; j++ {
			x += net.outputWeights[j][i] * net.hiddenResult[j]
		}
		net.outputResult[i] = sigmoide(x)
	}
}

// ProcessBackPropagation: Launch back propagation in the network
func ProcessBackPropagation(net Network, target []float64) {
	for i := 0; i < net.outputs; i++ {
		net.outputGradient[i] = (target[i] - net.outputResult[i]) * sigmoideDerivative(net.outputResult[i])
	}

	for i := 0; i < net.hiddens; i++ {
		var temp float64
		for j := 0; j < net.outputs; j++ {
			temp += net.outputGradient[j] * net.outputWeights[i][j]
		}
		net.hiddenGradient[i] = sigmoideDerivative(net.hiddenResult[i]) * temp
	}

	UpdateWeights(net)
}

// CreateNetwork: Create the neural network
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

// InitializeNetwork: Initialize the network
func InitializeNetwork(net Network) Network {

	// Initialize all the array
	net.networkInput = make([]float64, net.inputs)
	net.hiddenResult = make([]float64, net.hiddens)
	net.outputResult = make([]float64, net.outputs)
	net.hiddenGradient = make([]float64, net.hiddens)
	net.outputGradient = make([]float64, net.outputs)

	net.hiddenWeights = createMatrixRandom(net.inputs, net.hiddens)
	net.outputWeights = createMatrixRandom(net.hiddens, net.outputs)

	return net
}

// Training: Train the network
func Training(data, target []float64, net Network) {

	// Fill hiddenInput with Input Data
	copy(net.networkInput, data)

	// Calculate output of each layer
	UpdateNetwork(net)

	// BackPropagation
	ProcessBackPropagation(net, target)
}

// ShowResult: Show the network result
func ShowResult(data []float64, net Network) {

	// Fill hiddenInput with Input Data
	copy(net.networkInput, data)

	// Calculate output of each layer#
	UpdateNetwork(net)

	fmt.Println(net.outputResult[0])
}
