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
	networkInput                   []float64
	hiddenGradient, outputGradient []float64

	outputWeights, hiddenWeights [][]float64
	outputResult, hiddenResult   []float64
	rate                         float64
}

// UpdateWeights: Update the network weights after calculating the gradient
func UpdateWeights(net *Network) {
	for i, elt := range net.hiddenResult {
		for j, elt2 := range net.outputGradient {
			net.outputWeights[i][j] += elt2 * elt * net.rate
		}
	}

	for i, elt := range net.networkInput {
		for j, elt2 := range net.hiddenGradient {
			net.hiddenWeights[i][j] += elt2 * elt * net.rate
		}
	}
}

// UpdateNetwork: Calculate all network outputs
func UpdateNetwork(net *Network) {
	// Calculate output for hidden layer
	for i := 0; i < len(net.hiddenResult); i++ {
		var x float64
		for j, elt2 := range net.networkInput {
			x += net.hiddenWeights[j][i] * elt2
		}
		net.hiddenResult[i] = sigmoide(x)
	}

	// Calculate output for output layer
	for i := 0; i < len(net.outputResult); i++ {
		var x float64
		for j, elt2 := range net.hiddenResult {
			x += net.outputWeights[j][i] * elt2
		}
		net.outputResult[i] = sigmoide(x)
	}
}

// ProcessBackPropagation: Launch back propagation in the network
func ProcessBackPropagation(net *Network, target []float64) {
	for i, elt := range net.outputResult {
		net.outputGradient[i] = (target[i] - elt) * sigmoideDerivative(elt)
	}

	for i, elt := range net.hiddenResult {
		var temp float64

		for j, elt2 := range net.outputGradient {
			temp += elt2 * net.outputWeights[i][j]
		}

		net.hiddenGradient[i] = sigmoideDerivative(elt) * temp
	}

	UpdateWeights(net)
}

// CreateNetwork: Create the neural network
func CreateNetwork(nb_input, nb_hidden, nb_output int, rate float64) (net Network) {
	new_net := Network{
		rate: rate,
	}
	new_net = InitializeNetwork(&new_net, nb_input, nb_hidden, nb_output)
	return new_net
}

// InitializeNetwork: Initialize the network
func InitializeNetwork(net *Network, input, hidden, output int) Network {

	// Initialize all the array
	net.networkInput = make([]float64, input)
	net.hiddenResult = make([]float64, hidden)
	net.outputResult = make([]float64, output)
	net.hiddenGradient = make([]float64, hidden)
	net.outputGradient = make([]float64, output)

	net.hiddenWeights = createMatrixRandom(input, hidden)
	net.outputWeights = createMatrixRandom(hidden, output)

	return *net
}

// Training: Train the network
func Training(data, target []float64, net *Network) {

	// Fill hiddenInput with Input Data
	copy(net.networkInput, data)

	// Calculate output of each layer
	UpdateNetwork(net)

	// BackPropagation
	ProcessBackPropagation(net, target)
}

// ShowResult: Show the network result
func ShowResult(data []float64, net *Network) {

	// Fill hiddenInput with Input Data
	copy(net.networkInput, data)

	// Calculate output of each layer#
	UpdateNetwork(net)

	fmt.Println(net.outputResult[0])
}
