package neural

import (
	"math"
	"math/rand"
)

func sigmoide(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoideDerivative(x float64) float64 {
	return x * (1.0 - x)
}

func createMatrixRandom(nb, nb2 int) [][]float64 {
	var matrix [][]float64

	for i := 0; i < nb; i++ {
		temp := make([]float64, nb2)
		for j := 0; j < nb2; j++ {
			temp[j] = random(-1, 1)
		}
		matrix = append(matrix, temp)
	}

	return matrix
}

func random(min, max float64) float64 {
	return (max-min)*rand.Float64() + min
}
