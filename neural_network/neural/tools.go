package neural

import (
	"math"
)

func sigmoide(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(x))
}

func sigmoideDerivative(x float64) float64 {
	return x * (1.0 - x)
}
