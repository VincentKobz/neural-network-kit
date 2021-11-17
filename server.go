package main

import (
	"net/http"

	"github.com/VincentKobz/neural-network-kit/neural_network/neural"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

func main() {
	launcheNeuralNetwork()
	e := echo.New()

	e.Use(middleware.Logger())
	e.Use(middleware.Recover())

	e.GET("/", communicate)

	e.Logger.Fatal((e.Start(":1323")))
}

func communicate(c echo.Context) error {
	return c.String(http.StatusOK, "Connected !")
}

func launcheNeuralNetwork() {
	net := neural.CreateNetwork(2, 4, 1, 0.8)

	data := [][]float64{{1, 1}, {0, 1}, {1, 0}, {0, 0}}
	target := [][]float64{{1}, {1}, {1}, {0}}

	for i := 0; i < 10000; i++ {
		neural.Training(data[0], target[0], net)
		neural.Training(data[1], target[1], net)
		neural.Training(data[2], target[2], net)
		neural.Training(data[3], target[3], net)
	}

	neural.ShowResult(data[0], net)
	neural.ShowResult(data[1], net)
	neural.ShowResult(data[2], net)
	neural.ShowResult(data[3], net)
}
