package scoremodel

import (
	"autograd/nn"
	"math"
)

func BestPositionFromValues(model *nn.MLP, validMoves []int, inputs []*nn.Value) []int {
	outputs := model.Call(inputs, true)

	modelArgmaxIndex := validMoves[0]
	modelArgmax := outputs[modelArgmaxIndex].Data
	for _, vm := range validMoves[1:] {
		if outputs[vm].Data > modelArgmax {
			modelArgmaxIndex = vm
			modelArgmax = outputs[vm].Data
		}
	}

	results := make([]int, 0, 1) // Most likely there will be a single maximum.
	const eps = 1e-6
	for _, vm := range validMoves {
		if math.Abs(modelArgmax-outputs[vm].Data) < eps {
			results = append(results, vm)
		}
	}

	return results
}
