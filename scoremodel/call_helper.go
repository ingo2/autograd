package scoremodel

import (
	"autograd/nn"
	"cmp"
	"math"
	"slices"
)

// topPositionsFromValues returns the at most k valid moves with the highest model outputs, best
// first, keeping only those whose output is within margin of the highest one. Moves with equal
// outputs keep their order in validMoves.
func topPositionsFromValues(model *nn.MLP, validMoves []int, inputs []*nn.Value, k int,
	margin float64) []int {
	outputs := model.Call(inputs, true)
	sorted := slices.Clone(validMoves)
	slices.SortStableFunc(sorted, func(a, b int) int {
		return cmp.Compare(outputs[b].Data, outputs[a].Data)
	})
	best := outputs[sorted[0]].Data
	top := sorted[:min(k, len(sorted))]
	n := 0
	for n < len(top) && outputs[top[n]].Data >= best-margin {
		n++
	}
	return top[:n]
}

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
