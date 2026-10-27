package scoremodel

import (
	"math/rand"
	"slices"
	"testing"

	"autograd/nn"
)

// identityMLP returns a single-layer MLP that maps each input to the output of the same index, i.e.
// Call(inputs, true)[i].Data == inputs[i].Data.
func identityMLP(size int) *nn.MLP {
	mlp := nn.NewMLP(rand.New(rand.NewSource(0)), []int{size, size},
		[]nn.ActivationFunc{nn.Linear})
	for i, n := range mlp.Layers[0].Neurons {
		for j := range n.Weights {
			if i == j {
				n.Weights[j].Data = 1.0
			} else {
				n.Weights[j].Data = 0.0
			}
		}
		n.Bias.Data = 0.0
	}
	return mlp
}

func valuesOf(data ...float64) []*nn.Value {
	values := make([]*nn.Value, len(data))
	for i, d := range data {
		values[i] = nn.NewValue(d)
	}
	return values
}

func Test_BestPositionFromValues_SingleMax(t *testing.T) {
	mlp := identityMLP(3)
	inputs := valuesOf(0.1, 0.9, 0.5)

	got := BestPositionFromValues(mlp, []int{0, 1, 2}, inputs)

	if !slices.Equal(got, []int{1}) {
		t.Errorf("got %v, want [1]", got)
	}
}

func Test_BestPositionFromValues_TiedMax(t *testing.T) {
	mlp := identityMLP(3)
	inputs := valuesOf(0.5, 0.9, 0.9)

	got := BestPositionFromValues(mlp, []int{0, 1, 2}, inputs)

	if !slices.Equal(got, []int{1, 2}) {
		t.Errorf("got %v, want [1, 2]", got)
	}
}

func Test_BestPositionFromValues_RestrictedToValidMoves(t *testing.T) {
	mlp := identityMLP(3)
	// Position 1 has the overall highest value, but it is not a valid move.
	inputs := valuesOf(0.9, 1.0, 0.9)

	got := BestPositionFromValues(mlp, []int{0, 2}, inputs)

	if !slices.Equal(got, []int{0, 2}) {
		t.Errorf("got %v, want [0, 2]", got)
	}
}
