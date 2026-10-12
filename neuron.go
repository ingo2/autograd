package autograd

import (
	"fmt"
	"math/rand"
	"strings"
)

type Neuron struct {
	Weights    []*Value
	Bias       *Value
	Activation ActivationFunc
}

func NewNeuron(rng *rand.Rand, numInputs int, activation ActivationFunc) *Neuron {
	weights := make([]*Value, numInputs)
	for i := range weights {
		weights[i] = NewRandomValue(rng, -1.0, 1.0)
	}
	return &Neuron{
		Weights:    weights,
		Bias:       NewRandomValue(rng, -1.0, 1.0),
		Activation: activation,
	}
}

func (n Neuron) ZeroGrad() {
	for _, w := range n.Weights {
		w.Grad = 0
	}
	n.Bias.Grad = 0
}

// Call computes the neuron's output for inputs. If noGrad is true, the forward pass is computed
// directly on float64 data, bypassing the autograd graph: the result has no children and no
// backwardFunc, so Backward() cannot propagate through it.
func (n Neuron) Call(inputs []*Value, noGrad bool) *Value {
	if len(inputs) != len(n.Weights) {
		panic(fmt.Sprintf("len(inputs)=%d not matching len(weights)=%d", len(inputs), len(n.Weights)))
	}
	if noGrad {
		sum := n.Bias.Data
		for k, w := range n.Weights {
			sum += w.Data * inputs[k].Data
		}
		return n.Activation.Call(NewValue(sum), true)
	}
	output := n.Bias
	for k, w := range n.Weights {
		output = output.Add(w.Mul(inputs[k]))
	}
	return n.Activation.Call(output, false)
}

func (n Neuron) Params() []*Value {
	params := make([]*Value, 0, len(n.Weights)+1)
	params = append(params, n.Weights...)
	params = append(params, n.Bias)
	return params
}

func (n Neuron) ToString() string {
	weights := make([]string, len(n.Weights))
	for i, w := range n.Weights {
		weights[i] = fmt.Sprintf("%.4f", w.Data)
	}
	return fmt.Sprintf("Neuron(%s) w=[%s] b=%.4f",
		n.Activation.Name, strings.Join(weights, ", "), n.Bias.Data)
}
