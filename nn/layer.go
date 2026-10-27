package nn

import (
	"fmt"
	"math/rand"
	"strings"
)

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(rng *rand.Rand, numInputs, numOutputs int, activation ActivationFunc) *Layer {
	if numInputs <= 0 || numOutputs <= 0 {
		panic("numInputs and numOutputs must be > 0")
	}
	neurons := make([]*Neuron, 0, numOutputs)
	for range numOutputs {
		neurons = append(neurons, NewNeuron(rng, numInputs, activation))
	}
	return &Layer{Neurons: neurons}
}

func (l Layer) ZeroGrad() {
	for _, neuron := range l.Neurons {
		neuron.ZeroGrad()
	}
}

// Call computes the layer's outputs for inputs. See Neuron.Call for the meaning of noGrad.
func (l Layer) Call(inputs []*Value, noGrad bool) []*Value {
	if len(inputs) != len(l.Neurons[0].Weights) {
		panic(fmt.Sprintf(
			"len(values)=%d not matching len(weights)=%d", len(inputs), len(l.Neurons[0].Weights)))
	}
	outputs := make([]*Value, len(l.Neurons))
	for k, neuron := range l.Neurons {
		outputs[k] = neuron.Call(inputs, noGrad)
	}
	return outputs
}

func (l Layer) ParamCount() int {
	paramsPerNeuron := len(l.Neurons[0].Weights) + 1
	return len(l.Neurons) * paramsPerNeuron
}

func (l Layer) Params() []*Value {
	params := make([]*Value, 0, l.ParamCount())
	for _, neuron := range l.Neurons {
		params = append(params, neuron.Params()...)
	}
	return params
}

// Weights returns the weights of all neurons in this layer, excluding biases.
func (l Layer) Weights() []*Value {
	weights := make([]*Value, 0, len(l.Neurons)*len(l.Neurons[0].Weights))
	for _, neuron := range l.Neurons {
		weights = append(weights, neuron.Weights...)
	}
	return weights
}

// Biases returns the bias of each neuron in this layer.
func (l Layer) Biases() []*Value {
	biases := make([]*Value, 0, len(l.Neurons))
	for _, neuron := range l.Neurons {
		biases = append(biases, neuron.Bias)
	}
	return biases
}

func (l Layer) ToString() string {
	parts := make([]string, len(l.Neurons))
	for i, n := range l.Neurons {
		parts[i] = n.ToString()
	}
	return fmt.Sprintf("Layer[\n  %s\n]", strings.Join(parts, "\n  "))
}
