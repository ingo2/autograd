package nn

import (
	"fmt"
	"math/rand"
	"strings"
)

// MLP represents a Multi Layer Perceptron.
type MLP struct {
	Layers []*Layer
}

// NewMLP builds an MLP from layerSizes, where layerSizes[0] is the input size and every further
// entry is the output size of one weight layer. A network with n entries therefore has n-1 weight
// layers, and layerActivations holds one activation per weight layer: layerActivations[k] is
// applied to the layer mapping layerSizes[k] to layerSizes[k+1]. The input has no activation, so
// layerActivations is one shorter than layerSizes and its last entry is the output activation.
func NewMLP(rng *rand.Rand, layerSizes []int, layerActivations []ActivationFunc) *MLP {
	if len(layerSizes) < 2 {
		panic(fmt.Sprintf(
			"layerSizes must have at least 2 entries, got %d", len(layerSizes)))
	}
	if len(layerActivations) != len(layerSizes)-1 {
		panic(fmt.Sprintf(
			"layerActivations must have one entry per weight layer (len(layerSizes)-1 = %d), got %d",
			len(layerSizes)-1, len(layerActivations)))
	}

	for k, s := range layerSizes {
		if s <= 0 {
			panic(fmt.Sprintf("layerSizes[%d] must be > 0, got %d", k, s))
		}
	}

	layers := make([]*Layer, len(layerSizes)-1)
	for k := range len(layerSizes) - 1 {
		layers[k] = NewLayer(rng, layerSizes[k], layerSizes[k+1], layerActivations[k])
	}

	return &MLP{Layers: layers}
}

func (m MLP) ZeroGrad() {
	for _, layer := range m.Layers {
		layer.ZeroGrad()
	}
}

// Call computes the MLP's outputs for inputs. See Neuron.Call for the meaning of noGrad.
func (m MLP) Call(inputs []*Value, noGrad bool) []*Value {
	if len(m.Layers) == 0 {
		panic("MLP has no layers")
	}
	outputs := m.Layers[0].Call(inputs, noGrad)
	for _, layer := range m.Layers[1:] {
		outputs = layer.Call(outputs, noGrad)
	}
	return outputs
}

// InputSize returns the number of inputs the MLP expects, i.e. the width Call requires. Returns 0
// for an MLP without layers or neurons.
func (m MLP) InputSize() int {
	if len(m.Layers) == 0 || len(m.Layers[0].Neurons) == 0 {
		return 0
	}
	return len(m.Layers[0].Neurons[0].Weights)
}

func (m MLP) ParamCount() int {
	count := 0
	for _, layer := range m.Layers {
		count += layer.ParamCount()
	}
	return count
}

func (m MLP) Params() []*Value {
	params := make([]*Value, 0, m.ParamCount())
	for _, layer := range m.Layers {
		params = append(params, layer.Params()...)
	}
	return params
}

// Weights returns the weights of all neurons across all layers, excluding biases.
func (m MLP) Weights() []*Value {
	weights := make([]*Value, 0, m.ParamCount())
	for _, layer := range m.Layers {
		weights = append(weights, layer.Weights()...)
	}
	return weights
}

// Biases returns the bias of each neuron across all layers.
func (m MLP) Biases() []*Value {
	biases := make([]*Value, 0, len(m.Layers))
	for _, layer := range m.Layers {
		biases = append(biases, layer.Biases()...)
	}
	return biases
}

func (m MLP) ToString() string {
	parts := make([]string, len(m.Layers))
	for i, l := range m.Layers {
		parts[i] = fmt.Sprintf("Layer %d: %s", i, l.ToString())
	}
	return fmt.Sprintf("MLP[\n%s\n]", strings.Join(parts, "\n"))
}
