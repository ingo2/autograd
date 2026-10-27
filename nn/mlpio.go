package nn

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
)

type mlpWeightFile struct {
	Info         json.RawMessage `json:"info,omitempty"`
	Architecture []int           `json:"architecture"`
	Activations  []string        `json:"activations"`
	Layers       []layerWeights  `json:"layers"`
}

type layerWeights struct {
	Weight [][]float64 `json:"weight"`
	Bias   []float64   `json:"bias"`
}

var activationsByName = map[string]ActivationFunc{
	"linear": Linear,
	"relu":   ReLU,
	"tanh":   Tanh,
}

// SaveWeights writes the MLP weights to a JSON file at path.
// The format is human-readable and compatible with PyTorch's nn.Linear convention
// (weight shape [out_features, in_features]).
func (m *MLP) SaveWeights(path string) error {
	return m.SaveWeightsWithInfo(path, nil)
}

// SaveWeightsWithInfo writes the MLP weights to a JSON file at path, like SaveWeights, but also
// marshals info into the file's "info" field if info is non-nil. info is used to record training
// configuration and evaluation results alongside the weights.
func (m *MLP) SaveWeightsWithInfo(path string, info any) error {
	arch := make([]int, len(m.Layers)+1)
	arch[0] = len(m.Layers[0].Neurons[0].Weights)
	acts := make([]string, len(m.Layers))
	lws := make([]layerWeights, len(m.Layers))

	for i, layer := range m.Layers {
		arch[i+1] = len(layer.Neurons)
		acts[i] = layer.Neurons[0].Activation.Name
		weight := make([][]float64, len(layer.Neurons))
		bias := make([]float64, len(layer.Neurons))
		for j, n := range layer.Neurons {
			row := make([]float64, len(n.Weights))
			for k, w := range n.Weights {
				row[k] = w.Data
			}
			weight[j] = row
			bias[j] = n.Bias.Data
		}
		lws[i] = layerWeights{Weight: weight, Bias: bias}
	}

	f := mlpWeightFile{Architecture: arch, Activations: acts, Layers: lws}
	if info != nil {
		raw, err := json.Marshal(info)
		if err != nil {
			return err
		}
		f.Info = raw
	}
	data, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// LoadWeights reads weights from a JSON file and copies them into m.
// Returns an error if the file's architecture does not match m.
func (m *MLP) LoadWeights(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var f mlpWeightFile
	if err := json.Unmarshal(data, &f); err != nil {
		return err
	}
	if err := m.checkArchitecture(f); err != nil {
		return err
	}
	for i, layer := range m.Layers {
		for j, n := range layer.Neurons {
			for k, w := range n.Weights {
				w.Data = f.Layers[i].Weight[j][k]
			}
			n.Bias.Data = f.Layers[i].Bias[j]
		}
	}
	return nil
}

func (m *MLP) checkArchitecture(f mlpWeightFile) error {
	if len(f.Layers) != len(m.Layers) {
		return fmt.Errorf("file has %d layers, model has %d", len(f.Layers), len(m.Layers))
	}
	for i, layer := range m.Layers {
		if len(f.Layers[i].Weight) != len(layer.Neurons) {
			return fmt.Errorf("layer %d: file has %d neurons, model has %d",
				i, len(f.Layers[i].Weight), len(layer.Neurons))
		}
		for j, n := range layer.Neurons {
			if len(f.Layers[i].Weight[j]) != len(n.Weights) {
				return fmt.Errorf("layer %d neuron %d: file has %d weights, model has %d",
					i, j, len(f.Layers[i].Weight[j]), len(n.Weights))
			}
		}
	}
	return nil
}

// NewMLPFromFile creates a new MLP with architecture and weights loaded from a JSON file.
func NewMLPFromFile(path string) (*MLP, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f mlpWeightFile
	if err := json.Unmarshal(data, &f); err != nil {
		return nil, err
	}
	if len(f.Architecture) < 2 {
		return nil, fmt.Errorf("architecture must have at least 2 entries, got %d",
			len(f.Architecture))
	}
	if len(f.Activations) != len(f.Layers) {
		return nil, fmt.Errorf("activations length %d does not match layers length %d",
			len(f.Activations), len(f.Layers))
	}

	rng := rand.New(rand.NewSource(0))
	layers := make([]*Layer, len(f.Layers))
	for i, actName := range f.Activations {
		activation, ok := activationsByName[actName]
		if !ok {
			return nil, fmt.Errorf("unknown activation %q in layer %d", actName, i)
		}
		layers[i] = NewLayer(rng, f.Architecture[i], f.Architecture[i+1], activation)
	}
	m := &MLP{Layers: layers}
	if err := m.LoadWeights(path); err != nil {
		return nil, err
	}
	return m, nil
}
