package nn

import (
	"encoding/json"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func Test_NewMLP(t *testing.T) {
	mlp := NewMLP(rand.New(rand.NewSource(0)), []int{2, 3, 1}, []ActivationFunc{ReLU, ReLU})

	if len(mlp.Layers) != 2 {
		t.Fatalf("expected 2 layers, got %d", len(mlp.Layers))
	}
	if len(mlp.Layers[0].Neurons) != 3 {
		t.Errorf("layer 0: expected 3 neurons, got %d", len(mlp.Layers[0].Neurons))
	}
	if len(mlp.Layers[0].Neurons[0].Weights) != 2 {
		t.Errorf("layer 0: expected 2 weights per neuron, got %d",
			len(mlp.Layers[0].Neurons[0].Weights))
	}
	if len(mlp.Layers[1].Neurons) != 1 {
		t.Errorf("layer 1: expected 1 neuron, got %d", len(mlp.Layers[1].Neurons))
	}
	if len(mlp.Layers[1].Neurons[0].Weights) != 3 {
		t.Errorf("layer 1: expected 3 weights per neuron, got %d",
			len(mlp.Layers[1].Neurons[0].Weights))
	}
}

func Test_MLP_InputSize(t *testing.T) {
	mlp := NewMLP(rand.New(rand.NewSource(0)), []int{2, 3, 1}, []ActivationFunc{ReLU, ReLU})

	if got := mlp.InputSize(); got != 2 {
		t.Errorf("expected input size 2, got %d", got)
	}
	if got := (MLP{}).InputSize(); got != 0 {
		t.Errorf("expected input size 0 for an empty MLP, got %d", got)
	}
}

func Test_MLP_Clone(t *testing.T) {
	src := NewMLP(rand.New(rand.NewSource(0)), []int{2, 3, 1}, []ActivationFunc{ReLU, Tanh})
	src.Params()[0].Grad = 1.0
	clone := src.Clone()

	srcParams := src.Params()
	cloneParams := clone.Params()
	if len(cloneParams) != len(srcParams) {
		t.Fatalf("clone has %d params, want %d", len(cloneParams), len(srcParams))
	}
	for i := range srcParams {
		if cloneParams[i] == srcParams[i] {
			t.Fatalf("param %d: clone shares the Value with the original", i)
		}
		if cloneParams[i].Data != srcParams[i].Data {
			t.Errorf("param %d: clone %g, want %g", i, cloneParams[i].Data, srcParams[i].Data)
		}
	}
	if cloneParams[0].Grad != 0 {
		t.Errorf("clone grad = %g, want 0", cloneParams[0].Grad)
	}

	inputs := []*Value{NewValue(0.5), NewValue(-1.0)}
	if got, want := clone.Call(inputs, true)[0].Data, src.Call(inputs, true)[0].Data; got != want {
		t.Errorf("clone output %g, want %g", got, want)
	}

	// Changing the clone leaves the original untouched.
	cloneParams[0].Data += 1.0
	if srcParams[0].Data == cloneParams[0].Data {
		t.Error("changing the clone changed the original")
	}
}

func Test_NewMLP_PanicsOnTooFewSizes(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for layerSizes with 1 entry")
		}
	}()
	NewMLP(rand.New(rand.NewSource(0)), []int{3}, []ActivationFunc{})
}

// layerActivations holds one entry per weight layer, so it must be exactly one shorter than
// layerSizes. Both a too-long and a too-short slice are rejected.
func Test_NewMLP_PanicsOnActivationCountMismatch(t *testing.T) {
	cases := map[string][]ActivationFunc{
		"too many": {ReLU, ReLU, ReLU},
		"too few":  {ReLU},
	}
	for name, activations := range cases {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic for %d activations with 3 layer sizes", len(activations))
				}
			}()
			NewMLP(rand.New(rand.NewSource(0)), []int{2, 3, 1}, activations)
		})
	}
}

func Test_NewMLP_PanicsOnZeroSize(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for layerSizes containing 0")
		}
	}()
	NewMLP(rand.New(rand.NewSource(0)), []int{3, 0, 1}, []ActivationFunc{ReLU, ReLU})
}

func Test_MLP_Weights(t *testing.T) {
	mlp := NewMLP(rand.New(rand.NewSource(0)), []int{2, 3, 1}, []ActivationFunc{ReLU, ReLU})

	weights := mlp.Weights()
	// Layer 0: 3 neurons * 2 weights = 6; layer 1: 1 neuron * 3 weights = 3.
	if len(weights) != 9 {
		t.Fatalf("expected 9 weights, got %d", len(weights))
	}
	for _, layer := range mlp.Layers {
		for _, n := range layer.Neurons {
			for _, w := range weights {
				if w == n.Bias {
					t.Error("Weights must not include any bias")
				}
			}
		}
	}
}

func Test_MLP_Biases(t *testing.T) {
	mlp := NewMLP(rand.New(rand.NewSource(0)), []int{2, 3, 1}, []ActivationFunc{ReLU, ReLU})

	biases := mlp.Biases()
	// Layer 0: 3 neurons; layer 1: 1 neuron.
	if len(biases) != 4 {
		t.Fatalf("expected 4 biases, got %d", len(biases))
	}
	want := make([]*Value, 0, 4)
	for _, layer := range mlp.Layers {
		for _, n := range layer.Neurons {
			want = append(want, n.Bias)
		}
	}
	for i, b := range biases {
		if b != want[i] {
			t.Errorf("bias %d: expected same *Value as the neuron's Bias", i)
		}
	}
}

func Test_MLP_ZeroGrad(t *testing.T) {
	mlp := NewMLP(rand.New(rand.NewSource(0)), []int{2, 3, 1}, []ActivationFunc{Linear, Linear})
	mlp.Layers[0].Neurons[0].Weights[0].Grad = 9.0
	mlp.Layers[1].Neurons[0].Bias.Grad = -4.0

	mlp.ZeroGrad()

	for i, layer := range mlp.Layers {
		for j, n := range layer.Neurons {
			for k, w := range n.Weights {
				if w.Grad != 0 {
					t.Errorf("layer %d neuron %d weight %d: expected Grad=0, got %f", i, j, k, w.Grad)
				}
			}
			if n.Bias.Grad != 0 {
				t.Errorf("layer %d neuron %d bias: expected Grad=0, got %f", i, j, n.Bias.Grad)
			}
		}
	}
}

func Test_MLP_Call(t *testing.T) {
	// 2 inputs -> 1 hidden neuron -> 1 output neuron, all Linear.
	mlp := NewMLP(rand.New(rand.NewSource(0)), []int{2, 1, 1}, []ActivationFunc{Linear, Linear})
	// Nidden neuron: weights [1, 1], bias 0 -> sum of inputs = 3.0.
	mlp.Layers[0].Neurons[0].Weights[0].Data = 1.0
	mlp.Layers[0].Neurons[0].Weights[1].Data = 1.0
	mlp.Layers[0].Neurons[0].Bias.Data = 0.0
	// Output neuron: weights [2], bias 1 -> 2*3 + 1 = 7.0.
	mlp.Layers[1].Neurons[0].Weights[0].Data = 2.0
	mlp.Layers[1].Neurons[0].Bias.Data = 1.0

	outputs := mlp.Call([]*Value{NewValue(1.0), NewValue(2.0)}, false)

	if len(outputs) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outputs))
	}
	if outputs[0].Data != 7.0 {
		t.Errorf("got %f, want 7.0", outputs[0].Data)
	}
}

func Test_MLP_Call_PanicsOnNoLayers(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for MLP with no layers")
		}
	}()
	mlp := MLP{}
	mlp.Call([]*Value{NewValue(1.0)}, false)
}

func Test_Schirm_Oder_Nicht(t *testing.T) {
	type Sample struct {
		inputs []*Value
		target float64
	}

	samples := []Sample{
		// Viel Regen, wenig Sonne => Schirm (1)
		{inputs: []*Value{NewValue(0.9), NewValue(0.1)}, target: 1},
		// Wenig Regen, viel Sonne => Kein Schirm (-1)
		{inputs: []*Value{NewValue(0.2), NewValue(0.9)}, target: -1},
		{inputs: []*Value{NewValue(0.7), NewValue(0.6)}, target: 1},
		{inputs: []*Value{NewValue(0.3), NewValue(0.2)}, target: -1},
		{inputs: []*Value{NewValue(0.6), NewValue(0.1)}, target: 1},
		{inputs: []*Value{NewValue(0.4), NewValue(0.8)}, target: -1},
		{inputs: []*Value{NewValue(0.8), NewValue(0.4)}, target: 1},
	}
	numSamples := len(samples)

	// 2 inputs, 3 hidden, 1 output.
	mlp := NewMLP(rand.New(rand.NewSource(0)), []int{2, 3, 1}, []ActivationFunc{Linear, Tanh})

	const learningRate = 0.1
	converged := false
	for range 500 {
		totalLoss := 0.0
		for k := range numSamples {
			// Forward pass.
			score := mlp.Call(samples[k].inputs, false)
			// Single-element MSE, i.e. the squared error (score[0] - target)^2.
			loss := NewValueMSE(score, []float64{samples[k].target})
			totalLoss += loss.Data
			// Backward pass.
			mlp.ZeroGrad()
			loss.Backward()
			// Update.
			for _, param := range mlp.Params() {
				param.Data -= param.Grad * learningRate
			}
		}
		totalLoss /= float64(numSamples)
		log.Printf("totalLoss: %g", totalLoss)
		if totalLoss < 1e-2 {
			converged = true
			break
		}
	}

	if !converged {
		t.Fatal("did not converge")
	}

	approx(t, "mlp(1.0,0.0)", mlp.Call([]*Value{NewValue(1.0), NewValue(0.0)}, true)[0].Data, 1.0, 0.1)
	approx(t, "mlp(0.0,1.0)", mlp.Call([]*Value{NewValue(0.0), NewValue(1.0)}, true)[0].Data, -1.0, 0.1)
	approx(t, "mlp(0.8,0.5)", mlp.Call([]*Value{NewValue(0.8), NewValue(0.5)}, true)[0].Data, 1.0, 0.1)
	approx(t, "mlp(0.1,0.8)", mlp.Call([]*Value{NewValue(0.1), NewValue(0.8)}, true)[0].Data, -1.0, 0.1)
}

// Test_Schirm_Oder_Nicht_Batch learns the same rule as Test_Schirm_Oder_Nicht, but with batch
// gradient descent: a single NewValueMSE node over all samples and one update per epoch, instead
// of one update per sample. The mean loss is then the loss node's value, with no accumulating by
// hand. The learning rate is higher because an epoch now takes one step instead of numSamples.
func Test_Schirm_Oder_Nicht_Batch(t *testing.T) {
	type Sample struct {
		inputs []*Value
		target float64
	}

	samples := []Sample{
		{inputs: []*Value{NewValue(0.9), NewValue(0.1)}, target: 1},
		{inputs: []*Value{NewValue(0.2), NewValue(0.9)}, target: -1},
		{inputs: []*Value{NewValue(0.7), NewValue(0.6)}, target: 1},
		{inputs: []*Value{NewValue(0.3), NewValue(0.2)}, target: -1},
		{inputs: []*Value{NewValue(0.6), NewValue(0.1)}, target: 1},
		{inputs: []*Value{NewValue(0.4), NewValue(0.8)}, target: -1},
		{inputs: []*Value{NewValue(0.8), NewValue(0.4)}, target: 1},
	}
	numSamples := len(samples)
	targets := make([]float64, numSamples)
	for k := range numSamples {
		targets[k] = samples[k].target
	}

	mlp := NewMLP(rand.New(rand.NewSource(0)), []int{2, 3, 1}, []ActivationFunc{Linear, Tanh})

	const learningRate = 1.0
	converged := false
	for epoch := range 500 {
		// Forward pass over the whole batch.
		outputs := make([]*Value, numSamples)
		for k := range numSamples {
			outputs[k] = mlp.Call(samples[k].inputs, false)[0]
		}
		loss := NewValueMSE(outputs, targets)
		if loss.Data < 1e-2 {
			t.Logf("converged after %d epochs, mean loss %g", epoch, loss.Data)
			converged = true
			break
		}
		// Backward pass. Every parameter accumulates its gradient of the mean loss across all
		// samples. So one update per epoch for the batch gradient.
		mlp.ZeroGrad()
		loss.Backward()
		// Update.
		for _, param := range mlp.Params() {
			param.Data -= param.Grad * learningRate
		}
	}

	if !converged {
		t.Fatal("did not converge")
	}

	predict := func(rain, sun float64) float64 {
		return mlp.Call([]*Value{NewValue(rain), NewValue(sun)}, true)[0].Data
	}
	approx(t, "mlp(1.0,0.0)", predict(1.0, 0.0), 1.0, 0.1)
	approx(t, "mlp(0.0,1.0)", predict(0.0, 1.0), -1.0, 0.1)
	approx(t, "mlp(0.8,0.5)", predict(0.8, 0.5), 1.0, 0.1)
	approx(t, "mlp(0.1,0.8)", predict(0.1, 0.8), -1.0, 0.1)
}

func Test_MLP_Train_XOR(t *testing.T) {
	type Sample struct {
		inputs []*Value
		target float64
	}

	// Targets are -1/+1 rather than 0/1: both +-1 are only reached asymptotically by Tanh,
	// but -1/+1 is symmetric around 0 and matches Tanh's natural output range.
	samples := []Sample{
		{inputs: []*Value{NewValue(0.0), NewValue(0.0)}, target: -1.0},
		{inputs: []*Value{NewValue(0.0), NewValue(1.0)}, target: 1.0},
		{inputs: []*Value{NewValue(1.0), NewValue(0.0)}, target: 1.0},
		{inputs: []*Value{NewValue(1.0), NewValue(1.0)}, target: -1.0},
	}
	numSamples := len(samples)

	// Approximates an XOR which requires multi-layer / non-linear capabilities.
	mlp := NewMLP(rand.New(rand.NewSource(0)), []int{2, 2, 1}, []ActivationFunc{Tanh, Tanh})

	const learningRate = 0.1
	converged := false
	for range 1000 {
		totalLoss := 0.0
		for k := range numSamples {
			// Forward pass.
			score := mlp.Call(samples[k].inputs, false)
			loss := NewValueMSE(score, []float64{samples[k].target})
			totalLoss += loss.Data
			// Backward pass.
			mlp.ZeroGrad()
			loss.Backward()
			// Update.
			for _, param := range mlp.Params() {
				param.Data -= param.Grad * learningRate
			}
		}
		totalLoss /= float64(numSamples)
		log.Printf("totalLoss: %g", totalLoss)
		if totalLoss < 1e-3 {
			converged = true
			break
		}
	}
	if !converged {
		t.Fatal("did not converge")
	}

	approx(t, "mlp(0,0)", mlp.Call(samples[0].inputs, true)[0].Data, -1.0, 1e-1)
	approx(t, "mlp(0,1)", mlp.Call(samples[1].inputs, true)[0].Data, 1.0, 1e-1)
	approx(t, "mlp(1,0)", mlp.Call(samples[2].inputs, true)[0].Data, 1.0, 1e-1)
	approx(t, "mlp(1,1)", mlp.Call(samples[3].inputs, true)[0].Data, -1.0, 1e-1)
}

func Test_MLP_SaveLoad_RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	src := NewMLP(rng, []int{3, 4, 2}, []ActivationFunc{ReLU, ReLU})
	// Give weights distinct values so we can detect mis-mapping.
	for i, p := range src.Params() {
		p.Data = float64(i) * 0.1
	}

	path := filepath.Join(t.TempDir(), "weights.json")
	if err := src.SaveWeights(path); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	dst := NewMLP(rand.New(rand.NewSource(0)), []int{3, 4, 2}, []ActivationFunc{ReLU, ReLU})
	if err := dst.LoadWeights(path); err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}

	srcParams := src.Params()
	dstParams := dst.Params()
	if len(srcParams) != len(dstParams) {
		t.Fatalf("param count mismatch: %d vs %d", len(srcParams), len(dstParams))
	}
	for i, sp := range srcParams {
		if sp.Data != dstParams[i].Data {
			t.Errorf("param %d: saved %.6f, loaded %.6f", i, sp.Data, dstParams[i].Data)
		}
	}
}

func Test_MLP_SaveWeightsWithInfo(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	mlp := NewMLP(rng, []int{2, 3, 1}, []ActivationFunc{ReLU, ReLU})

	type info struct {
		Note string  `json:"note"`
		Loss float64 `json:"loss"`
	}
	want := info{Note: "test run", Loss: 0.1234}

	pathWithInfo := filepath.Join(t.TempDir(), "weights_with_info.json")
	if err := mlp.SaveWeightsWithInfo(pathWithInfo, want); err != nil {
		t.Fatalf("SaveWeightsWithInfo: %v", err)
	}

	var fileWithInfo struct {
		Info *info `json:"info"`
	}
	data, err := os.ReadFile(pathWithInfo)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if err := json.Unmarshal(data, &fileWithInfo); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if fileWithInfo.Info == nil {
		t.Fatal("expected non-nil info field")
	}
	if *fileWithInfo.Info != want {
		t.Errorf("info = %+v, want %+v", *fileWithInfo.Info, want)
	}

	pathNoInfo := filepath.Join(t.TempDir(), "weights_no_info.json")
	if err := mlp.SaveWeights(pathNoInfo); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	var fileNoInfo struct {
		Info *info `json:"info"`
	}
	data, err = os.ReadFile(pathNoInfo)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if err := json.Unmarshal(data, &fileNoInfo); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if fileNoInfo.Info != nil {
		t.Errorf("expected nil info field, got %+v", *fileNoInfo.Info)
	}
}

func Test_MLP_NewMLPFromFile_RoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	src := NewMLP(rng, []int{2, 3, 1}, []ActivationFunc{Tanh, Tanh})
	for i, p := range src.Params() {
		p.Data = float64(i+1) * -0.13
	}

	path := filepath.Join(t.TempDir(), "weights.json")
	if err := src.SaveWeights(path); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	dst, err := NewMLPFromFile(path)
	if err != nil {
		t.Fatalf("NewMLPFromFile: %v", err)
	}

	srcParams := src.Params()
	dstParams := dst.Params()
	for i, sp := range srcParams {
		if sp.Data != dstParams[i].Data {
			t.Errorf("param %d: saved %.6f, loaded %.6f", i, sp.Data, dstParams[i].Data)
		}
	}
}

func Test_MLP_LoadWeights_ArchitectureMismatch(t *testing.T) {
	src := NewMLP(rand.New(rand.NewSource(0)), []int{2, 3, 1}, []ActivationFunc{ReLU, ReLU})
	path := filepath.Join(t.TempDir(), "weights.json")
	if err := src.SaveWeights(path); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	dst := NewMLP(rand.New(rand.NewSource(0)), []int{2, 4, 1}, []ActivationFunc{ReLU, ReLU})
	if err := dst.LoadWeights(path); err == nil {
		t.Error("expected error for architecture mismatch, got nil")
	}
}
