package autograd

import (
	"math"
	"math/rand"
	"testing"
)

func Test_NewLayer(t *testing.T) {
	layer := NewLayer(rand.New(rand.NewSource(0)), 3, 4, ReLU)

	if len(layer.Neurons) != 4 {
		t.Fatalf("expected 4 neurons, got %d", len(layer.Neurons))
	}
	for i, n := range layer.Neurons {
		if len(n.Weights) != 3 {
			t.Errorf("neuron %d: expected 3 weights, got %d", i, len(n.Weights))
		}
		if n.Activation.Name != ReLU.Name {
			t.Errorf("neuron %d: expected activation %q, got %q", i, ReLU.Name, n.Activation.Name)
		}
	}
}

func Test_Layer_Weights(t *testing.T) {
	layer := NewLayer(rand.New(rand.NewSource(0)), 3, 2, Linear)

	weights := layer.Weights()
	if len(weights) != 6 {
		t.Fatalf("expected 6 weights, got %d", len(weights))
	}
	for i, n := range layer.Neurons {
		for j, w := range n.Weights {
			if weights[i*len(n.Weights)+j] != w {
				t.Errorf("neuron %d weight %d: expected same *Value, got a different one", i, j)
			}
		}
		for _, w := range weights {
			if w == n.Bias {
				t.Errorf("neuron %d: Weights must not include the bias", i)
			}
		}
	}
}

func Test_Layer_Biases(t *testing.T) {
	layer := NewLayer(rand.New(rand.NewSource(0)), 3, 2, Linear)

	biases := layer.Biases()
	if len(biases) != 2 {
		t.Fatalf("expected 2 biases, got %d", len(biases))
	}
	for i, n := range layer.Neurons {
		if biases[i] != n.Bias {
			t.Errorf("neuron %d: expected same *Value as n.Bias", i)
		}
	}
}

func Test_Layer_ZeroGrad(t *testing.T) {
	layer := NewLayer(rand.New(rand.NewSource(0)), 2, 2, Linear)
	layer.Neurons[0].Weights[0].Grad = 5.0
	layer.Neurons[1].Bias.Grad = -3.0

	layer.ZeroGrad()

	for i, n := range layer.Neurons {
		for j, w := range n.Weights {
			if w.Grad != 0 {
				t.Errorf("neuron %d weight %d: expected Grad=0, got %f", i, j, w.Grad)
			}
		}
		if n.Bias.Grad != 0 {
			t.Errorf("neuron %d bias: expected Grad=0, got %f", i, n.Bias.Grad)
		}
	}
}

func Test_Layer_Call(t *testing.T) {
	layer := NewLayer(rand.New(rand.NewSource(0)), 2, 2, Linear)
	// neuron 0: weights [1.0, 0.0], bias 0.0 -> 1.0 * 2.0 + 0.0 * 3.0 + 0.0 = 2.0
	layer.Neurons[0].Weights[0].Data = 1.0
	layer.Neurons[0].Weights[1].Data = 0.0
	layer.Neurons[0].Bias.Data = 0.0
	// neuron 1: weights [0.0, 1.0], bias 0.5 -> 0.0 * 2.0 + 1.0 * 3.0 + 0.5 = 3.5
	layer.Neurons[1].Weights[0].Data = 0.0
	layer.Neurons[1].Weights[1].Data = 1.0
	layer.Neurons[1].Bias.Data = 0.5

	outputs := layer.Call([]*Value{NewValue(2.0), NewValue(3.0)}, false)

	if len(outputs) != 2 {
		t.Fatalf("expected 2 outputs, got %d", len(outputs))
	}
	if math.Abs(outputs[0].Data-2.0) > 1e-9 {
		t.Errorf("output 0: got %f, want 2.0", outputs[0].Data)
	}
	if math.Abs(outputs[1].Data-3.5) > 1e-9 {
		t.Errorf("output 1: got %f, want 3.5", outputs[1].Data)
	}
}

func Test_Layer_Basic_Training(t *testing.T) {
	type Point struct {
		x, y *Value
	}
	// Data generation: 100 samples in x1- and x2- direction, so 10_000 in total.
	y1 := func(x1, x2 float64) float64 { return 0.5*x1 + 0.3*x2 }
	y2 := func(x1, x2 float64) float64 { return -0.2*x1 + 0.8*x2 }
	const sampleCount = 100
	inputs := make([]*Point, 0, sampleCount*sampleCount)
	targets := make([]*Point, 0, sampleCount*sampleCount)
	for x1 := range sampleCount {
		for x2 := range sampleCount {
			x1f := float64(x1) / float64(sampleCount)
			x2f := float64(x2) / float64(sampleCount)
			inputs = append(inputs,
				&Point{NewValue(x1f), NewValue(x2f)})
			targets = append(targets,
				&Point{NewValue(y1(x1f, x2f)), NewValue(y2(x1f, x2f))})
		}
	}

	l := NewLayer(rand.New(rand.NewSource(0)), 2, 2, Linear)

	const learningRate = 1e-3
	converged := false
	for range 1000 {
		totalLoss := 0.0
		for k := range sampleCount * sampleCount {
			// Forward pass.
			scores := l.Call([]*Value{inputs[k].x, inputs[k].y}, false)
			loss := NewValueMSE(scores, []float64{targets[k].x.Data, targets[k].y.Data})
			totalLoss += loss.Data

			// Backward pass.
			l.ZeroGrad()
			loss.Backward()

			// Update step.
			for _, v := range l.Params() {
				v.Data -= v.Grad * learningRate
			}
		}
		totalLoss /= float64(sampleCount * sampleCount)
		if totalLoss < 1e-8 {
			converged = true
			break
		}
	}
	if !converged {
		t.Fatal("did not converge within 1000 epochs")
	}
	approx(t, "l.n[0].bias", l.Neurons[0].Bias.Data, 0.0, 1e-3)
	approx(t, "l.n[0].weight[0]", l.Neurons[0].Weights[0].Data, 0.5, 1e-3)
	approx(t, "l.n[0].weight[1]", l.Neurons[0].Weights[1].Data, 0.3, 1e-3)

	approx(t, "l.n[1].bias", l.Neurons[1].Bias.Data, 0.0, 1e-3)
	approx(t, "l.n[1].weight[0]", l.Neurons[1].Weights[0].Data, -0.2, 1e-3)
	approx(t, "l.n[1].weight[1]", l.Neurons[1].Weights[1].Data, 0.8, 1e-3)
}
