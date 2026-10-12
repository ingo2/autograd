package autograd

import (
	"math"
	"math/rand"
	"testing"
)

func Test_NewNeuron(t *testing.T) {
	n := NewNeuron(rand.New(rand.NewSource(0)), 3, ReLU)

	if len(n.Weights) != 3 {
		t.Fatalf("expected 3 weights, got %d", len(n.Weights))
	}
	if n.Bias == nil {
		t.Fatal("expected bias to be set")
	}
	if n.Activation.Name != ReLU.Name {
		t.Fatalf("expected activation %q, got %q", ReLU, n.Activation)
	}
	for i, w := range n.Weights {
		if w == nil {
			t.Fatalf("weight %d is nil", i)
		}
		if w.Data < -1.0 || w.Data > 1.0 {
			t.Fatalf("weight %d out of range: %f", i, w.Data)
		}
	}
	if n.Bias.Data < -1.0 || n.Bias.Data > 1.0 {
		t.Fatalf("bias out of range: %f", n.Bias.Data)
	}
}

func Test_ZeroGrad(t *testing.T) {
	n := NewNeuron(rand.New(rand.NewSource(0)), 2, Linear)
	n.Weights[0].Grad = 3.0
	n.Weights[1].Grad = -1.5
	n.Bias.Grad = 7.0

	n.ZeroGrad()

	for i, w := range n.Weights {
		if w.Grad != 0 {
			t.Errorf("weight %d: expected Grad=0, got %f", i, w.Grad)
		}
	}
	if n.Bias.Grad != 0 {
		t.Errorf("bias: expected Grad=0, got %f", n.Bias.Grad)
	}
}

func Test_Call_Linear(t *testing.T) {
	n := NewNeuron(rand.New(rand.NewSource(0)), 2, Linear)
	n.Weights[0].Data = 0.5
	n.Weights[1].Data = -0.5
	n.Bias.Data = 0.1
	// 0.5 * 2.0 + (-0.5) * 3.0 + 0.1 = -0.4; linear passes it through unchanged.
	got := n.Call([]*Value{NewValue(2.0), NewValue(3.0)}, false)
	if math.Abs(got.Data-(-0.4)) > 1e-9 {
		t.Errorf("got %f, want -0.4", got.Data)
	}
}

func Test_Call_ReLU(t *testing.T) {
	n := NewNeuron(rand.New(rand.NewSource(0)), 2, ReLU)
	n.Weights[0].Data = 0.5
	n.Weights[1].Data = -0.5
	n.Bias.Data = 0.1
	// pre-activation = -0.4; ReLU clamps to 0.
	got := n.Call([]*Value{NewValue(2.0), NewValue(3.0)}, false)
	if got.Data != 0.0 {
		t.Errorf("got %f, want 0.0", got.Data)
	}
}

func Test_Call_Tanh(t *testing.T) {
	n := NewNeuron(rand.New(rand.NewSource(0)), 2, Tanh)
	n.Weights[0].Data = 0.5
	n.Weights[1].Data = -0.5
	n.Bias.Data = 0.1
	// pre-activation = -0.4; tanh(-0.4) ≈ -0.3799.
	want := math.Tanh(-0.4)
	got := n.Call([]*Value{NewValue(2.0), NewValue(3.0)}, false)
	if math.Abs(got.Data-want) > 1e-9 {
		t.Errorf("got %f, want %f", got.Data, want)
	}
}

func Test_Call_NoGrad_DetachesFromGraph(t *testing.T) {
	n := NewNeuron(rand.New(rand.NewSource(0)), 2, Tanh)
	n.Weights[0].Data = 0.5
	n.Weights[1].Data = -0.5
	n.Bias.Data = 0.1
	inputs := []*Value{NewValue(2.0), NewValue(3.0)}

	withGrad := n.Call(inputs, false)
	noGrad := n.Call(inputs, true)

	if math.Abs(noGrad.Data-withGrad.Data) > 1e-9 {
		t.Errorf("noGrad result = %f, want %f (same as with-grad result)", noGrad.Data, withGrad.Data)
	}
	if noGrad.children != nil {
		t.Errorf("noGrad result has %d children, want none", len(noGrad.children))
	}

	// Backward() through a computation built on top of the noGrad result must not reach the
	// neuron's weights or bias.
	noGrad.Mul(NewValue(2.0)).Backward()
	if n.Weights[0].Grad != 0 || n.Weights[1].Grad != 0 || n.Bias.Grad != 0 {
		t.Error("Backward() through a noGrad result must not propagate gradients to the neuron's params")
	}
}

func Test_Neuron_Basic_Training(t *testing.T) {
	type Point struct {
		x, y *Value
	}
	// Create a line g(x) = -0.5*x + 2.0.
	const pointCount = 100
	points := make([]*Point, pointCount)
	for k := range points {
		x := float64(k) / float64(pointCount) // x in [0, 1)
		points[k] = &Point{
			x: NewValue(x),
			y: NewValue(-0.5*x + 2.0),
		}
	}

	n := NewNeuron(rand.New(rand.NewSource(0)), 1, Linear)

	const learningRate = 1e-3
	converged := false
	for range 1000 {
		totalLoss := 0.0
		for _, p := range points {
			// Forward pass.
			score := n.Call([]*Value{p.x}, false)
			loss := score.Add(p.y.Negate()).Pow(2.0) // squared error
			totalLoss += loss.Data
			// Backward pass.
			n.ZeroGrad()
			loss.Backward()
			// Update step.
			for _, v := range n.Params() {
				v.Data -= v.Grad * learningRate
			}
		}
		totalLoss /= float64(len(points))
		if totalLoss < 1e-8 {
			converged = true
			break
		}
	}
	if !converged {
		t.Fatal("did not converge within 1000 epochs")
	}
	approx(t, "bias", n.Bias.Data, 2.0, 1e-3)
	approx(t, "weight[0]", n.Weights[0].Data, -0.5, 1e-3)
}
