package autograd

import (
	"math"
	"testing"
)

func Test_NewValue(t *testing.T) {
	value := NewValue(0.0)

	if value.Data != 0.0 {
		t.Errorf("wrong data: %f", value.Data)
	}
	if value.Grad != 0.0 {
		t.Errorf("wrong grad: %f", value.Grad)
	}
	if value.children != nil {
		t.Errorf("expected no children")
	}
	if value.backwardFunc == nil {
		t.Errorf("expected non-nil BackwardFunc")
	}
	if value.op != "" {
		t.Errorf("expected empty op: %s", value.op)
	}
	if value.label != "" {
		t.Errorf("expected empty label: %s", value.label)
	}
}

func Test_ValueAdd(t *testing.T) {
	a := NewValueWithLabel(1.0, "a")
	b := NewValueWithLabel(0.5, "b")
	c := a.Add(b)

	if c.Data != 1.5 {
		t.Errorf("wrong Data: %f", c.Data)
	}
	if c.op != "+" {
		t.Errorf("wrong op: %s", c.op)
	}
	if len(c.children) != 2 {
		t.Errorf("expected 2 children, got %d", len(c.children))
	}
	if c.children[0] != a || c.children[1] != b {
		t.Errorf("wrong children")
	}
	if c.backwardFunc == nil {
		t.Errorf("expected non-nil backwardFunc")
	}
}

func Test_ValueAdd_Backward(t *testing.T) {
	a := NewValueWithLabel(1.0, "a")
	b := NewValueWithLabel(0.5, "b")
	c := a.Add(b)
	c.Grad = 1.0

	c.Backward()

	if a.Grad != 1.0 {
		t.Errorf("wrong grad for a: %f", a.Grad)
	}
	if b.Grad != 1.0 {
		t.Errorf("wrong grad for b: %f", b.Grad)
	}
}

func Test_ValueMul(t *testing.T) {
	a := NewValueWithLabel(2.0, "a")
	b := NewValueWithLabel(3.0, "b")
	c := a.Mul(b)

	if c.Data != 6.0 {
		t.Errorf("wrong Data: %f", c.Data)
	}
	if c.op != "*" {
		t.Errorf("wrong op: %s", c.op)
	}
	if len(c.children) != 2 {
		t.Errorf("expected 2 children, got %d", len(c.children))
	}
	if c.children[0] != a || c.children[1] != b {
		t.Errorf("wrong children")
	}
}

func Test_ValueMul_Backward(t *testing.T) {
	a := NewValueWithLabel(2.0, "a")
	b := NewValueWithLabel(3.0, "b")
	c := a.Mul(b)
	c.Grad = 1.0

	c.Backward()

	if a.Grad != 3.0 {
		t.Errorf("wrong grad for a: %f", a.Grad)
	}
	if b.Grad != 2.0 {
		t.Errorf("wrong grad for b: %f", b.Grad)
	}
}

func Test_ValueNegate(t *testing.T) {
	a := NewValueWithLabel(3.0, "a")
	c := a.Negate()

	if c.Data != -3.0 {
		t.Errorf("wrong Data: %f", c.Data)
	}

	c.Backward()

	// d/da(-a) = -1
	if a.Grad != -1.0 {
		t.Errorf("wrong grad for a: %f", a.Grad)
	}
}

func Test_ValueReciprocal(t *testing.T) {
	a := NewValueWithLabel(2.0, "a")
	c := a.Reciprocal()

	if c.Data != 0.5 {
		t.Errorf("wrong Data: %f", c.Data)
	}

	c.Backward()

	// d/da(1/a) = -1/a^2 = -1/4 = -0.25
	if a.Grad != -0.25 {
		t.Errorf("wrong grad for a: %f", a.Grad)
	}
}

func Test_ValuePow(t *testing.T) {
	a := NewValueWithLabel(3.0, "a")
	c := a.Pow(2.0)

	if c.Data != 9.0 {
		t.Errorf("wrong Data: %f", c.Data)
	}
	if c.op != "^{2.000000}" {
		t.Errorf("wrong op: %s", c.op)
	}
	if len(c.children) != 1 || c.children[0] != a {
		t.Errorf("wrong children")
	}
}

func Test_ValuePow_Backward(t *testing.T) {
	a := NewValueWithLabel(3.0, "a")
	c := a.Pow(2.0)
	c.Grad = 1.0

	c.Backward()

	// dL/da = M.Grad * e * a**(e-1) = 1 * 2 * 3**1 = 6
	if a.Grad != 6.0 {
		t.Errorf("wrong grad for a: %f", a.Grad)
	}
}

func Test_ValueExp(t *testing.T) {
	a := NewValueWithLabel(1.0, "a")
	c := a.Exp()

	if c.Data != math.E {
		t.Errorf("wrong Data: %f", c.Data)
	}
	if c.op != "exp" {
		t.Errorf("wrong op: %s", c.op)
	}
	if len(c.children) != 1 || c.children[0] != a {
		t.Errorf("wrong children")
	}
}

func Test_ValueExp_Backward(t *testing.T) {
	a := NewValueWithLabel(1.0, "a")
	c := a.Exp()
	c.Grad = 1.0

	c.Backward()

	// dL/da = M.Grad * M.Data = 1 * e^1 = e
	if a.Grad != math.E {
		t.Errorf("wrong grad for a: %f", a.Grad)
	}
}

func Test_ValueLog(t *testing.T) {
	a := NewValueWithLabel(1.0, "a")
	c := a.Log()

	if c.Data != 0.0 {
		t.Errorf("wrong Data: %f", c.Data)
	}
	if c.op != "ln" {
		t.Errorf("wrong op: %s", c.op)
	}
	if len(c.children) != 1 || c.children[0] != a {
		t.Errorf("wrong children")
	}
}

func Test_ValueLog_Backward(t *testing.T) {
	a := NewValueWithLabel(1.0, "a")
	c := a.Log()
	c.Grad = 1.0

	c.Backward()

	// dL/da = M.Grad * 1/a = 1 * 1/1 = 1
	if a.Grad != 1.0 {
		t.Errorf("wrong grad for a: %f", a.Grad)
	}
}

func Test_ValueLog_Zero(t *testing.T) {
	// ln(0) is -Inf and its derivative 1/0 is +Inf. The clip-before-log trick must keep both
	// the forward and backward results finite.
	a := NewValueWithLabel(0.0, "a")
	c := a.Log()

	if math.IsInf(c.Data, 0) || math.IsNaN(c.Data) {
		t.Errorf("Data should be finite for input 0, got %f", c.Data)
	}

	c.Grad = 1.0
	c.Backward()

	if math.IsInf(a.Grad, 0) || math.IsNaN(a.Grad) {
		t.Errorf("grad should be finite for input 0, got %f", a.Grad)
	}
	if a.Grad <= 0 {
		t.Errorf("grad should be positive for input 0, got %f", a.Grad)
	}
}

func Test_ValueReLU(t *testing.T) {
	pos := NewValueWithLabel(3.0, "pos")
	neg := NewValueWithLabel(-2.0, "neg")

	if pos.ReLU().Data != 3.0 {
		t.Errorf("wrong Data for positive input: %f", pos.ReLU().Data)
	}
	if neg.ReLU().Data != 0.0 {
		t.Errorf("wrong Data for negative input: %f", neg.ReLU().Data)
	}
}

func Test_ValueReLU_Backward(t *testing.T) {
	pos := NewValueWithLabel(3.0, "pos")
	cPos := pos.ReLU()
	cPos.Grad = 1.0
	cPos.Backward()

	// input > 0: gradient passes through unchanged.
	if pos.Grad != 1.0 {
		t.Errorf("wrong grad for positive input: %f", pos.Grad)
	}

	neg := NewValueWithLabel(-2.0, "neg")
	cNeg := neg.ReLU()
	cNeg.Grad = 1.0
	cNeg.Backward()

	// input <= 0: gradient is blocked.
	if neg.Grad != 0.0 {
		t.Errorf("wrong grad for negative input: %f", neg.Grad)
	}
}

func Test_ValueTanh(t *testing.T) {
	a := NewValueWithLabel(0.0, "a")
	c := a.Tanh()

	if c.Data != 0.0 {
		t.Errorf("wrong Data: %f", c.Data)
	}
	if c.op != "tanh" {
		t.Errorf("wrong op: %s", c.op)
	}
	if len(c.children) != 1 || c.children[0] != a {
		t.Errorf("wrong children")
	}
}

func Test_ValueTanh_Backward(t *testing.T) {
	a := NewValueWithLabel(0.0, "a")
	c := a.Tanh()
	c.Grad = 1.0

	c.Backward()

	// tanh(0) = 0, so derivative = 1 - 0**2 = 1
	if a.Grad != 1.0 {
		t.Errorf("wrong grad: %f", a.Grad)
	}
}

func Test_ValueMSE(t *testing.T) {
	ys := []*Value{NewValueWithLabel(2.0, "y0"), NewValueWithLabel(4.0, "y1")}
	ts := []float64{1.0, 1.0}

	mse := NewValueMSE(ys, ts)

	// diffs = [1, 3], squared = [1, 9], mean = 5.0
	if mse.Data != 5.0 {
		t.Errorf("wrong Data: %f", mse.Data)
	}
	if mse.op != "mse" {
		t.Errorf("wrong op: %s", mse.op)
	}
	if len(mse.children) != 2 || mse.children[0] != ys[0] || mse.children[1] != ys[1] {
		t.Errorf("wrong children")
	}
	if mse.backwardFunc == nil {
		t.Errorf("expected non-nil backwardFunc")
	}
}

func Test_ValueMSE_Backward(t *testing.T) {
	ys := []*Value{NewValueWithLabel(2.0, "y0"), NewValueWithLabel(4.0, "y1")}
	ts := []float64{1.0, 1.0}

	mse := NewValueMSE(ys, ts)
	mse.Backward()

	// d(mse)/dyi = 2/n * (yi - ti): 2/2*(2-1) = 1.0, 2/2*(4-1) = 3.0
	if ys[0].Grad != 1.0 {
		t.Errorf("wrong grad for y0: %f", ys[0].Grad)
	}
	if ys[1].Grad != 3.0 {
		t.Errorf("wrong grad for y1: %f", ys[1].Grad)
	}
}

func Test_ValueCrossEntropy(t *testing.T) {
	// logits = (0, ln 3) => softmax = (1/4, 3/4)
	ys := []*Value{NewValueWithLabel(0.0, "y0"), NewValueWithLabel(math.Log(3.0), "y1")}
	ts := []float64{0.0, 1.0}

	ce := NewValueCrossEntropy(ys, ts)

	// LSE = ln(exp(0) + exp(ln 3)) = ln(4)
	// H = -(y1 - LSE) = ln(4) - ln(3) = ln(4/3)
	want := math.Log(4.0 / 3.0)
	if math.Abs(ce.Data-want) > 1e-12 {
		t.Errorf("wrong Data: %f, want %f", ce.Data, want)
	}
	if ce.op != "ce" {
		t.Errorf("wrong op: %s", ce.op)
	}
	if len(ce.children) != 2 || ce.children[0] != ys[0] || ce.children[1] != ys[1] {
		t.Errorf("wrong children")
	}
	if ce.backwardFunc == nil {
		t.Errorf("expected non-nil backwardFunc")
	}
}

func Test_ValueCrossEntropy_Backward(t *testing.T) {
	// logits = (0, ln 3) => softmax = (1/4, 3/4)
	ys := []*Value{NewValueWithLabel(0.0, "y0"), NewValueWithLabel(math.Log(3.0), "y1")}
	ts := []float64{0.0, 1.0}

	ce := NewValueCrossEntropy(ys, ts)
	ce.Backward()

	// d(ce)/dyk = softmax(y)k - tk: 0.25 - 0 = 0.25, 0.75 - 1 = -0.25
	if math.Abs(ys[0].Grad-0.25) > 1e-12 {
		t.Errorf("wrong grad for y0: %f", ys[0].Grad)
	}
	if math.Abs(ys[1].Grad-(-0.25)) > 1e-12 {
		t.Errorf("wrong grad for y1: %f", ys[1].Grad)
	}
}

func Test_ValueCrossEntropy_LargeLogits(t *testing.T) {
	// Without the max trick exp(1000) overflows to +Inf and Data becomes NaN.
	// With M = 1000 pulled out the result must stay finite and exact:
	// shifted logits = (0, -1000), LSE = ln(1 + exp(-1000)) + 1000 ~= 1000,
	// H = LSE - y0 ~= 0 (the model is already certain about class 0).
	ys := []*Value{NewValueWithLabel(1000.0, "y0"), NewValueWithLabel(0.0, "y1")}
	ts := []float64{1.0, 0.0}

	ce := NewValueCrossEntropy(ys, ts)

	if math.IsNaN(ce.Data) || math.IsInf(ce.Data, 0) {
		t.Errorf("Data not finite: %f", ce.Data)
	}
	if math.Abs(ce.Data-0.0) > 1e-12 {
		t.Errorf("wrong Data: %f, want ~0", ce.Data)
	}

	ce.Backward()

	// softmax ~= (1, 0), t = (1, 0) => gradients ~= (0, 0), but finite.
	for i, y := range ys {
		if math.IsNaN(y.Grad) || math.IsInf(y.Grad, 0) {
			t.Errorf("grad for y%d not finite: %f", i, y.Grad)
		}
	}
}

func Test_ValueCrossEntropy_NumericalGradient(t *testing.T) {
	// Compare the analytical gradient against the central difference
	// (H(yk+eps) - H(yk-eps)) / (2*eps) for a non-trivial distribution.
	logits := []float64{0.3, -1.2, 2.0}
	ts := []float64{0.2, 0.5, 0.3}
	eps := 1e-6

	// Analytical gradients via backward.
	ys := make([]*Value, len(logits))
	for i, l := range logits {
		ys[i] = NewValue(l)
	}
	ce := NewValueCrossEntropy(ys, ts)
	ce.Backward()

	// Helper: forward pass value of H for given logit data.
	forward := func(data []float64) float64 {
		vs := make([]*Value, len(data))
		for i, d := range data {
			vs[i] = NewValue(d)
		}
		return NewValueCrossEntropy(vs, ts).Data
	}

	for k := range logits {
		plus := make([]float64, len(logits))
		minus := make([]float64, len(logits))
		copy(plus, logits)
		copy(minus, logits)
		plus[k] += eps
		minus[k] -= eps

		numerical := (forward(plus) - forward(minus)) / (2 * eps)
		if math.Abs(ys[k].Grad-numerical) > 1e-6 {
			t.Errorf("grad mismatch for y%d: analytical %f, numerical %f",
				k, ys[k].Grad, numerical)
		}
	}
}

func Test_TopologicalSort(t *testing.T) {
	// Build a graph where a and b are shared between two operations.
	// Without the visited check, a and b would appear twice in the result.
	//
	//   a   b
	//   |\ /|
	//   | X |
	//   |/ \|
	//   c   d
	//    \ /
	//     e
	//
	//   c = a + b
	//   d = a * b
	//   e = c + d
	a := NewValueWithLabel(2.0, "a")
	b := NewValueWithLabel(3.0, "b")
	c := a.Add(b)
	d := a.Mul(b)
	e := c.Add(d)

	topo := topologicalSort(e)

	// Each node must appear exactly once.
	counts := map[*Value]int{}
	for _, v := range topo {
		counts[v]++
	}
	for _, node := range []*Value{a, b, c, d, e} {
		if counts[node] != 1 {
			t.Errorf("node appeared %d times, expected 1", counts[node])
		}
	}

	// e must appear after c and d; c and d must appear after a and b.
	indexOf := func(v *Value) int {
		for i, n := range topo {
			if n == v {
				return i
			}
		}
		return -1
	}
	if indexOf(e) < indexOf(c) || indexOf(e) < indexOf(d) {
		t.Errorf("e must appear after c and d")
	}
	if indexOf(c) < indexOf(a) || indexOf(c) < indexOf(b) {
		t.Errorf("c must appear after a and b")
	}
	if indexOf(d) < indexOf(a) || indexOf(d) < indexOf(b) {
		t.Errorf("d must appear after a and b")
	}
}

func Test_AddMul(t *testing.T) {
	// d = a + b * c
	a := NewValueWithLabel(2.0, "a")
	b := NewValueWithLabel(3.0, "b")
	c := NewValueWithLabel(-5.0, "c")
	d := a.Add(b.Mul(c))

	if d.op != "+" {
		t.Errorf("wrong op: %s", d.op)
	}
	if d.Data != 2.0+3.0*-5.0 {
		t.Errorf("wrong Data: %f", d.Data)
	}

	d.Backward()

	if d.Grad != 1.0 {
		t.Errorf("wrong grad for d: %f", d.Grad)
	}
	if a.Grad != 1.0 {
		t.Errorf("wrong grad for a: %f", a.Grad)
	}
	if b.Grad != -5.0 {
		t.Errorf("wrong grad for b: %f", b.Grad)
	}
	if c.Grad != 3.0 {
		t.Errorf("wrong grad for c: %f", c.Grad)
	}
}

func Test_MulAdd(t *testing.T) {
	// d = a * b + c
	a := NewValueWithLabel(2.0, "a")
	b := NewValueWithLabel(3.0, "b")
	c := NewValueWithLabel(-5.0, "c")
	d := a.Mul(b).Add(c)

	if d.op != "+" {
		t.Errorf("wrong op: %s", d.op)
	}
	if d.Data != 2.0*3.0+-5.0 {
		t.Errorf("wrong Data: %f", d.Data)
	}

	d.Backward()

	if d.Grad != 1.0 {
		t.Errorf("wrong grad for d: %f", d.Grad)
	}
	if c.Grad != 1.0 {
		t.Errorf("wrong grad for c: %f", c.Grad)
	}
	if a.Grad != 3.0 {
		t.Errorf("wrong grad for a: %f", a.Grad)
	}
	if b.Grad != 2.0 {
		t.Errorf("wrong grad for b: %f", b.Grad)
	}
}

func Test_MulPow(t *testing.T) {
	// c = a * b**3
	a := NewValueWithLabel(3.0, "a")
	b := NewValueWithLabel(-2.0, "b")
	c := a.Mul(b.Pow(3))

	if c.op != "*" {
		t.Errorf("wrong op: %s", c.op)
	}
	if c.Data != 3.0*math.Pow(-2.0, 3) {
		t.Errorf("wrong Data: %f", c.Data)
	}

	c.Backward()

	if c.Grad != 1.0 {
		t.Errorf("wrong grad for c: %f", c.Grad)
	}
	// dc/da = b**3 = (-2)**3 = -8
	if a.Grad != -8.0 {
		t.Errorf("wrong grad for a: %f", a.Grad)
	}
	// dc/db = a * 3 * b**2 = 3 * 3 * 4 = 36
	if b.Grad != 36.0 {
		t.Errorf("wrong grad for b: %f", b.Grad)
	}
}

func Test_Backward_SingleNeuron(t *testing.T) {
	// A single neuron: o = tanh(x1*w1 + x2*w2 + b)
	//
	// x1 \
	//     * x1w1 \
	// w1 /        \
	//              + --- + n -> tanh -> o
	// x2 \        /      |
	//     * x2w2 /       b
	// w2 /
	//
	// b is chosen so tanh'(n) = 0.5, making all intermediate grads clean.
	x1 := NewValueWithLabel(2.0, "x1")
	w1 := NewValueWithLabel(-3.0, "w1")
	x2 := NewValueWithLabel(0.0, "x2")
	w2 := NewValueWithLabel(1.0, "w2")
	b := NewValueWithLabel(6.881373587, "b")

	x1w1 := x1.Mul(w1)
	x2w2 := x2.Mul(w2)
	n := x1w1.Add(x2w2).Add(b)
	o := n.Tanh()

	o.Backward()

	const tol = 1e-4
	approx(t, "x1.Grad", x1.Grad, -1.5, tol)
	approx(t, "w1.Grad", w1.Grad, 1.0, tol)
	approx(t, "x2.Grad", x2.Grad, 0.5, tol)
	approx(t, "w2.Grad", w2.Grad, 0.0, tol)
	approx(t, "x1w1.Grad", x1w1.Grad, 0.5, tol)
	approx(t, "x2w2.Grad", x2w2.Grad, 0.5, tol)
	approx(t, "b.Grad", b.Grad, 0.5, tol)
	approx(t, "n.Grad", n.Grad, 0.5, tol)
}

func Test_Backward_SharedNode(t *testing.T) {
	// b = a + a = 2a, so db/da = 2
	// Tests that gradients accumulate correctly when a node is used twice.
	a := NewValueWithLabel(3.0, "a")
	b := a.Add(a)

	b.Backward()

	if a.Data != 3.0 {
		t.Errorf("a.Data changed: %f", a.Data)
	}
	if a.Grad != 2.0 {
		t.Errorf("wrong grad for a: %f", a.Grad)
	}
}

func Test_Backward_ReadmeExample(t *testing.T) {
	// The example from value_liessmich.md: L(a, b, c) = 2ab^2 + abc = (ab) * (2b + c)
	// at a=1, b=-2, c=3. b has two parents, so its gradient is the sum of both contributions.
	a := NewValueWithLabel(1, "a")
	b := NewValueWithLabel(-2, "b")
	c := NewValueWithLabel(3, "c")

	d := a.Mul(b)
	e := b.Mul(NewValue(2)).Add(c)
	L := d.Mul(e)

	L.Backward()

	if d.Data != -2.0 {
		t.Errorf("wrong Data for d: %f", d.Data)
	}
	if e.Data != -1.0 {
		t.Errorf("wrong Data for e: %f", e.Data)
	}
	if L.Data != 2.0 {
		t.Errorf("wrong Data for L: %f", L.Data)
	}

	// dL/dd = e = -1, dL/de = d = -2
	if d.Grad != -1.0 {
		t.Errorf("wrong grad for d: %f", d.Grad)
	}
	if e.Grad != -2.0 {
		t.Errorf("wrong grad for e: %f", e.Grad)
	}

	// dL/da = 2b^2 + bc = 2, dL/db = 4ab + ac = -5, dL/dc = ab = -2
	if a.Grad != 2.0 {
		t.Errorf("wrong grad for a: %f", a.Grad)
	}
	if b.Grad != -5.0 {
		t.Errorf("wrong grad for b: %f", b.Grad)
	}
	if c.Grad != -2.0 {
		t.Errorf("wrong grad for c: %f", c.Grad)
	}
}

func (v *Value) squaredError(other *Value) *Value {
	return v.Add(other.Negate()).Pow(2.0)
}

func Test_Linear_Regression(t *testing.T) {
	type Point struct {
		x, y float64
	}

	const (
		pointCount = 50
		epochs     = 500
		epsilon    = 1e-3
	)

	// Generate a point cloud for g(x).
	points := make([]Point, pointCount)
	for k := range points {
		x := float64(k) / float64(pointCount)
		points[k] = Point{x: x, y: 0.5*x + 2.0}
	}

	// Now "figure out" m and b.
	// y = m*x + b
	m := NewValueWithLabel(0.0, "m")
	b := NewValueWithLabel(1.0, "b")

	totalLosses := make([]float64, epochs) // We track them all.
	for k := range epochs {
		for _, point := range points {
			x := NewValueWithLabel(point.x, "x")
			y := NewValueWithLabel(point.y, "y")

			// Forward pass.
			yPredicted := m.Mul(x).Add(b)

			// Compute loss.
			loss := yPredicted.squaredError(y)
			totalLosses[k] += loss.Data

			// Reset gradients (ZeroGrad).
			m.Grad = 0
			b.Grad = 0
			loss.Backward()

			// Gradient descent.
			m.Data -= epsilon * m.Grad
			b.Data -= epsilon * b.Grad
		}
		// Loss should be monotonically decrease in this case (convex problem, no noise).
		if k > 0 && totalLosses[k] >= totalLosses[k-1] {
			t.Errorf("loss should be monotonically dropping but L[%d]=%f >= L[%d]=%f",
				k, totalLosses[k], k-1, totalLosses[k-1])
		}
	}
	if totalLosses[epochs-1] > 1e-6 {
		t.Errorf("loss larger than expected: %f", totalLosses[epochs-1])
	}
	// Check that the model learned the right values.
	approx(t, "m", m.Data, 0.5, 1e-3)
	approx(t, "b", b.Data, 2.0, 1e-3)
}
