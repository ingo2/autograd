package nn

import (
	"fmt"
	"math"
	"math/rand"
)

// An autograd engine. This is a micrograd-clone.
// See https://github.com/karpathy/micrograd/tree/master for the original.
type Value struct {
	Data         float64
	Grad         float64
	children     []*Value
	backwardFunc func()
	op           string
	label        string
}

func NewValue(data float64) *Value {
	return &Value{
		Data:         data,
		Grad:         0.0,
		children:     nil,
		backwardFunc: func() {},
		op:           "",
		label:        "",
	}
}

func NewValueWithLabel(data float64, label string) *Value {
	return &Value{
		Data:         data,
		Grad:         0.0,
		children:     nil,
		backwardFunc: func() {},
		op:           "",
		label:        label,
	}
}

func NewRandomValue(rng *rand.Rand, min, max float64) *Value {
	return NewValue(min + rng.Float64()*(max-min))
}

func (v *Value) Add(other *Value) *Value {
	newValue := NewValue(v.Data + other.Data)
	newValue.op = "+"
	newValue.children = []*Value{v, other}
	newValue.backwardFunc = func() {
		// For M = P + Q, by the chain rule:
		//   P.Grad = dL/dP = dL/dM * dM/dP = M.Grad * 1
		//   Q.Grad = dL/dQ = dL/dM * dM/dQ = M.Grad * 1
		v.Grad += newValue.Grad
		other.Grad += newValue.Grad
	}
	return newValue
}

func (v *Value) Mul(other *Value) *Value {
	newValue := NewValue(v.Data * other.Data)
	newValue.op = "*"
	newValue.children = []*Value{v, other}
	newValue.backwardFunc = func() {
		// For M = P * Q, by the chain rule:
		//   P.Grad = dL/dP = dL/dM * dM/dP = M.Grad * Q
		//   Q.Grad = dL/dQ = dL/dM * dM/dQ = M.Grad * P
		v.Grad += newValue.Grad * other.Data
		other.Grad += newValue.Grad * v.Data
	}
	return newValue
}

func (v *Value) Negate() *Value {
	return v.Mul(NewValue(-1.0))
}

func (v *Value) Reciprocal() *Value {
	return v.Pow(-1.0)
}

func (v *Value) Pow(exp float64) *Value {
	newValue := NewValue(math.Pow(v.Data, exp))
	newValue.op = fmt.Sprintf("^{%f}", exp)
	newValue.children = []*Value{v}
	newValue.backwardFunc = func() {
		// For M = P^e, by the power rule:
		//   P.Grad = dL/dP = dL/dM * dM/dP = M.Grad * (e * P^(e-1))
		v.Grad += exp * math.Pow(v.Data, exp-1.0) * newValue.Grad
	}
	return newValue
}

func (v *Value) Exp() *Value {
	newValue := NewValue(math.Exp(v.Data))
	newValue.op = "exp"
	newValue.children = []*Value{v}
	newValue.backwardFunc = func() {
		// For M = e^P, by the chain rule:
		//   P.Grad = dL/dP = dL/dM * dM/dP = M.Grad * e^P = M.Grad * M.Data
		v.Grad += newValue.Data * newValue.Grad
	}
	return newValue
}

func (v *Value) Log() *Value {
	// Clip before log trick.
	const eps = 1e-12
	x := math.Max(v.Data, eps)
	newValue := NewValue(math.Log(x))
	newValue.op = "ln"
	newValue.children = []*Value{v}
	newValue.backwardFunc = func() {
		// For M := ln(P):
		//   P.Grad = dL/dP = dL/dM * dM/dP = M.Grad * 1/P (P clamped to x)
		v.Grad += (1.0 / x) * newValue.Grad
	}
	return newValue
}

func (v *Value) ReLU() *Value {
	newValue := NewValue(math.Max(0, v.Data))
	newValue.op = "relu"
	newValue.children = []*Value{v}
	newValue.backwardFunc = func() {
		// For M = max(0, P), the derivative is 1 if P > 0, else 0.
		//   P.Grad = dL/dP = M.Grad * (1 if P > 0 else 0)
		if v.Data > 0 {
			v.Grad += newValue.Grad
		}
	}
	return newValue
}

func (v *Value) Tanh() *Value {
	newValue := NewValue(math.Tanh(v.Data))
	newValue.op = "tanh"
	newValue.children = []*Value{v}
	newValue.backwardFunc = func() {
		// For M = tanh(P), the derivative is 1 - tanh(P)^2 = 1 - M^2:
		//   P.Grad = dL/dP = dL/dM * dM/dP = M.Grad * (1 - M.Data^2)
		t := newValue.Data
		v.Grad += (1.0 - t*t) * newValue.Grad
	}
	return newValue
}

// NewValueMSE calculates the mean squared error between ys and targets.
// Both slices must have the same non-zero length; otherwise this panics (mismatched lengths) or
// produces NaN (empty input).
func NewValueMSE(ys []*Value, ts []float64) *Value {
	scale := 1.0 / float64(len(ys))
	mse := 0.0
	for i, y := range ys {
		diff := y.Data - ts[i]
		mse += diff * diff
	}
	mse *= scale

	newValue := NewValue(mse)
	newValue.op = "mse"
	newValue.children = make([]*Value, len(ys))
	copy(newValue.children, ys)

	newValue.backwardFunc = func() {
		// Partial derivatives wrt yi gives d(mse)/dyi = 2/n * (yi - ti).
		for i, c := range newValue.children {
			c.Grad += 2.0 * scale * (c.Data - ts[i]) * newValue.Grad
		}
	}

	return newValue
}

// NewValueCrossEntropy calculates the cross entropy between the model's raw  scores (logits) and a
// target probability distribution probDistrib. It is a fused softmax + cross-entropy node: it
// expects unnormalized logits, not probabilities.
// Both slices must have the same non-zero length; otherwise this panics (mismatched lengths) or
// produces NaN (empty input).
func NewValueCrossEntropy(logits []*Value, probDistrib []float64) *Value {
	// [1] Cross entropy H(t, y) = -sum_{k=1..K} t_k * (y_k - LSE(y))
	// [2] LSE(y) = ln( sum_{k=1..K} exp(y_k) )
	//
	// For numerical stability reasons we want to avoid exp(x > 0), so we pull
	// out the max score M = max_{k=1..K} y_k:
	// => LSE = ln( sum_{k=1..K} exp(y_k) )
	//        = ln( sum_{k=1..K} exp(y_k - M + M) )
	//        = ln( sum_{k=1..K} exp(y_k - M) * exp(M) )
	//        = ln( sum_{k=1..K} exp(y_k - M) ) + ln(exp(M))
	//        = ln( sum_{k=1..K} exp(y_k - M) ) + M
	//
	// First calculate the LogSumExp alias LSE(y).
	maxLogit := math.Inf(-1)
	for _, y := range logits {
		if y.Data > maxLogit {
			maxLogit = y.Data
		}
	}

	expLogits := make([]float64, len(logits))
	sumExp := 0.0
	for i, y := range logits {
		e := math.Exp(y.Data - maxLogit)
		expLogits[i] = e
		sumExp += e
	}
	logSumExp := math.Log(sumExp) + maxLogit

	// Next is Cross Entropy, see [1]. Terms with t_k = 0 contribute nothing and are skipped
	// (micro-optimization, not a stability requirement: y_k - LSE is always finite).
	ce := 0.0
	for i, p := range probDistrib {
		if p > 0.0 {
			ce -= p * (logits[i].Data - logSumExp)
		}
	}

	newValue := NewValue(ce)
	newValue.op = "ce"
	newValue.children = make([]*Value, len(logits))
	copy(newValue.children, logits)

	newValue.backwardFunc = func() {
		// Derivation of d{H}/d{y_k}:
		//
		// Split [1] and use that LSE(y) does not depend on the summation index and that t is a
		// distribution meaning sum_k t_k = 1:
		//
		//   H = -sum_k t_k * y_k  +  LSE(y) * sum_k t_k
		//     = -sum_k t_k * y_k  +  LSE(y) * 1
		//
		// First term: only the k-th summand depends on y_k:
		//
		//   d/d{y_k} [ -sum_j t_j * y_j ] = -t_k
		//
		// Second term: with S = sum_j exp(y_j) the chain rule gives
		//
		//   d{LSE}/d{y_k} = d{ln(S)}/d{y_k} = (1/S) * exp(y_k) = softmax(y)_k
		//
		// The max trick does not change this: expanding the stabilized quotient
		// exp(y_k - M) / sum_j exp(y_j - M) by exp(M) yields the same expression, so we can reuse
		// expLogits and sumExp from the forward pass directly.
		//
		// Together:
		//
		//   d{H}/d{y_k} = softmax(y)_k - t_k
		//
		// The chain rule with the upstream gradient newValue.Grad then gives the contribution
		// accumulated into each child.
		for i, c := range newValue.children {
			softmax := expLogits[i] / sumExp
			c.Grad += (softmax - probDistrib[i]) * newValue.Grad
		}
	}

	return newValue
}

// Backward runs backpropagation from this node through the entire graph.
//
// All backward functions accumulate gradients with += instead of =. This handles nodes that are
// used more than once in the graph. If C feeds into both A and B, its total gradient is the sum of
// contributions from both:
//
//	     A = f(C)
//	    /        \
//	   C          ... L
//	    \        /
//	     B = g(C)
//
//	dL/dC = dL/dA * dA/dC + dL/dB * dB/dC
//
// The same applies when a node is used twice as input of one operation: for M = a.Mul(a), both
// lines in Mul's backward function add to a.Grad, giving dM/da = 2a. With = the second line would
// overwrite the first and give a.
//
// As a consequence, gradients must be reset to 0 before each call. Nodes created during the forward
// pass are new and start at 0, but reused nodes like model parameters must be reset explicitly,
// e.g. via ZeroGrad.
func (v *Value) Backward() {
	topo := topologicalSort(v)
	v.Grad = 1.0
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].backwardFunc()
	}
}

func (v *Value) ToString() string {
	return fmt.Sprintf("Value: d=%f, g=%f", v.Data, v.Grad)
}

func topologicalSort(v *Value) []*Value {
	visited := map[*Value]struct{}{}
	topoOrder := []*Value{}

	var dfs func(*Value)
	dfs = func(v *Value) {
		if _, exists := visited[v]; exists {
			return
		}
		visited[v] = struct{}{}
		for _, c := range v.children {
			dfs(c)
		}
		topoOrder = append(topoOrder, v)
	}

	dfs(v)
	return topoOrder
}
