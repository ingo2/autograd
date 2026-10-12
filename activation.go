package autograd

import (
	"encoding/json"
	"fmt"
	"math"
)

type ActivationFunc struct {
	Name string
	fn   func(*Value) *Value
	// rawFn computes the same function directly on float64, bypassing the autograd graph. Used by
	// Call when noGrad is true.
	rawFn func(float64) float64
}

// Call applies the activation to x. If noGrad is true, the result is computed directly on x.Data
// via rawFn and is not connected to the autograd graph (no children, no backwardFunc).
func (a ActivationFunc) Call(x *Value, noGrad bool) *Value {
	if noGrad {
		return NewValue(a.rawFn(x.Data))
	}
	return a.fn(x)
}
func (a ActivationFunc) String() string { return a.Name }

// MarshalJSON encodes the activation function as its name.
func (a ActivationFunc) MarshalJSON() ([]byte, error) {
	return json.Marshal(a.Name)
}

// UnmarshalJSON decodes an activation function from its name.
func (a *ActivationFunc) UnmarshalJSON(data []byte) error {
	var name string
	if err := json.Unmarshal(data, &name); err != nil {
		return err
	}
	act, ok := activationsByName[name]
	if !ok {
		return fmt.Errorf("unknown activation %q", name)
	}
	*a = act
	return nil
}

var (
	Linear = ActivationFunc{"linear",
		func(x *Value) *Value { return x },
		func(x float64) float64 { return x }}
	ReLU = ActivationFunc{"relu",
		func(x *Value) *Value { return x.ReLU() },
		func(x float64) float64 { return math.Max(0, x) }}
	Tanh = ActivationFunc{"tanh",
		func(x *Value) *Value { return x.Tanh() }, math.Tanh}
)
