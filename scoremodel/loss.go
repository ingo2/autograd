package scoremodel

import (
	"autograd/nn"
	"encoding/json"
	"fmt"
	"math"
)

// LossFunc wraps a named per-sample loss function for use in TrainingParams.
type LossFunc struct {
	Name string
	fn   func(scores []*nn.Value, targets []float64, validMoves []int) *nn.Value
}

func (l LossFunc) Call(scores []*nn.Value, targets []float64, validMoves []int) *nn.Value {
	return l.fn(scores, targets, validMoves)
}

func (l LossFunc) String() string { return l.Name }

// MarshalJSON encodes the loss function as its name.
func (l LossFunc) MarshalJSON() ([]byte, error) {
	return json.Marshal(l.Name)
}

// UnmarshalJSON decodes a loss function from its name.
func (l *LossFunc) UnmarshalJSON(data []byte) error {
	var name string
	if err := json.Unmarshal(data, &name); err != nil {
		return err
	}
	lf, ok := lossFuncsByName[name]
	if !ok {
		return fmt.Errorf("unknown loss function %q", name)
	}
	*l = lf
	return nil
}

// NewHingeLoss returns a LossFunc that uses maskedHingeLoss with the given margin.
func NewHingeLoss(margin float64) LossFunc {
	return LossFunc{"hinge",
		func(scores []*nn.Value, targets []float64, validMoves []int) *nn.Value {
			return maskedHingeLoss(scores, targets, validMoves, margin)
		}}
}

var (
	MSELoss          = LossFunc{"mse", maskedMSELoss}
	HingeLoss        = NewHingeLoss(0.1) // default margin
	CrossEntropyLoss = LossFunc{"crossentropy", maskedCrossEntropyLoss}
)

var lossFuncsByName = map[string]LossFunc{
	"mse":          MSELoss,
	"hinge":        HingeLoss,
	"crossentropy": CrossEntropyLoss,
}

func extractValid(scores []*nn.Value, targets []float64, validMoves []int) (ys []*nn.Value, ts []float64) {
	if len(validMoves) == 0 {
		return nil, nil
	}
	n := len(validMoves)
	ys = make([]*nn.Value, 0, n)
	ts = make([]float64, 0, n)
	for _, vm := range validMoves {
		ys = append(ys, scores[vm])
		ts = append(ts, targets[vm])
	}
	return ys, ts
}

// maskedMSELoss returns the mean squared error over validMoves only; occupied positions are
// excluded.
func maskedMSELoss(scores []*nn.Value, targets []float64, validMoves []int) *nn.Value {
	ys, ts := extractValid(scores, targets, validMoves)
	if ys == nil {
		return nn.NewValue(0.0)
	}
	return nn.NewValueMSE(ys, ts)
}

// maskedHingeLoss returns a pairwise ranking loss over validMoves. Valid moves are split into
// "best" (target score within eps of the maximum) and "nonBest". For each (best, nonBest) pair the
// hinge max(0, margin + scores[nonBest] - scores[best]) is summed. The loss is zero when every best
// move's predicted score exceeds every nonBest move's predicted score by at least  margin.
func maskedHingeLoss(scores []*nn.Value, targets []float64, validMoves []int, margin float64) *nn.Value {
	ys, ts := extractValid(scores, targets, validMoves)
	if ys == nil {
		return nn.NewValue(0.0)
	}

	bestScore := math.Inf(-1)
	for _, t := range ts {
		if t > bestScore {
			bestScore = t
		}
	}

	const eps = 1e-9
	n := len(validMoves)
	bestMoves := make([]int, 0, n)
	nonBestMoves := make([]int, 0, n)
	for i, t := range ts {
		if math.Abs(t-bestScore) < eps {
			bestMoves = append(bestMoves, i)
		} else {
			nonBestMoves = append(nonBestMoves, i)
		}
	}

	loss := nn.NewValue(0.0)
	for _, best := range bestMoves {
		for _, nonBest := range nonBestMoves {
			yb := ys[best]
			ynb := ys[nonBest]
			// hinge = max(0, margin - (y[b] - y[nb])) = max(0, margin + y[nb] - y[b])
			h := nn.NewValue(margin).Add((ynb)).Add(yb.Negate())
			if h.Data > 0.0 {
				loss = loss.Add(h)
			}
		}
	}
	// Normalize by the number of pairs so that boards with many valid moves do not produce a larger
	// gradient than boards with few. Without this, an early-game board (up to 7 nonBest moves) would
	// dominate the update over a late-game board (1-2 nonBest moves).
	numPairs := len(bestMoves) * len(nonBestMoves)
	if numPairs > 0 {
		loss = loss.Mul(nn.NewValue(1.0 / float64(numPairs)))
	}

	return loss
}

func maskedCrossEntropyLoss(scores []*nn.Value, targets []float64, validMoves []int) *nn.Value {
	ys, ts := extractValid(scores, targets, validMoves)
	if ys == nil {
		return nn.NewValue(0.0)
	}

	// First find maximum target score.
	tMax := math.Inf(-1)
	for _, t := range ts {
		if t > tMax {
			tMax = t
		}
	}

	// Count how often it occurs.
	const eps = 1e-9
	tMaxCount := 0
	for _, t := range ts {
		if math.Abs(t-tMax) < eps {
			tMaxCount++
		}
	}

	// Build the target probability distribution.
	ps := make([]float64, len(ts))
	for i, t := range ts {
		if math.Abs(t-tMax) < eps {
			ps[i] = 1.0 / float64(tMaxCount)
		}
	}

	return nn.NewValueCrossEntropy(ys, ps)
}
