package scoremodel

import (
	"autograd/nn"
	"math"
	"testing"
)

func TestMaskedHingeLoss(t *testing.T) {
	const tol = 1e-9

	v := func(f float64) *nn.Value { return nn.NewValue(f) }

	// No violation: scores[best=0]=0.8 > scores[nonBest=1]=0.5 + margin(0.1).
	// h = 0.1 + 0.5 - 0.8 = -0.2 => clamped to 0.
	loss := maskedHingeLoss(
		[]*nn.Value{v(0.8), v(0.5)},
		[]float64{1.0, 0.0},
		[]int{0, 1},
		0.1,
	)
	if loss.Data != 0.0 {
		t.Errorf("no violation: want 0, got %g", loss.Data)
	}

	// Single violation: h = 0.1 + 0.8 - 0.5 = 0.4.
	loss = maskedHingeLoss(
		[]*nn.Value{v(0.5), v(0.8)},
		[]float64{1.0, 0.0},
		[]int{0, 1},
		0.1,
	)
	if math.Abs(loss.Data-0.4) > tol {
		t.Errorf("single violation: want 0.4, got %g", loss.Data)
	}

	// Two nonBest moves: pairs (best=0,nonBest=1) and (best=0,nonBest=2).
	// h0 = 0.1 + 0.5 - 0.3 = 0.3, h1 = 0.1 + 0.6 - 0.3 = 0.4 => sum 0.7, numPairs 2 => 0.35.
	loss = maskedHingeLoss(
		[]*nn.Value{v(0.3), v(0.5), v(0.6)},
		[]float64{1.0, 0.5, 0.0},
		[]int{0, 1, 2},
		0.1,
	)
	if math.Abs(loss.Data-0.35) > tol {
		t.Errorf("two nonBest: want 0.35, got %g", loss.Data)
	}

	// All tied for best: nonBestMoves is empty, no pairs => loss 0.
	loss = maskedHingeLoss(
		[]*nn.Value{v(0.3), v(0.5)},
		[]float64{1.0, 1.0},
		[]int{0, 1},
		0.1,
	)
	if loss.Data != 0.0 {
		t.Errorf("all best: want 0, got %g", loss.Data)
	}

	// Empty validMoves => loss 0.
	loss = maskedHingeLoss(nil, nil, nil, 0.1)
	if loss.Data != 0.0 {
		t.Errorf("empty: want 0, got %g", loss.Data)
	}
}

func TestMaskedMSELoss(t *testing.T) {
	scores := []*nn.Value{
		nn.NewValue(0.5),
		nn.NewValue(1.0),
		nn.NewValue(0.0),
	}
	targets := []float64{1.0, 1.0, 0.0}

	// Only positions 0 and 1 are valid; position 2 is occupied and must be ignored.
	validMoves := []int{0, 1}
	loss := maskedMSELoss(scores, targets, validMoves)

	// MSE = ((0.5-1.0)^2 + (1.0-1.0)^2) / 2 = (0.25 + 0) / 2 = 0.125
	want := 0.125
	if loss.Data != want {
		t.Errorf("loss: got %g, want %g", loss.Data, want)
	}

	// No valid moves must return 0.
	zero := maskedMSELoss(scores, targets, nil)
	if zero.Data != 0.0 {
		t.Errorf("empty validMoves: got %g, want 0", zero.Data)
	}
}

func TestMaskedCrossEntropyLoss(t *testing.T) {
	const tol = 1e-9

	scores := []*nn.Value{
		nn.NewValue(0.0),
		nn.NewValue(0.0),
		nn.NewValue(0.5),
	}
	targets := []float64{1.0, 0.0, 0.5}

	// Only positions 0 and 1 are valid; position 2 is occupied and must be ignored.
	validMoves := []int{0, 1}
	loss := maskedCrossEntropyLoss(scores, targets, validMoves)

	// Equal scores give softmax q_0 = q_1 = 0.5. The target picks index 0 as the unique best move,
	// so H = -ln(q_0) = -ln(0.5) = ln(2).
	want := math.Log(2)
	if math.Abs(loss.Data-want) > tol {
		t.Errorf("loss: got %g, want %g", loss.Data, want)
	}
}

func TestMaskedCrossEntropyLoss_AsymmetricScores(t *testing.T) {
	const tol = 1e-9

	scores := []*nn.Value{
		nn.NewValue(2.0),
		nn.NewValue(0.0),
	}
	targets := []float64{0.0, 1.0}

	loss := maskedCrossEntropyLoss(scores, targets, []int{0, 1})

	// The target's best move is index 1, even though the model scores index 0 higher. The loss must
	// be based on the target's best move, not the model's:
	// H = -(s_1 - LogSumExp) = -(0 - (ln(1+exp(0-2)) + 2)) = ln(1+exp(-2)) + 2.
	want := math.Log(1+math.Exp(-2.0)) + 2.0
	if math.Abs(loss.Data-want) > tol {
		t.Errorf("loss: got %g, want %g", loss.Data, want)
	}
}
