package scoremodel

import (
	"autograd/nn"
	"autograd/tictactoe"
	"math/rand"
	"testing"
)

func TestPlayMatchUp(t *testing.T) {
	perfect := tictactoe.MinimaxPlayer{
		Rand: rand.New(rand.NewSource(1)), MinimaxConfig: tictactoe.DefaultMinimaxConfig()}
	// All scores equal: the deterministic player always plays the first free cell.
	weak := NewScoreModelPlayerFromModel(fixedScoreMLP(0, 0, 0, 0, 0, 0, 0, 0, 0))

	// The results are from player's point of view: perfect minimax never loses, and beats the weak
	// player.
	ps := playMatchUp(perfect, weak, 10)
	if ps.NumGames != 10 || ps.Wins+ps.Draws+ps.Losses != 10 {
		t.Fatalf("got %+v, want 10 games", ps)
	}
	if ps.Losses != 0 || ps.Wins == 0 {
		t.Errorf("perfect vs weak: %d-%d-%d, want wins and no losses", ps.Wins, ps.Draws, ps.Losses)
	}
}

func TestEvaluateMatchUp(t *testing.T) {
	const seed = 99

	model := NewScoreModel(rand.New(rand.NewSource(seed)), []int{16}, nn.ReLU, nn.Tanh)

	perfect := tictactoe.MinimaxPlayer{
		Rand: rand.New(rand.NewSource(seed)), MinimaxConfig: tictactoe.DefaultMinimaxConfig()}
	ps := EvaluateMatchUp(model, perfect, 10)

	if ps.NumGames != 10 {
		t.Errorf("NumGames: want 10, got %d", ps.NumGames)
	}
	if ps.Wins+ps.Draws+ps.Losses != ps.NumGames {
		t.Errorf("wins+draws+losses=%d != NumGames=%d", ps.Wins+ps.Draws+ps.Losses, ps.NumGames)
	}
	if ps.Score < 0 || ps.Score > 1 {
		t.Errorf("Score out of [0,1]: %g", ps.Score)
	}

	zero := EvaluateMatchUp(model, perfect, 0)
	if zero.NumGames != 0 || zero.Score != 0 {
		t.Errorf("zero games: unexpected stats %+v", zero)
	}

	t.Logf("play matchup (10 games): wins=%d draws=%d losses=%d score=%g",
		ps.Wins, ps.Draws, ps.Losses, ps.Score)
}

func TestEvaluateStandardStats(t *testing.T) {
	const seed = 99

	cfg := tictactoe.MinimaxConfig{
		MaxDepth: 0, NodeBudget: 0, Gamma: 0.9}

	// Use 50 annotated boards - enough to exercise the stats without being slow.
	raw := tictactoe.DeduplicateBoards(tictactoe.FilterBoards(tictactoe.Enumerate3x3Boards()))
	raw = raw[:50]
	samples := make([]tictactoe.AnnotatedBoard, len(raw))
	for i, b := range raw {
		samples[i] = tictactoe.AnnotateCanonicalBoard(b, tictactoe.X, cfg)
	}

	model := NewScoreModel(rand.New(rand.NewSource(seed)), []int{16}, nn.ReLU, nn.Tanh)

	// Empty input must return zero-value stats.
	if got := EvaluateStandardStats(model, nil); got.Samples != 0 {
		t.Errorf("empty input: expected 0 samples, got %d", got.Samples)
	}

	eval := EvaluateStandardStats(model, samples)

	if eval.Samples != len(samples) {
		t.Errorf("Samples: want %d, got %d", len(samples), eval.Samples)
	}
	if eval.MSEMin > eval.MSE {
		t.Errorf("MSEMin (%g) > MSE (%g)", eval.MSEMin, eval.MSE)
	}
	if eval.MSEMax < eval.MSE {
		t.Errorf("MSEMax (%g) < MSE (%g)", eval.MSEMax, eval.MSE)
	}

	for name, val := range map[string]float64{
		"Precision": eval.Precision,
		"Recall":    eval.Recall,
		"F1Score":   eval.F1Score,
	} {
		if val < 0 || val > 1 {
			t.Errorf("%s out of [0,1]: %g", name, val)
		}
	}

	t.Logf("eval: mse=%g precision=%g recall=%g f1=%g", eval.MSE, eval.Precision, eval.Recall, eval.F1Score)
}

func TestEvaluateGameTree(t *testing.T) {
	const seed = 99

	cfg := tictactoe.MinimaxConfig{
		MaxDepth: 0, NodeBudget: 0, Gamma: 0.9}

	raw := tictactoe.DeduplicateBoards(tictactoe.FilterBoards(tictactoe.Enumerate3x3Boards()))
	raw = raw[:50]
	samples := make([]tictactoe.AnnotatedBoard, len(raw))
	for i, b := range raw {
		samples[i] = tictactoe.AnnotateCanonicalBoard(b, tictactoe.X, cfg)
	}

	model := NewScoreModel(rand.New(rand.NewSource(seed)), []int{16}, nn.ReLU, nn.Tanh)

	// Empty input must return zero-value stats.
	if got := EvaluateGameTree(model, nil); got.OverallAcc != 0 || got.CriticalPositionAcc != 0 {
		t.Errorf("empty input: expected zero stats, got %+v", got)
	}

	ps := EvaluateGameTree(model, samples)

	for name, val := range map[string]float64{
		"OverallAcc":          ps.OverallAcc,
		"CriticalPositionAcc": ps.CriticalPositionAcc,
		"NeverLosesAcc":       ps.NeverLosesAcc,
		"WinConversionAcc":    ps.WinConversionAcc,
	} {
		if val < 0 || val > 1 {
			t.Errorf("%s out of [0,1]: %g", name, val)
		}
	}

	t.Logf("play strength: overallAcc=%g criticalPositionAcc=%g", ps.OverallAcc, ps.CriticalPositionAcc)
}

// newPerfectModel builds an MLP that implements a lookup table over samples, reproducing each
// board's exact minimax scores so that EvaluateGameTree sees a "perfect" player.
//
// The hidden layer has one ReLU neuron per sample and acts as the table's row selector: weight +1
// for set bits, -1 for unset bits, bias 1-K (K = number of set bits). For the matching input this
// sums to exactly 1; for any other distinct one-hot board encoding it sums to <= 0 and is clipped
// to 0 by ReLU, so at most one row is ever selected. The linear output layer holds the table's rows
// as per-sample weight vectors (one row of minimax scores per sample); since the hidden layer is
// one-hot, the weighted sum simply reads out the selected row.
func newPerfectModel(samples []tictactoe.AnnotatedBoard) *nn.MLP {
	n := len(samples)
	sizes := []int{inputLayerSize, n, outputLayerSize}
	activations := []nn.ActivationFunc{nn.ReLU, nn.Linear}
	model := nn.NewMLP(rand.New(rand.NewSource(1)), sizes, activations)

	hidden := model.Layers[0].Neurons
	for i, ab := range samples {
		input := BoardToScoreModelInput(&ab)
		k := 0.0
		for j, v := range input {
			if v.Data > 0 {
				hidden[i].Weights[j].Data = 1
				k++
			} else {
				hidden[i].Weights[j].Data = -1
			}
		}
		hidden[i].Bias.Data = 1 - k
	}

	output := model.Layers[1].Neurons
	for pos, neuron := range output {
		neuron.Bias.Data = 0
		for i, ab := range samples {
			scoreMap := scoreMapFromMoves(ab.Scores)
			neuron.Weights[i].Data = scoreMap[pos]
		}
	}

	return model
}

func TestEvaluateGameTreePerfect(t *testing.T) {
	cfg := tictactoe.MinimaxConfig{
		MaxDepth: 0, NodeBudget: 0, Gamma: 0.9}

	// The empty board (samples[0]) is critical: from an empty 3x3 board, edge-cell first moves
	// lose against perfect play while corner/center moves draw, so legal moves span both the Loss
	// and Draw classes.
	raw := tictactoe.DeduplicateBoards(tictactoe.FilterBoards(tictactoe.Enumerate3x3Boards()))
	raw = raw[:50]
	samples := make([]tictactoe.AnnotatedBoard, len(raw))
	for i, b := range raw {
		samples[i] = tictactoe.AnnotateCanonicalBoard(b, tictactoe.X, cfg)
	}

	model := newPerfectModel(samples)

	ps := EvaluateGameTree(model, samples)
	if ps.OverallAcc != 1 {
		t.Errorf("OverallAcc: want 1, got %g", ps.OverallAcc)
	}
	if ps.CriticalPositionAcc != 1 {
		t.Errorf("CriticalPositionAcc: want 1, got %g", ps.CriticalPositionAcc)
	}
	if perfect := ps.PerfectWin + ps.PerfectDraw + ps.PerfectLoss; perfect != len(samples) {
		t.Errorf("PerfectWin+PerfectDraw+PerfectLoss: want %d, got %d", len(samples), perfect)
	}
	if ps.WinToDraw != 0 || ps.WinToLoss != 0 || ps.DrawToLoss != 0 {
		t.Errorf("expected no imperfect moves, got %+v", ps)
	}
	if ps.NeverLosesAcc != 1 {
		t.Errorf("NeverLosesAcc: want 1, got %g", ps.NeverLosesAcc)
	}
	if ps.WinConversionAcc != 1 {
		t.Errorf("WinConversionAcc: want 1, got %g", ps.WinConversionAcc)
	}
}
