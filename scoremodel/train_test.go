package scoremodel

import (
	"autograd/nn"
	"autograd/tictactoe"
	"bytes"
	"math"
	"math/rand"
	"testing"
)

func TestTrainScoreModel_FullCycle(t *testing.T) {
	const seed = 99

	// Generates and sanitizes test and training data.
	cfg := tictactoe.MinimaxConfig{
		MaxDepth:   0,
		NodeBudget: 0,
		Gamma:      0.9,
	}
	boards := tictactoe.DeduplicateBoards(tictactoe.FilterBoards(tictactoe.Enumerate3x3Boards()))
	annotatedBoards := make([]tictactoe.AnnotatedBoard, len(boards))
	for i, b := range boards {
		annotatedBoards[i] = tictactoe.AnnotateCanonicalBoard(b, tictactoe.X, cfg)
	}
	train, test, _, err := tictactoe.Split(
		tictactoe.Shuffle(annotatedBoards, rand.New(rand.NewSource(seed))), 0.1, 0.0)
	if err != nil {
		t.Fatalf("setting up samples failed: %v", err)
	}

	// Training loop.
	params := DefaultTrainParams()
	params.MaxEpochs = 5 // cap as our autograd engine is very slow.
	model := NewScoreModel(rand.New(rand.NewSource(seed)), []int{32}, nn.ReLU, nn.Tanh)

	stats, err := TrainScoreModel(model, params, train, nil, rand.New(rand.NewSource(0)))
	if err != nil {
		t.Fatalf("TrainScoreModel failed: %v", err)
	}
	t.Logf("training stats: loss(total): %g, epochs: %d",
		stats.TotalLoss[len(stats.TotalLoss)-1],
		stats.Epochs)

	if stats.Epochs != params.MaxEpochs {
		t.Errorf("wrong number of epochs: expected %d, got %d", params.MaxEpochs, stats.Epochs)
	}

	for i := int64(1); i < params.MaxEpochs; i++ {
		if stats.TotalLoss[i] > stats.TotalLoss[i-1] {
			t.Errorf("total loss expected to decrease, but %g (%d) > %g (%d)",
				stats.TotalLoss[i], i, stats.TotalLoss[i-1], i-1)
		}
	}

	eval := EvaluateStandardStats(model, test)
	t.Logf("eval: mse=%g (min=%g max=%g)",
		eval.MSE, eval.MSEMin, eval.MSEMax)
	if eval.MSE > 0.3 {
		t.Errorf("test MSE unexpectedly large: %g", eval.MSE)
	}
}

func TestTrainScoreModel_KeepsLastModel(t *testing.T) {
	const seed = 99

	cfg := tictactoe.MinimaxConfig{
		MaxDepth:   0,
		NodeBudget: 0,
		Gamma:      0.9,
	}
	boards := tictactoe.DeduplicateBoards(tictactoe.FilterBoards(tictactoe.Enumerate3x3Boards()))
	annotatedBoards := make([]tictactoe.AnnotatedBoard, len(boards))
	for i, b := range boards {
		annotatedBoards[i] = tictactoe.AnnotateCanonicalBoard(b, tictactoe.X, cfg)
	}
	train, _, _, err := tictactoe.Split(
		tictactoe.Shuffle(annotatedBoards, rand.New(rand.NewSource(seed))), 0.1, 0.0)
	if err != nil {
		t.Fatalf("setting up samples failed: %v", err)
	}

	params := DefaultTrainParams()
	// A large learning rate makes the loss worsen after the first epoch, so the epoch with the
	// lowest loss (epoch 1) is not the last one.
	params.LearningRate = 0.5
	params.MaxEpochs = 3

	// Run A: TrainScoreModel without matchups.
	modelA := NewScoreModel(rand.New(rand.NewSource(seed)), []int{32}, nn.ReLU, nn.Tanh)
	statsA, err := TrainScoreModel(modelA, params, train, nil, rand.New(rand.NewSource(0)))
	if err != nil {
		t.Fatalf("TrainScoreModel failed: %v", err)
	}
	if statsA.Epochs != params.MaxEpochs {
		t.Fatalf("Epochs = %d, want %d", statsA.Epochs, params.MaxEpochs)
	}
	if statsA.TotalLoss[1] <= statsA.TotalLoss[0] {
		t.Fatalf("expected loss to worsen after epoch 1, got epoch1=%g, epoch2=%g",
			statsA.TotalLoss[0], statsA.TotalLoss[1])
	}

	// Run B: the same epochs run directly, from the same initial weights and with the rng consumed
	// identically. The returned model must hold the weights after the last epoch, not those of the
	// epoch with the lowest loss.
	modelB := NewScoreModel(rand.New(rand.NewSource(seed)), []int{32}, nn.ReLU, nn.Tanh)
	inputs := BoardsToScoreModelInputs(train)
	outputs := BoardsToScoreModelOutputs(train)
	rng := rand.New(rand.NewSource(0))
	for range params.MaxEpochs {
		trainEpoch(modelB, params, train, inputs, outputs, rng)
	}

	if !sameParams(modelA, modelB) {
		t.Error("returned model differs from the model after the last epoch")
	}
}

func TestSnapshotRestoreParams(t *testing.T) {
	model := nn.NewMLP(rand.New(rand.NewSource(0)), []int{2, 3, 1},
		[]nn.ActivationFunc{nn.ReLU, nn.Tanh})

	original := snapshotParams(model)

	for _, p := range model.Params() {
		p.Data += 1.0
	}

	restoreParams(model, original)

	for i, p := range model.Params() {
		if p.Data != original[i] {
			t.Errorf("param %d: got %g, want %g", i, p.Data, original[i])
		}
	}
}

func TestTrainScoreModel_NoTrainBoards(t *testing.T) {
	const seed = 99

	params := DefaultTrainParams()
	model := NewScoreModel(rand.New(rand.NewSource(seed)), []int{8}, nn.ReLU, nn.Tanh)

	_, err := TrainScoreModel(model, params, nil, nil, rand.New(rand.NewSource(0)))
	if err == nil {
		t.Fatal("expected an error for empty training boards, got nil")
	}
}

func TestTrainEpoch(t *testing.T) {
	const seed = 99

	cfg := tictactoe.MinimaxConfig{MaxDepth: 0, NodeBudget: 0, Gamma: 0.9}
	boards := tictactoe.DeduplicateBoards(tictactoe.FilterBoards(tictactoe.Enumerate3x3Boards()))
	samples := make([]tictactoe.AnnotatedBoard, 50)
	for i := range samples {
		samples[i] = tictactoe.AnnotateCanonicalBoard(boards[i], tictactoe.X, cfg)
	}
	inputs := BoardsToScoreModelInputs(samples)
	outputs := BoardsToScoreModelOutputs(samples)

	// Zero learning rate: the weights stay unchanged, so the returned mean loss equals the loss of a
	// pure forward pass.
	params := DefaultTrainParams()
	params.LearningRate = 0.0
	model := NewScoreModel(rand.New(rand.NewSource(seed)), []int{8}, nn.ReLU, nn.Tanh)
	before := snapshotParams(model)
	got := trainEpoch(model, params, samples, inputs, outputs, rand.New(rand.NewSource(0)))
	want := evaluateLoss(model, samples, params.LossFunc)
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("zero LR: trainEpoch = %g, want %g", got, want)
	}
	for i, p := range model.Params() {
		if p.Data != before[i] {
			t.Errorf("zero LR: param %d changed from %g to %g", i, before[i], p.Data)
		}
	}

	// Positive learning rate: one epoch must reduce the loss on the training samples.
	params.LearningRate = 0.05
	lossBefore := evaluateLoss(model, samples, params.LossFunc)
	trainEpoch(model, params, samples, inputs, outputs, rand.New(rand.NewSource(0)))
	lossAfter := evaluateLoss(model, samples, params.LossFunc)
	if lossAfter >= lossBefore {
		t.Errorf("expected loss to decrease after one epoch, got %g -> %g", lossBefore, lossAfter)
	}
}

func TestLogEpoch(t *testing.T) {
	var buf bytes.Buffer
	logEpoch(&buf, 3, 0.25)
	want := "e: 3, loss(total): 0.25\n"
	if got := buf.String(); got != want {
		t.Errorf("logEpoch() = %q, want %q", got, want)
	}
}

// evaluateLoss returns the mean LossFunc loss over boards, using a forward pass only (no
// backward, no parameter updates).
func evaluateLoss(model *nn.MLP, boards []tictactoe.AnnotatedBoard, lossFunc LossFunc) float64 {
	inputs := BoardsToScoreModelInputs(boards)
	outputs := BoardsToScoreModelOutputs(boards)
	total := 0.0
	for k, ab := range boards {
		scores := model.Call(inputs[k], true)
		validMoves := tictactoe.AvailableMoves(ab.Board)
		total += lossFunc.Call(scores, outputs[k], validMoves).Data
	}
	return total / float64(len(boards))
}

// annotatedSamples returns the first n reachable 3x3 boards, annotated with perfect minimax scores.
func annotatedSamples(n int) []tictactoe.AnnotatedBoard {
	cfg := tictactoe.MinimaxConfig{MaxDepth: 0, NodeBudget: 0, Gamma: 0.9}
	boards := tictactoe.DeduplicateBoards(tictactoe.FilterBoards(tictactoe.Enumerate3x3Boards()))
	return tictactoe.AnnotateCanonicalBoards(boards[:n], tictactoe.X, cfg)
}

// sameParams reports whether a and b have identical parameter values.
func sameParams(a, b *nn.MLP) bool {
	pa, pb := a.Params(), b.Params()
	if len(pa) != len(pb) {
		return false
	}
	for i := range pa {
		if pa[i].Data != pb[i].Data {
			return false
		}
	}
	return true
}

func TestMinimaxOpponent(t *testing.T) {
	model := NewScoreModel(rand.New(rand.NewSource(99)), []int{8}, nn.ReLU, nn.Tanh)
	opponent := MinimaxOpponent(0)
	if got := opponent(nil).Identity(); got != tictactoe.IdentityAIMinimax {
		t.Errorf("Identity() = %q, want %q", got, tictactoe.IdentityAIMinimax)
	}

	// Every opponent starts from the same seed, so repeated matchups give the same result.
	first := EvaluateMatchUp(model, opponent(nil), 20)
	second := EvaluateMatchUp(model, opponent(nil), 20)
	if first != second {
		t.Errorf("repeated matchup: %+v, want %+v", second, first)
	}
}

func TestSelfPlayOpponent(t *testing.T) {
	model := NewScoreModel(rand.New(rand.NewSource(99)), []int{8}, nn.ReLU, nn.Tanh)
	opponent := SelfPlayOpponent(0)

	// The opponent plays the best model, picking randomly among its best moves.
	p, ok := opponent(model).(*ScoreModelPlayer)
	if !ok || p.model != model || p.rng == nil || p.topK != matchUpTopK ||
		p.margin != matchUpMargin {
		t.Fatal("opponent is not a sampled player for the best model")
	}

	// Every opponent starts from the same seed, so repeated matchups give the same result.
	first := EvaluateMatchUp(model, opponent(model), 20)
	second := EvaluateMatchUp(model, opponent(model), 20)
	if first != second {
		t.Errorf("repeated matchup: %+v, want %+v", second, first)
	}
}

func TestMatchUpGate(t *testing.T) {
	strong, err := nn.NewMLPFromFile("../models/scoremodel_3x3_hinge_18-32-32-9.json")
	if err != nil {
		t.Fatalf("model not available: %v", err)
	}
	// With all parameters zero, every cell scores the same, so in matchups the model picks randomly
	// among the first three free cells.
	weak := strong.Clone()
	for _, p := range weak.Params() {
		p.Data = 0
	}
	gate := &matchUpGate{opponent: MinimaxOpponent(0), games: 10, best: strong.Clone()}

	// The first matchup keeps the model, even though it scores lower than the best model.
	model := weak.Clone()
	r := gate.playMatch(model, 1.0)
	if r.matchUp.Score >= r.bestScore {
		t.Fatalf("first matchup: weak model scores %g vs best %g, want lower",
			r.matchUp.Score, r.bestScore)
	}
	if !r.first || !r.kept || r.sinceImproved != 0 || gate.bestLoss != 1.0 ||
		!sameParams(gate.best, weak) {
		t.Errorf("first matchup: first=%v kept=%v sinceImproved=%d bestLoss=%g replaced=%v, "+
			"want true, true, 0, 1, true",
			r.first, r.kept, r.sinceImproved, gate.bestLoss, sameParams(gate.best, weak))
	}

	// After that, a model that scores higher replaces the best model, even with a higher loss.
	restoreParams(model, snapshotParams(strong))
	r = gate.playMatch(model, 2.0)
	if r.first || !r.kept || r.matchUp.Score <= r.bestScore {
		t.Fatalf("stronger model: first=%v kept=%v with score %g vs best %g, "+
			"want kept with a higher score", r.first, r.kept, r.matchUp.Score, r.bestScore)
	}
	if r.sinceImproved != 0 || gate.bestLoss != 2.0 || !sameParams(gate.best, strong) {
		t.Errorf("stronger model: sinceImproved=%d bestLoss=%g replaced=%v, want 0, 2, true",
			r.sinceImproved, gate.bestLoss, sameParams(gate.best, strong))
	}

	// The same score without a lower loss is no improvement.
	r = gate.playMatch(model, 2.0)
	if r.kept || r.matchUp.Score != r.bestScore || r.sinceImproved != 1 {
		t.Errorf("tie, same loss: kept=%v score %g vs best %g sinceImproved=%d, "+
			"want reverted on equal scores, 1", r.kept, r.matchUp.Score, r.bestScore, r.sinceImproved)
	}

	// The same score with a lower loss replaces the best model, but is no improvement of the score.
	r = gate.playMatch(model, 1.5)
	if !r.kept || r.matchUp.Score != r.bestScore || r.bestLoss != 2.0 || r.sinceImproved != 2 ||
		gate.bestLoss != 1.5 {
		t.Errorf("tie, lower loss: kept=%v score %g vs best %g, loss vs best %g, sinceImproved=%d, "+
			"new bestLoss=%g, want kept on equal scores, 2, 2, 1.5",
			r.kept, r.matchUp.Score, r.bestScore, r.bestLoss, r.sinceImproved, gate.bestLoss)
	}

	// A model that scores lower is reverted to the best one, even with a lower loss.
	restoreParams(model, snapshotParams(weak))
	r = gate.playMatch(model, 0.1)
	if r.kept || r.sinceImproved != 3 || !sameParams(model, strong) {
		t.Errorf("weaker model: kept=%v sinceImproved=%d reverted=%v, want false, 3, true",
			r.kept, r.sinceImproved, sameParams(model, strong))
	}

	// A higher score resets the count again.
	restoreParams(gate.best, snapshotParams(weak))
	r = gate.playMatch(model, 1.0)
	if !r.kept || r.matchUp.Score <= r.bestScore || r.sinceImproved != 0 {
		t.Errorf("higher score: kept=%v score %g vs best %g sinceImproved=%d, "+
			"want kept with a higher score, 0", r.kept, r.matchUp.Score, r.bestScore, r.sinceImproved)
	}
}

func TestLogMatchUp(t *testing.T) {
	matchUp := MatchUpStats{NumGames: 100, Wins: 0, Draws: 80, Losses: 20, Score: 0.4}

	tests := []struct {
		name string
		r    matchUpResult
		want string // without the common prefix and the trailing newline
	}{
		{"kept as first matchup",
			matchUpResult{bestScore: 0.45, first: true, kept: true},
			"score: 0.4 < best: 0.45, kept as first matchup, noimprove: 0"},
		{"first matchup tie shows no loss",
			matchUpResult{bestScore: 0.4, loss: 0.7, first: true, kept: true},
			"score: 0.4 = best: 0.4, kept as first matchup, noimprove: 0"},
		{"kept as new best",
			matchUpResult{bestScore: 0.35, kept: true},
			"score: 0.4 > best: 0.35, kept as new best, noimprove: 0"},
		{"reverted",
			matchUpResult{bestScore: 0.45, sinceImproved: 2},
			"score: 0.4 < best: 0.45, reverted, noimprove: 2"},
		{"tie kept by lower loss",
			matchUpResult{bestScore: 0.4, loss: 0.7, bestLoss: 0.8, kept: true, sinceImproved: 1},
			"score: 0.4 = best: 0.4, loss: 0.7 < best: 0.8, kept as new best, noimprove: 1"},
		{"tie reverted by loss",
			matchUpResult{bestScore: 0.4, loss: 0.8, bestLoss: 0.8, sinceImproved: 1},
			"score: 0.4 = best: 0.4, loss: 0.8 >= best: 0.8, reverted, noimprove: 1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			tt.r.matchUp = matchUp
			logMatchUp(&buf, 3, tt.r)
			want := "e: 3, matchup: 0-80-20, " + tt.want + "\n"
			if got := buf.String(); got != want {
				t.Errorf("logMatchUp() = %q, want %q", got, want)
			}
		})
	}
}

func TestTrainScoreModel_MatchUpPatienceStops(t *testing.T) {
	params := DefaultTrainParams()
	params.MaxEpochs = 500
	params.MatchUpInterval = 2
	params.MatchUpPatience = 2
	params.MatchUpGames = 10
	params.LearningRate = 0.0 // zero LR: the model never changes, so it only ties with the best model
	// A single board makes every epoch's loss exactly equal, so no tie is broken by the loss.
	boards := annotatedSamples(1)
	model := NewScoreModel(rand.New(rand.NewSource(99)), []int{8}, nn.ReLU, nn.Tanh)

	stats, err := TrainScoreModel(model, params, boards, MinimaxOpponent(0),
		rand.New(rand.NewSource(0)))
	if err != nil {
		t.Fatalf("TrainScoreModel failed: %v", err)
	}

	// The first matchup after epoch 2 always keeps the model; the ties after epochs 4 and 6 do not.
	if stats.Epochs != 6 {
		t.Errorf("Epochs = %d, want 6", stats.Epochs)
	}
	wantKept := []bool{true, false, false}
	if len(stats.MatchUpScore) != 3 || len(stats.MatchUpBestScore) != 3 ||
		len(stats.MatchUpKept) != 3 {
		t.Fatalf("got %d matchup scores, %d best scores and %d kept flags, want 3 each",
			len(stats.MatchUpScore), len(stats.MatchUpBestScore), len(stats.MatchUpKept))
	}
	for i := range stats.MatchUpScore {
		if stats.MatchUpScore[i] != stats.MatchUpBestScore[i] {
			t.Errorf("matchup %d: score %g, want tie with best %g",
				i, stats.MatchUpScore[i], stats.MatchUpBestScore[i])
		}
		if stats.MatchUpKept[i] != wantKept[i] {
			t.Errorf("matchup %d: kept = %v, want %v", i, stats.MatchUpKept[i], wantKept[i])
		}
	}
}

func TestTrainScoreModel_MatchUpRestoresBest(t *testing.T) {
	params := DefaultTrainParams()
	params.MaxEpochs = 5
	params.MatchUpInterval = 2
	params.MatchUpGames = 20
	params.LearningRate = 0.5
	boards := annotatedSamples(200)
	model := NewScoreModel(rand.New(rand.NewSource(99)), []int{8}, nn.ReLU, nn.Tanh)

	stats, err := TrainScoreModel(model, params, boards, MinimaxOpponent(0),
		rand.New(rand.NewSource(0)))
	if err != nil {
		t.Fatalf("TrainScoreModel failed: %v", err)
	}

	// Matchups after epochs 2 and 4, and after the last epoch 5.
	if len(stats.MatchUpScore) != 3 {
		t.Fatalf("len(MatchUpScore) = %d, want 3", len(stats.MatchUpScore))
	}

	// After the last matchup, the best model is the trained one if it was kept, otherwise the
	// previous best one.
	want := stats.MatchUpBestScore[2]
	if stats.MatchUpKept[2] {
		want = stats.MatchUpScore[2]
	}
	// Replays the matchup with the players' seed, which TrainScoreModel draws first from its rng.
	seed := rand.New(rand.NewSource(0)).Int63()
	got := playMatchUp(newMatchUpPlayer(model, seed), MinimaxOpponent(0)(nil), params.MatchUpGames)
	if got.Score != want {
		t.Errorf("returned model scores %g, want %g", got.Score, want)
	}
}

func TestTrainScoreModel_MatchUpEpochs(t *testing.T) {
	tests := []struct {
		name         string
		maxEpochs    int64
		interval     int64
		wantMatchUps int
	}{
		{"every epoch", 3, 1, 3},
		{"max epochs multiple of interval", 4, 2, 2},
		{"extra matchup after last epoch", 5, 2, 3},
		{"interval equals max epochs", 3, 3, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := DefaultTrainParams()
			params.MaxEpochs = tt.maxEpochs
			params.MatchUpInterval = tt.interval
			params.MatchUpGames = 2
			model := NewScoreModel(rand.New(rand.NewSource(99)), []int{8}, nn.ReLU, nn.Tanh)

			stats, err := TrainScoreModel(model, params, annotatedSamples(20), MinimaxOpponent(0),
				rand.New(rand.NewSource(0)))
			if err != nil {
				t.Fatalf("TrainScoreModel failed: %v", err)
			}
			if len(stats.MatchUpScore) != tt.wantMatchUps {
				t.Errorf("got %d matchups, want %d", len(stats.MatchUpScore), tt.wantMatchUps)
			}
		})
	}
}

func TestTrainScoreModel_SelfPlay(t *testing.T) {
	params := DefaultTrainParams()
	params.MaxEpochs = 3
	params.MatchUpInterval = 1
	params.MatchUpGames = 10
	boards := annotatedSamples(200)
	model := NewScoreModel(rand.New(rand.NewSource(99)), []int{8}, nn.ReLU, nn.Tanh)

	stats, err := TrainScoreModel(model, params, boards, SelfPlayOpponent(0),
		rand.New(rand.NewSource(0)))
	if err != nil {
		t.Fatalf("TrainScoreModel failed: %v", err)
	}

	// The keep-or-revert logic is covered by the minimax tests, this checks self-play end to end.
	if len(stats.MatchUpScore) != 3 || len(stats.MatchUpBestScore) != 3 ||
		len(stats.MatchUpKept) != 3 {
		t.Fatalf("got %d matchup scores, %d best scores and %d kept flags, want 3 each",
			len(stats.MatchUpScore), len(stats.MatchUpBestScore), len(stats.MatchUpKept))
	}
	if !stats.MatchUpKept[0] {
		t.Error("first matchup: reverted, want kept")
	}
}

func TestTrainScoreModel_InvalidMatchUp(t *testing.T) {
	tests := []struct {
		name     string
		interval int64
		games    int
		opponent MatchUpOpponent
	}{
		{"no games", 1, 0, MinimaxOpponent(0)},
		{"no opponent", 1, 10, nil},
		{"interval exceeds max epochs", 4, 10, MinimaxOpponent(0)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params := DefaultTrainParams()
			params.MaxEpochs = 3
			params.MatchUpInterval = tt.interval
			params.MatchUpGames = tt.games
			model := NewScoreModel(rand.New(rand.NewSource(99)), []int{8}, nn.ReLU, nn.Tanh)

			_, err := TrainScoreModel(model, params, annotatedSamples(10), tt.opponent,
				rand.New(rand.NewSource(0)))
			if err == nil {
				t.Fatal("expected an error, got nil")
			}
		})
	}
}
