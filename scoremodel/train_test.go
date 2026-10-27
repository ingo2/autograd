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
	data := &TrainData{TrainBoards: train, TestBoards: test}
	model := NewScoreModel(rand.New(rand.NewSource(seed)), []int{32}, nn.ReLU, nn.Tanh)

	stats, err := TrainScoreModel(model, params, data, rand.New(rand.NewSource(0)))
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

	if stats.BestLoss > stats.TotalLoss[len(stats.TotalLoss)-1] {
		t.Errorf("BestLoss (%g) must be <= final TotalLoss (%g)",
			stats.BestLoss, stats.TotalLoss[len(stats.TotalLoss)-1])
	}

	if len(stats.TestLoss) != int(stats.Epochs) {
		t.Errorf("len(TestLoss) = %d, want %d", len(stats.TestLoss), stats.Epochs)
	}
	if len(stats.TestLossDelta) != int(stats.Epochs) {
		t.Errorf("len(TestLossDelta) = %d, want %d", len(stats.TestLossDelta), stats.Epochs)
	}
	if stats.TestLossDelta[0] != math.MaxFloat64 {
		t.Errorf("TestLossDelta[0] = %g, want %g", stats.TestLossDelta[0], math.MaxFloat64)
	}

	gotTestLoss := evaluateLoss(model, test, params.LossFunc)
	if math.Abs(gotTestLoss-stats.BestTestLoss) > 1e-9 {
		t.Errorf("returned model's test loss = %g, want BestTestLoss = %g", gotTestLoss, stats.BestTestLoss)
	}

	eval := EvaluateStandardStats(model, test)
	t.Logf("eval: mse=%g (min=%g max=%g)",
		eval.MSE, eval.MSEMin, eval.MSEMax)
	if eval.MSE > 0.3 {
		t.Errorf("test MSE unexpectedly large: %g", eval.MSE)
	}
}

func TestTrainScoreModel_PatienceStops(t *testing.T) {
	const seed = 99

	cfg := tictactoe.MinimaxConfig{
		MaxDepth: 0, NodeBudget: 0, Gamma: 0.9}
	boards := tictactoe.DeduplicateBoards(tictactoe.FilterBoards(tictactoe.Enumerate3x3Boards()))
	samples := make([]tictactoe.AnnotatedBoard, len(boards))
	for i, b := range boards {
		samples[i] = tictactoe.AnnotateCanonicalBoard(b, tictactoe.X, cfg)
	}

	const patience = int64(3)
	params := DefaultTrainParams()
	params.MaxEpochs = 500
	params.Patience = patience
	params.LearningRate = 0.0 // zero LR: weights never change, loss never improves
	data := &TrainData{TrainBoards: samples}
	model := NewScoreModel(rand.New(rand.NewSource(seed)), []int{8}, nn.ReLU, nn.Tanh)

	stats, err := TrainScoreModel(model, params, data, rand.New(rand.NewSource(0)))
	if err != nil {
		t.Fatalf("TrainScoreModel failed: %v", err)
	}

	if stats.Epochs >= params.MaxEpochs {
		t.Errorf("expected patience to stop training before MaxEpochs, but ran %d epochs", stats.Epochs)
	}
	t.Logf("patience=%d stopped at epoch %d (best loss %g)", patience, stats.Epochs, stats.BestLoss)
}

func TestTrainScoreModel_TestPatienceStops(t *testing.T) {
	const seed = 99

	cfg := tictactoe.MinimaxConfig{
		MaxDepth: 0, NodeBudget: 0, Gamma: 0.9}
	boards := tictactoe.DeduplicateBoards(tictactoe.FilterBoards(tictactoe.Enumerate3x3Boards()))
	samples := make([]tictactoe.AnnotatedBoard, len(boards))
	for i, b := range boards {
		samples[i] = tictactoe.AnnotateCanonicalBoard(b, tictactoe.X, cfg)
	}

	const testPatience = int64(3)
	params := DefaultTrainParams()
	params.MaxEpochs = 500
	params.Patience = 0 // disabled, to isolate test-patience
	params.TestPatience = testPatience
	params.LearningRate = 0.0 // zero LR: weights never change, loss never improves
	data := &TrainData{TrainBoards: samples, TestBoards: samples}
	model := NewScoreModel(rand.New(rand.NewSource(seed)), []int{8}, nn.ReLU, nn.Tanh)

	stats, err := TrainScoreModel(model, params, data, rand.New(rand.NewSource(0)))
	if err != nil {
		t.Fatalf("TrainScoreModel failed: %v", err)
	}

	if stats.Epochs >= params.MaxEpochs {
		t.Errorf("expected test-patience to stop training before MaxEpochs, but ran %d epochs", stats.Epochs)
	}
	if len(stats.TestLoss) != int(stats.Epochs) {
		t.Errorf("len(TestLoss) = %d, want %d", len(stats.TestLoss), stats.Epochs)
	}
	t.Logf("test-patience=%d stopped at epoch %d (best test loss %g)",
		testPatience, stats.Epochs, stats.BestTestLoss)
}

func TestTrainScoreModel_RestoresBestModel_NoTestBoards(t *testing.T) {
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
	// A large learning rate makes the loss worsen after the first epoch, so the best epoch (epoch 1)
	// is not the last one.
	params.LearningRate = 0.5
	data := &TrainData{TrainBoards: train}

	// Run A: train for a single epoch. Since this is the only (and therefore best) epoch, the
	// returned model is exactly the post-epoch-1 weights.
	params.MaxEpochs = 1
	modelA := NewScoreModel(rand.New(rand.NewSource(seed)), []int{32}, nn.ReLU, nn.Tanh)
	statsA, err := TrainScoreModel(modelA, params, data, rand.New(rand.NewSource(0)))
	if err != nil {
		t.Fatalf("TrainScoreModel (run A) failed: %v", err)
	}

	// Run B: train for several epochs, starting from the same initial weights and consuming the rng
	// identically. Epoch 1 should remain the best epoch, so the restored model should match modelA's
	// post-epoch-1 weights, even though training continued for more epochs.
	params.MaxEpochs = 5
	modelB := NewScoreModel(rand.New(rand.NewSource(seed)), []int{32}, nn.ReLU, nn.Tanh)
	statsB, err := TrainScoreModel(modelB, params, data, rand.New(rand.NewSource(0)))
	if err != nil {
		t.Fatalf("TrainScoreModel (run B) failed: %v", err)
	}

	if statsB.Epochs <= 1 {
		t.Fatalf("expected run B to run more than 1 epoch, got %d", statsB.Epochs)
	}
	if statsB.TotalLoss[1] <= statsB.TotalLoss[0] {
		t.Fatalf("expected loss to worsen after epoch 1, got epoch1=%g, epoch2=%g",
			statsB.TotalLoss[0], statsB.TotalLoss[1])
	}
	if statsA.BestLoss != statsB.BestLoss {
		t.Errorf("BestLoss mismatch: run A=%g, run B=%g", statsA.BestLoss, statsB.BestLoss)
	}
	if statsB.BestTestLoss != 0 {
		t.Errorf("BestTestLoss = %g without test boards, want 0", statsB.BestTestLoss)
	}

	paramsA := modelA.Params()
	paramsB := modelB.Params()
	for i := range paramsA {
		if paramsA[i].Data != paramsB[i].Data {
			t.Errorf("param %d: run A=%g, run B (restored)=%g", i, paramsA[i].Data, paramsB[i].Data)
		}
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

	_, err := TrainScoreModel(model, params, &TrainData{}, rand.New(rand.NewSource(0)))
	if err == nil {
		t.Fatal("expected an error for empty TrainBoards, got nil")
	}
}

func TestBestTracker(t *testing.T) {
	losses := []float64{0.5, 0.4, 0.4, 0.6, 0.3}
	wantImproved := []bool{true, true, false, false, true}
	wantBest := []float64{0.5, 0.4, 0.4, 0.4, 0.3}
	wantSinceBest := []int64{0, 0, 1, 2, 0}

	tracker := newBestTracker()
	for i, loss := range losses {
		if got := tracker.update(loss); got != wantImproved[i] {
			t.Errorf("step %d: update(%g) = %v, want %v", i, loss, got, wantImproved[i])
		}
		if tracker.best != wantBest[i] {
			t.Errorf("step %d: best = %g, want %g", i, tracker.best, wantBest[i])
		}
		if tracker.sinceBest != wantSinceBest[i] {
			t.Errorf("step %d: sinceBest = %d, want %d", i, tracker.sinceBest, wantSinceBest[i])
		}
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

func TestTrainStatsAppendTestLoss(t *testing.T) {
	var stats TrainStats
	for _, loss := range []float64{0.5, 0.3, 0.4} {
		stats.appendTestLoss(loss)
	}

	wantLoss := []float64{0.5, 0.3, 0.4}
	wantDelta := []float64{math.MaxFloat64, 0.2, -0.1}
	if len(stats.TestLoss) != len(wantLoss) || len(stats.TestLossDelta) != len(wantDelta) {
		t.Fatalf("got %d losses and %d deltas, want %d each",
			len(stats.TestLoss), len(stats.TestLossDelta), len(wantLoss))
	}
	for i := range wantLoss {
		if stats.TestLoss[i] != wantLoss[i] {
			t.Errorf("TestLoss[%d] = %g, want %g", i, stats.TestLoss[i], wantLoss[i])
		}
		if math.Abs(stats.TestLossDelta[i]-wantDelta[i]) > 1e-9 {
			t.Errorf("TestLossDelta[%d] = %g, want %g", i, stats.TestLossDelta[i], wantDelta[i])
		}
	}
}

func TestLogEpoch(t *testing.T) {
	trainBest := bestTracker{best: 0.25, sinceBest: 2}
	testBest := bestTracker{best: 0.5, sinceBest: 1}

	tests := []struct {
		name          string
		hasTestBoards bool
		want          string
	}{
		{"with test boards", true,
			"e: 3, loss(total): 0.25, noimprove: 2, loss(test): 0.75, noimprove(test): 1\n"},
		{"without test boards", false, "e: 3, loss(total): 0.25, noimprove: 2\n"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			logEpoch(&buf, 3, 0.25, trainBest, tt.hasTestBoards, 0.75, testBest)
			if got := buf.String(); got != tt.want {
				t.Errorf("logEpoch() = %q, want %q", got, tt.want)
			}
		})
	}
}
