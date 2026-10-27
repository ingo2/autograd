package scoremodel

import (
	"autograd/nn"
	"autograd/tictactoe"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
)

// TrainParams controls the training loop behaviour.
// Training stops at MaxEpochs or earlier when a termination criterion is met.
type TrainParams struct {
	// Hard stop criterion for the training loop to end.
	MaxEpochs int64 `json:"maxEpochs"`
	// Used to counter overfitting. Pulls each weight with a small factor towards zero; 0 = disabled.
	// Typical values are between 1e-4 and 1e-2.
	WeightDecay float64 `json:"weightDecay"`
	// Stop when best loss has not improved for this many epochs; 0 = disabled.
	Patience int64 `json:"patience"`
	// Stop when best test loss has not improved for this many epochs; 0 = disabled. Requires
	// TestBoards to be non-empty; ignored otherwise.
	TestPatience int64 `json:"testPatience"`
	// SGD step size applied to each parameter after every sample.
	LearningRate float64 `json:"learningRate"`
	// Per-sample loss function used during training.
	LossFunc LossFunc `json:"loss"`
	// Margin for HingeLoss; ignored by other loss functions.
	HingeMargin float64 `json:"hingeMargin"`
	// Hidden layer activation.
	ActivationHidden nn.ActivationFunc `json:"activationHidden"`
	// Output layer activation.
	ActivationOut nn.ActivationFunc `json:"activationOut"`
}

// TrainData holds the training and held-out evaluation samples for TrainScoreModel.
type TrainData struct {
	// Training data used to fit the model.
	TrainBoards []tictactoe.AnnotatedBoard
	// Held-out data used to monitor generalization during training.
	TestBoards []tictactoe.AnnotatedBoard
}

// TrainStats records per-epoch metrics from a completed training run.
// TestLossDelta[0] is math.MaxFloat64 (no previous epoch to diff against).
type TrainStats struct {
	// Number of epochs that were actually run.
	Epochs int64 `json:"epochs"`
	// Lowest total loss observed during training.
	BestLoss float64 `json:"bestLoss"`
	// BestTestLoss is the lowest value in TestLoss; 0 when TestLoss is empty.
	BestTestLoss float64 `json:"bestTestLoss,omitempty"`
	// Mean loss across all samples, one entry per epoch.
	TotalLoss []float64 `json:"totalLoss"`
	// TestLoss[i] is the mean LossFunc loss over TestBoards after epoch i+1 (forward pass only).
	// Empty when TestBoards is empty.
	TestLoss []float64 `json:"testLoss,omitempty"`
	// TestLossDelta[i] is TestLoss[i-1] - TestLoss[i]; TestLossDelta[0] is math.MaxFloat64.
	TestLossDelta []float64 `json:"testLossDelta,omitempty"`
}

// DefaultTrainParams returns TrainParams with defaults for 3x3 score regression.
func DefaultTrainParams() *TrainParams {
	return &TrainParams{
		MaxEpochs:        300,
		Patience:         10,
		TestPatience:     10,
		LearningRate:     0.05,
		LossFunc:         MSELoss,
		WeightDecay:      0.0,
		HingeMargin:      0.1,
		ActivationHidden: nn.ReLU,
		ActivationOut:    nn.Tanh,
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

// snapshotParams returns a copy of the current value of every parameter (weight and bias) in model.
func snapshotParams(model *nn.MLP) []float64 {
	params := model.Params()
	snapshot := make([]float64, len(params))
	for i, p := range params {
		snapshot[i] = p.Data
	}
	return snapshot
}

// restoreParams overwrites model's parameters (weights and biases) with the values in snapshot.
func restoreParams(model *nn.MLP, snapshot []float64) {
	for i, p := range model.Params() {
		p.Data = snapshot[i]
	}
}

// bestTracker tracks the lowest loss seen so far and the number of epochs since it last improved.
type bestTracker struct {
	best      float64
	sinceBest int64
}

// newBestTracker returns a bestTracker with no loss recorded yet.
func newBestTracker() bestTracker {
	return bestTracker{best: math.MaxFloat64}
}

// update records loss and reports whether it is a new best.
func (t *bestTracker) update(loss float64) bool {
	if loss < t.best {
		t.best = loss
		t.sinceBest = 0
		return true
	}
	t.sinceBest++
	return false
}

// trainEpoch runs one online-SGD pass over boards in random order and returns the mean loss.
// inputs and outputs are the precomputed model inputs and targets for boards.
func trainEpoch(model *nn.MLP, params *TrainParams, boards []tictactoe.AnnotatedBoard,
	inputs [][]*nn.Value, outputs [][]float64, rng *rand.Rand) float64 {
	totalLoss := 0.0
	// Random order each epoch prevents memorising the update sequence and cycling through the same
	// gradient path repeatedly.
	for _, k := range rng.Perm(len(boards)) {
		model.ZeroGrad()
		scores := model.Call(inputs[k], false)
		// Only valid moves have meaningful target scores; including occupied cells would train the
		// model on arbitrary 0.0 targets and waste capacity.
		validMoves := tictactoe.AvailableMoves(boards[k].Board)
		loss := params.LossFunc.Call(scores, outputs[k], validMoves)
		totalLoss += loss.Data
		loss.Backward()

		// Weight decay (L2 regularization) is applied only to weights, not biases: a bias shifts
		// the function but does not add to its complexity, so penalizing it wouldn't help
		// generalization.
		for _, w := range model.Weights() {
			w.Data -= params.LearningRate * (w.Grad + 2.0*params.WeightDecay*w.Data)
		}
		for _, b := range model.Biases() {
			b.Data -= params.LearningRate * b.Grad
		}
	}
	return totalLoss / float64(len(boards))
}

// appendTestLoss records testLoss and its delta to the previous epoch's test loss.
func (s *TrainStats) appendTestLoss(testLoss float64) {
	testDelta := math.MaxFloat64
	if len(s.TestLoss) > 0 {
		testDelta = s.TestLoss[len(s.TestLoss)-1] - testLoss
	}
	s.TestLoss = append(s.TestLoss, testLoss)
	s.TestLossDelta = append(s.TestLossDelta, testDelta)
}

// logEpoch writes a one-line progress report for epoch to w. The test loss columns are only
// written when hasTestBoards is true.
func logEpoch(w io.Writer, epoch int64, trainLoss float64, trainBest bestTracker, hasTestBoards bool,
	testLoss float64, testBest bestTracker) {
	if hasTestBoards {
		fmt.Fprintf(w, "e: %d, loss(total): %g, noimprove: %d, "+
			"loss(test): %g, noimprove(test): %d\n",
			epoch, trainLoss, trainBest.sinceBest, testLoss, testBest.sinceBest)
	} else {
		fmt.Fprintf(w, "e: %d, loss(total): %g, noimprove: %d\n",
			epoch, trainLoss, trainBest.sinceBest)
	}
}

// TrainScoreModel trains model on samples using online SGD (one weight update per sample).
// Loss is computed by params.LossFunc over valid moves only; occupied cells are masked out and do
// not contribute.
// On return, model holds the weights from the best epoch: the lowest test loss if TestBoards is
// non-empty, otherwise the lowest train loss.
func TrainScoreModel(model *nn.MLP, params *TrainParams, data *TrainData, rng *rand.Rand) (TrainStats, error) {
	if len(data.TrainBoards) == 0 {
		return TrainStats{}, fmt.Errorf("no training boards")
	}
	// When test data is provided, evaluate and record the held-out loss and its delta each epoch.
	// This feeds best-epoch tracking and the TestPatience stop criterion below.
	hasTestBoards := len(data.TestBoards) > 0

	inputs := BoardsToScoreModelInputs(data.TrainBoards)
	outputs := BoardsToScoreModelOutputs(data.TrainBoards)
	stats := TrainStats{
		TotalLoss: make([]float64, 0, params.MaxEpochs),
	}
	trainBest := newBestTracker()
	testBest := newBestTracker()
	var bestParams []float64
	for i := range params.MaxEpochs {
		totalLoss := trainEpoch(model, params, data.TrainBoards, inputs, outputs, rng)
		stats.TotalLoss = append(stats.TotalLoss, totalLoss)
		stats.Epochs = i + 1

		var testLoss float64
		if hasTestBoards {
			testLoss = evaluateLoss(model, data.TestBoards, params.LossFunc)
			stats.appendTestLoss(testLoss)
		}

		// Track the best epoch so far (by test loss if available, otherwise by train loss) and the
		// number of epochs since either last improved; these drive Patience/TestPatience below.
		if trainBest.update(totalLoss) && !hasTestBoards {
			bestParams = snapshotParams(model)
		}
		if hasTestBoards && testBest.update(testLoss) {
			bestParams = snapshotParams(model)
		}

		logEpoch(os.Stdout, i, totalLoss, trainBest, hasTestBoards, testLoss, testBest)

		if params.Patience > 0 && trainBest.sinceBest >= params.Patience {
			fmt.Printf("termination: no improvement for %d epochs (patience: %d)\n",
				trainBest.sinceBest, params.Patience)
			break
		}
		if hasTestBoards && params.TestPatience > 0 && testBest.sinceBest >= params.TestPatience {
			fmt.Printf("termination: test loss has not improved for %d epochs (test-patience: %d)\n",
				testBest.sinceBest, params.TestPatience)
			break
		}
	}

	if bestParams != nil {
		restoreParams(model, bestParams)
	}

	stats.BestLoss = trainBest.best
	if hasTestBoards {
		stats.BestTestLoss = testBest.best
	}
	return stats, nil
}
