package scoremodel

import (
	"autograd/nn"
	"autograd/tictactoe"
	"fmt"
	"io"
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
	// Play a matchup against the opponent passed to TrainScoreModel every this many epochs and after
	// the last epoch; 0 = disabled. Must not exceed MaxEpochs. The first matchup always keeps the
	// model. After that, the model is kept if it scores higher than the best model so far against
	// the same opponent, or the same with a lower train loss. Otherwise it is reverted to the best
	// model.
	MatchUpInterval int64 `json:"matchUpInterval"`
	// Stop when the matchup score has not improved for this many matchups; 0 = disabled. A model kept
	// only for its lower train loss does not count as an improvement. Requires MatchUpInterval > 0;
	// ignored otherwise.
	MatchUpPatience int64 `json:"matchUpPatience"`
	// Number of games per matchup, alternating sides.
	MatchUpGames int `json:"matchUpGames"`
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

// TrainStats records per-epoch metrics from a completed training run.
type TrainStats struct {
	// Number of epochs that were actually run.
	Epochs int64 `json:"epochs"`
	// Mean loss across all samples, one entry per epoch.
	TotalLoss []float64 `json:"totalLoss"`
	// MatchUpScore[i] is the model's score in the (i+1)-th matchup. Empty when MatchUpInterval is 0.
	MatchUpScore []float64 `json:"matchUpScore,omitempty"`
	// MatchUpBestScore[i] is the best model's score against the same opponent.
	MatchUpBestScore []float64 `json:"matchUpBestScore,omitempty"`
	// MatchUpKept[i] tells whether the model was kept or reverted. Except for the first matchup,
	// which always keeps the model. A model is kept if MatchUpScore[i] > MatchUpBestScore[i], or if
	// both are equal and its train loss is lower than the best model's.
	MatchUpKept []bool `json:"matchUpKept,omitempty"`
}

// DefaultTrainParams returns TrainParams with defaults for 3x3 score regression.
func DefaultTrainParams() *TrainParams {
	return &TrainParams{
		MaxEpochs:        300,
		MatchUpInterval:  0,
		MatchUpPatience:  0,
		MatchUpGames:     100,
		LearningRate:     0.05,
		LossFunc:         MSELoss,
		WeightDecay:      0.0,
		HingeMargin:      0.1,
		ActivationHidden: nn.ReLU,
		ActivationOut:    nn.Tanh,
	}
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

// MatchUpOpponent returns a new opponent for one matchup during training. best is the best model so
// far, which lets the model play against its own best version (see SelfPlayOpponent).
type MatchUpOpponent func(best *nn.MLP) tictactoe.AIPlayer

// MinimaxOpponent returns a MatchUpOpponent that plays perfect minimax. Each opponent gets a new
// Rand with the same seed, so every matchup is played against the same opponent.
func MinimaxOpponent(seed int64) MatchUpOpponent {
	return func(*nn.MLP) tictactoe.AIPlayer {
		return tictactoe.MinimaxPlayer{
			Rand: rand.New(rand.NewSource(seed)), MinimaxConfig: tictactoe.DefaultMinimaxConfig()}
	}
}

// In training matchups, score model players pick randomly among their matchUpTopK highest-scored
// moves within matchUpMargin of the best one. This varies the games,especially for self-play, while
// a move the model rates clearly best is still always played.
const (
	matchUpTopK   = 3
	matchUpMargin = 0.1
)

// newMatchUpPlayer returns a player for model that picks randomly among its best moves (see
// matchUpTopK), with a new Rand seeded with seed.
func newMatchUpPlayer(model *nn.MLP, seed int64) tictactoe.AIPlayer {
	return newSampledScoreModelPlayer(model, rand.New(rand.NewSource(seed)), matchUpTopK,
		matchUpMargin)
}

// SelfPlayOpponent returns a MatchUpOpponent that lets the model play against the best model so
// far. Like the model, it picks randomly among its best moves (see matchUpTopK). Each opponent gets
// a new Rand with the same seed, so every matchup is played against the same opponent.
func SelfPlayOpponent(seed int64) MatchUpOpponent {
	return func(best *nn.MLP) tictactoe.AIPlayer {
		return newMatchUpPlayer(best, seed)
	}
}

// matchUpGate holds the best model so far. After the first matchup, a model replaces it only if it
// scores higher against the opponent than the best model itself does, or the same with a lower
// train loss; otherwise the model is reverted to the best one.
type matchUpGate struct {
	opponent      MatchUpOpponent
	games         int
	seed          int64   // seed of the model's and best model's players, the same in every matchup
	best          *nn.MLP // copy of the best model so far, initially the untrained model
	bestLoss      float64 // train loss of best; unset before the first matchup
	sinceImproved int64   // matchups since the score last improved
	played        bool    // whether a matchup has been played yet
}

// matchUpResult describes one matchup played by matchUpGate.check and the decision taken on it.
type matchUpResult struct {
	matchUp       MatchUpStats // model's matchup result
	bestScore     float64      // best model's score against the same opponent
	loss          float64      // model's train loss
	bestLoss      float64      // best model's train loss, compared when the scores are equal
	first         bool         // whether this was the first matchup, which always keeps model
	kept          bool         // whether model was kept or reverted to the best one
	sinceImproved int64        // matchups since the score last improved, after the decision
}

// playMatch plays a matchup with model and one with the best model, both against the opponent. Both
// pick randomly among their best moves, with the same seed, so equal models get equal scores. loss
// is model's train loss. model replaces the best model if this is the first matchup, if it scores
// higher, or if it scores the same with a lower loss; otherwise model is reverted to the best one.
// Only the first matchup and a higher score reset sinceImproved.
func (g *matchUpGate) playMatch(model *nn.MLP, loss float64) matchUpResult {
	r := matchUpResult{
		matchUp:   playMatchUp(newMatchUpPlayer(model, g.seed), g.opponent(g.best), g.games),
		bestScore: playMatchUp(newMatchUpPlayer(g.best, g.seed), g.opponent(g.best), g.games).Score,
		loss:      loss,
		bestLoss:  g.bestLoss,
		// Keeping the first matchup unconditionally means the first interval of training is never
		// thrown away in favor of the untrained model.
		first: !g.played,
	}
	g.played = true
	improved := r.first || r.matchUp.Score > r.bestScore
	r.kept = improved || (r.matchUp.Score == r.bestScore && loss < g.bestLoss)
	if r.kept {
		restoreParams(g.best, snapshotParams(model))
		g.bestLoss = loss
	} else {
		restoreParams(model, snapshotParams(g.best))
	}
	if improved {
		g.sinceImproved = 0
	} else {
		g.sinceImproved++
	}
	r.sinceImproved = g.sinceImproved
	return r
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

// logEpoch writes a one-line progress report for epoch to w.
func logEpoch(w io.Writer, epoch int64, trainLoss float64) {
	fmt.Fprintf(w, "e: %d, loss(total): %g\n", epoch, trainLoss)
}

// logMatchUp writes a one-line report of the matchup r played after epoch to w. The scores are
// compared against the best model's, the bar the model had to beat; on a tie, the losses are shown
// as well, since they decide.
func logMatchUp(w io.Writer, epoch int64, r matchUpResult) {
	cmp := "="
	switch {
	case r.matchUp.Score > r.bestScore:
		cmp = ">"
	case r.matchUp.Score < r.bestScore:
		cmp = "<"
	}
	line := fmt.Sprintf("e: %d, matchup: %d-%d-%d, score: %g %s best: %g", epoch, r.matchUp.Wins,
		r.matchUp.Draws, r.matchUp.Losses, r.matchUp.Score, cmp, r.bestScore)
	// The first matchup has no best loss to compare to.
	if cmp == "=" && !r.first {
		lossCmp := ">="
		if r.loss < r.bestLoss {
			lossCmp = "<"
		}
		line += fmt.Sprintf(", loss: %g %s best: %g", r.loss, lossCmp, r.bestLoss)
	}
	action := "reverted"
	switch {
	case r.first:
		action = "kept as first matchup"
	case r.kept:
		action = "kept as new best"
	}
	fmt.Fprintf(w, "%s, %s, noimprove: %d\n", line, action, r.sinceImproved)
}

// TrainScoreModel trains model on samples using online SGD (one weight update per sample).
// Loss is computed by params.LossFunc over valid moves only; occupied cells are masked out and do
// not contribute.
// If params.MatchUpInterval > 0, model plays a matchup against opponent every MatchUpInterval
// epochs and after the last epoch, so no trained epoch goes unchecked. The first matchup always
// keeps model. After that, model is kept if it scores higher than the best model so far against
// the same opponent, or the same with a lower train loss. Otherwise it is reverted to the best
// model and training continues from there. In the matchups, model and the best model pick randomly
// among their best moves (see matchUpTopK).
// opponent is ignored when matchups are disabled and may be nil then. On return, model holds the
// weights of the best model if matchups are enabled, otherwise the weights after the last epoch.
func TrainScoreModel(model *nn.MLP, params *TrainParams, boards []tictactoe.AnnotatedBoard, opponent MatchUpOpponent, rng *rand.Rand) (TrainStats, error) {
	if len(boards) == 0 {
		return TrainStats{}, fmt.Errorf("no training boards")
	}
	if params.MatchUpInterval > 0 && params.MatchUpGames <= 0 {
		return TrainStats{}, fmt.Errorf("matchup interval %d requires matchup games > 0, got %d",
			params.MatchUpInterval, params.MatchUpGames)
	}
	if params.MatchUpInterval > 0 && opponent == nil {
		return TrainStats{}, fmt.Errorf("matchup interval %d requires an opponent",
			params.MatchUpInterval)
	}
	if params.MatchUpInterval > params.MaxEpochs {
		return TrainStats{}, fmt.Errorf("matchup interval %d exceeds max epochs %d",
			params.MatchUpInterval, params.MaxEpochs)
	}
	// When matchups are enabled, they select the returned model and drive the MatchUpPatience stop
	// criterion below.
	var gate *matchUpGate
	if params.MatchUpInterval > 0 {
		gate = &matchUpGate{opponent: opponent, games: params.MatchUpGames, seed: rng.Int63(),
			best: model.Clone()}
	}

	inputs := BoardsToScoreModelInputs(boards)
	outputs := BoardsToScoreModelOutputs(boards)
	stats := TrainStats{
		TotalLoss: make([]float64, 0, params.MaxEpochs),
	}
	for i := range params.MaxEpochs {
		totalLoss := trainEpoch(model, params, boards, inputs, outputs, rng)
		stats.TotalLoss = append(stats.TotalLoss, totalLoss)
		stats.Epochs = i + 1
		logEpoch(os.Stdout, i, totalLoss)

		if gate != nil && ((i+1)%params.MatchUpInterval == 0 || i+1 == params.MaxEpochs) {
			r := gate.playMatch(model, totalLoss)
			stats.MatchUpScore = append(stats.MatchUpScore, r.matchUp.Score)
			stats.MatchUpBestScore = append(stats.MatchUpBestScore, r.bestScore)
			stats.MatchUpKept = append(stats.MatchUpKept, r.kept)
			logMatchUp(os.Stdout, i, r)
			if params.MatchUpPatience > 0 && gate.sinceImproved >= params.MatchUpPatience {
				fmt.Printf("termination: matchup score has not improved for %d matchups "+
					"(matchup-patience: %d)\n", gate.sinceImproved, params.MatchUpPatience)
				break
			}
		}
	}

	// The loop always ends right after a matchup, which already leaves model at the best one. This
	// keeps the returned model correct for any future exit path.
	if gate != nil {
		restoreParams(model, snapshotParams(gate.best))
	}
	return stats, nil
}
