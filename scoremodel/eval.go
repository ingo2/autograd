package scoremodel

import (
	"autograd/nn"
	"autograd/tictactoe"
	"math"
)

// StandardStats records evaluation metrics for a model, populated by EvaluateModel.
type StandardStats struct {
	Samples int     `json:"samples"` // number of boards evaluated
	MSE     float64 `json:"mse"`     // mean per-board MSE over valid moves
	MSEMin  float64 `json:"mseMin"`  // minimum per-board MSE (best case)
	MSEMax  float64 `json:"mseMax"`  // maximum per-board MSE (worst case)

	// Precision, Recall, and F1Score treat the model's argmax move as a positive prediction if
	// the minimax score for that move is >= 0 (a win or draw).
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1Score   float64 `json:"f1Score"`
}

// MatchUpStats records game outcomes from matches played against an AIPlayer opponent.
// Score uses standard game scoring: wins count 1, draws 0.5, losses 0.
type MatchUpStats struct {
	NumGames int     `json:"numGames"`
	Wins     int     `json:"wins"` // expected to be 0 against perfect minimax
	Draws    int     `json:"draws"`
	Losses   int     `json:"losses"`
	Score    float64 `json:"score"` // (wins + 0.5*draws) / numGames
}

// GameTreeStats records how often the model's chosen move preserves the optimal outcome class
// (Loss/Draw/Win) over every position of a labeled dataset: the whole game tree or a split of
// it. Populated by EvaluateGameTree. The four accuracy fields are derived from the count fields
// below.
type GameTreeStats struct {
	// OverallAcc is the fraction of boards where the model's chosen move preserves the optimal
	// outcome class: (PerfectWin+PerfectDraw+PerfectLoss) / total.
	OverallAcc float64 `json:"overallAcc"`
	// CriticalPositionAcc is the same fraction restricted to "critical" boards, where legal moves
	// span more than one outcome class (i.e. a wrong move actually matters):
	// CriticalPreserved / CriticalTotal.
	CriticalPositionAcc float64 `json:"criticalPositionAcc"`
	// NeverLosesAcc is the fraction of boards with a non-losing move available (optimal class Draw
	// or Win) on which the model's chosen move avoids a loss.
	NeverLosesAcc float64 `json:"neverLosesAcc"`
	// WinConversionAcc is the fraction of winning boards (optimal class Win) on which the model's
	// chosen move keeps the win, rather than settling for a draw or a loss.
	WinConversionAcc float64 `json:"winConversionAcc"`

	// PerfectWin, PerfectDraw, and PerfectLoss count boards where the model's chosen move achieves
	// the optimal outcome class (Win, Draw, or Loss; PerfectLoss means the position was already lost
	// no matter which move was chosen).
	PerfectWin  int `json:"perfectWin"`
	PerfectDraw int `json:"perfectDraw"`
	PerfectLoss int `json:"perfectLoss"`
	// WinToDraw and WinToLoss count boards where the optimal outcome was a Win but the model's move
	// only achieves a Draw or a Loss.
	WinToDraw int `json:"winToDraw"`
	WinToLoss int `json:"winToLoss"`
	// DrawToLoss counts boards where the optimal outcome was a Draw but the model's move loses.
	DrawToLoss int `json:"drawToLoss"`

	// CriticalTotal and CriticalPreserved are the counts behind CriticalPositionAcc: CriticalTotal
	// is the number of critical boards, and CriticalPreserved is how many of those preserve the
	// optimal outcome class.
	CriticalTotal     int `json:"criticalTotal"`
	CriticalPreserved int `json:"criticalPreserved"`
}

type outcomeClass int // ordered: Loss < Draw < Win

const (
	Loss outcomeClass = iota
	Draw
	Win
)

// scoreMapFromMoves builds a map from board position to minimax score.
func scoreMapFromMoves(scores []tictactoe.ScoredMove) map[int]float64 {
	scoreMap := make(map[int]float64, len(scores))
	for _, sm := range scores {
		scoreMap[sm.Pos] = sm.Score
	}
	return scoreMap
}

// EvaluateStandardStats scores model against samples and returns StandardStats.
// Only valid (empty) cells contribute to the MSE; occupied cells are ignored. Precision, Recall,
// and F1Score are derived from treating the model's argmax move per board as a positive
// prediction if its minimax score is >= 0 (a win or draw), negative otherwise.
func EvaluateStandardStats(model *nn.MLP, samples []tictactoe.AnnotatedBoard) StandardStats {
	if len(samples) == 0 {
		return StandardStats{}
	}
	var totalMSE float64
	mseMin := math.MaxFloat64
	mseMax := 0.0
	truePositive, falsePositive, falseNegative := 0, 0, 0
	for _, ab := range samples {
		validMoves := tictactoe.AvailableMoves(ab.Board)
		if len(validMoves) == 0 {
			continue
		}
		outputs := model.Call(BoardToScoreModelInput(&ab), true)
		scoreMap := scoreMapFromMoves(ab.Scores)

		mse := 0.0
		for _, vm := range validMoves {
			diff := outputs[vm].Data - scoreMap[vm]
			mse += diff * diff
		}
		mse /= float64(len(validMoves))
		totalMSE += mse
		if mse < mseMin {
			mseMin = mse
		}
		if mse > mseMax {
			mseMax = mse
		}

		// Get the move the model considers as best.
		modelArgmaxIndex := validMoves[0]
		modelArgmax := outputs[modelArgmaxIndex].Data
		for _, vm := range validMoves[1:] {
			if outputs[vm].Data > modelArgmax {
				modelArgmaxIndex = vm
				modelArgmax = outputs[vm].Data
			}
		}

		if modelArgmax >= 0 {
			// Model predicts "not a loss". Treating win (> 0) and draw (= 0) equally as the model will
			// basically never predict draw.
			if scoreMap[modelArgmaxIndex] < 0 {
				// Buts it's actually a loss.
				falsePositive++
			} else {
				// And it is a win/draw.
				truePositive++
			}
		} else if scoreMap[modelArgmaxIndex] >= 0 {
			// Model predicts loss, but it's actually a win/draw.
			falseNegative++
		}
		// Model predicts loss and it is a loss: true negative, not used by Precision/Recall/F1Score.
	}

	n := float64(len(samples))
	prec := precision(truePositive, falsePositive)
	rec := recall(truePositive, falseNegative)
	return StandardStats{
		Samples:   len(samples),
		MSE:       totalMSE / n,
		MSEMin:    mseMin,
		MSEMax:    mseMax,
		Precision: prec,
		Recall:    rec,
		F1Score:   f1Score(prec, rec),
	}
}

// EvaluateMatchUp measures play strength by playing numGames against opponent, alternating
// which side the model takes to avoid first-move bias. The caller is responsible for configuring
// opponent (including its Rand) before passing it in.
func EvaluateMatchUp(model *nn.MLP, opponent tictactoe.AIPlayer, numGames int) MatchUpStats {
	return playMatchUp(NewScoreModelPlayerFromModel(model), opponent, numGames)
}

// playMatchUp plays numGames between player and opponent, alternating which side player takes, and
// returns the results from player's point of view.
func playMatchUp(player, opponent tictactoe.AIPlayer, numGames int) MatchUpStats {
	if numGames <= 0 {
		return MatchUpStats{}
	}

	wins, draws, losses := 0, 0, 0
	for i := range numGames {
		playerMark := tictactoe.X
		if i%2 != 0 {
			playerMark = tictactoe.O
		}
		game := tictactoe.NewConnectNGame(3, 3)

		for game.Status == tictactoe.Running {
			var col, row int
			var err error
			if game.ActivePlayer == playerMark {
				col, row, err = player.Move(game)
			} else {
				col, row, err = opponent.Move(game)
			}
			if err != nil {
				break
			}
			game, err = tictactoe.PlaceMark(game, col, row)
			if err != nil {
				break
			}
		}

		switch game.Status {
		case tictactoe.Won:
			if game.Winner == playerMark {
				wins++
			} else {
				losses++
			}
		case tictactoe.Draw:
			draws++
		}
	}

	return MatchUpStats{
		NumGames: numGames,
		Wins:     wins,
		Draws:    draws,
		Losses:   losses,
		Score:    (float64(wins) + 0.5*float64(draws)) / float64(numGames),
	}
}

// classOf maps a perfect minimax solver score to its outcome class.
// -0.0 falls through both comparisons and is treated as Draw.
func classOf(score float64) outcomeClass {
	if score > 0 {
		return Win
	}
	if score < 0 {
		return Loss
	}
	return Draw
}

func EvaluateGameTree(model *nn.MLP, samples []tictactoe.AnnotatedBoard) GameTreeStats {
	overallTotal := 0
	stats := GameTreeStats{}

	for _, ab := range samples {
		validMoves := tictactoe.AvailableMoves(ab.Board)
		if len(validMoves) == 0 {
			continue
		}
		scoreMap := scoreMapFromMoves(ab.Scores)

		// From the perfect minimax scores: optimal class = highest class among legal moves, and
		// "critical" = the legal moves span more than one class.
		minClass := classOf(scoreMap[validMoves[0]])
		maxClass := minClass
		for _, vm := range validMoves[1:] {
			c := classOf(scoreMap[vm])
			if c < minClass {
				minClass = c
			}
			if c > maxClass {
				maxClass = c
			}
		}
		optimalClass := maxClass
		isCritical := minClass != maxClass

		// The move the model chooses: argmax over the legal outputs.
		chosen := BestPositionFromValues(model, validMoves, BoardToScoreModelInput(&ab))[0]

		// Class actually achieved = class of the chosen move's minimax score.
		achievedClass := classOf(scoreMap[chosen])
		preserved := achievedClass == optimalClass

		overallTotal++
		if isCritical {
			stats.CriticalTotal++
			if preserved {
				stats.CriticalPreserved++
			}
		}

		// achievedClass <= optimalClass always, so the cases below cover every outcome.
		switch {
		case optimalClass == Win && achievedClass == Win:
			stats.PerfectWin++
		case optimalClass == Win && achievedClass == Draw:
			stats.WinToDraw++
		case optimalClass == Win && achievedClass == Loss:
			stats.WinToLoss++
		case optimalClass == Draw && achievedClass == Draw:
			stats.PerfectDraw++
		case optimalClass == Draw && achievedClass == Loss:
			stats.DrawToLoss++
		default: // optimalClass == Loss && achievedClass == Loss
			stats.PerfectLoss++
		}
	}

	if overallTotal > 0 {
		stats.OverallAcc = float64(stats.PerfectWin+stats.PerfectDraw+stats.PerfectLoss) /
			float64(overallTotal)
	}
	if stats.CriticalTotal > 0 {
		stats.CriticalPositionAcc = float64(stats.CriticalPreserved) / float64(stats.CriticalTotal)
	}
	if neverLosesTotal := stats.PerfectWin + stats.WinToDraw + stats.WinToLoss +
		stats.PerfectDraw + stats.DrawToLoss; neverLosesTotal > 0 {
		neverLosesAchieved := stats.PerfectWin + stats.WinToDraw + stats.PerfectDraw
		stats.NeverLosesAcc = float64(neverLosesAchieved) / float64(neverLosesTotal)
	}
	if winTotal := stats.PerfectWin + stats.WinToDraw + stats.WinToLoss; winTotal > 0 {
		stats.WinConversionAcc = float64(stats.PerfectWin) / float64(winTotal)
	}
	return stats
}

// EvaluatePerfection runs EvaluateGameTree over every reachable board, scored with perfect
// minimax play, giving an exact play-strength breakdown across the whole state space.
func EvaluatePerfection(model *nn.MLP) GameTreeStats {
	boards := tictactoe.DeduplicateBoards(tictactoe.FilterBoards(tictactoe.Enumerate3x3Boards()))
	samples := tictactoe.AnnotateCanonicalBoards(
		boards, tictactoe.X, tictactoe.DefaultMinimaxConfig())
	return EvaluateGameTree(model, samples)
}
