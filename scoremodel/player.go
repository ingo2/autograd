package scoremodel

import (
	"autograd/nn"
	"autograd/tictactoe"
	"fmt"
	"math/rand"
	"path/filepath"
	"strings"
)

// ScoreModelPlayer is an AIPlayer backed by a trained score-regression MLP.
// Move selection is 1-ply: the MLP predicts a score for each cell and the valid move with the
// highest score is played. A player from newSampledScoreModelPlayer picks randomly among the best
// moves instead.
type ScoreModelPlayer struct {
	model    *nn.MLP
	identity string // unique identity string, one per trained model
	name     string // human-readable display name

	// With rng set, Move picks uniformly at random among the topK highest-scored valid moves within
	// margin of the best one; nil = always the best move.
	rng    *rand.Rand
	topK   int
	margin float64
}

// NewScoreModelPlayerFromModel wraps an already-loaded MLP as a ready-to-use player.
func NewScoreModelPlayerFromModel(model *nn.MLP) *ScoreModelPlayer {
	return &ScoreModelPlayer{
		model: model, identity: tictactoe.IdentityAIScoreModel, name: "Score MLP"}
}

// newSampledScoreModelPlayer returns a player for model that picks uniformly at random among its
// topK highest-scored valid moves whose score is within margin of the best one. A move the model
// rates clearly best is always played.
func newSampledScoreModelPlayer(model *nn.MLP, rng *rand.Rand, topK int,
	margin float64) *ScoreModelPlayer {
	p := NewScoreModelPlayerFromModel(model)
	p.rng, p.topK, p.margin = rng, topK, margin
	return p
}

// NewScoreModelPlayer loads weights from weightsPath and returns a ready-to-use player. It is
// identified by the weights file name and displays the loss function it was trained with, taken
// from the metadata embedded in the file.
func NewScoreModelPlayer(weightsPath string) (*ScoreModelPlayer, error) {
	model, err := nn.NewMLPFromFile(weightsPath)
	if err != nil {
		return nil, fmt.Errorf("loading scoremodel model: %w", err)
	}
	file := strings.TrimSuffix(filepath.Base(weightsPath), filepath.Ext(weightsPath))
	name := "Score MLP"
	if info, err := LoadModelInfo(weightsPath); err == nil && info.Config.LossFunc.Name != "" {
		name = fmt.Sprintf("%s (%s)", name, info.Config.LossFunc.Name)
	}
	return &ScoreModelPlayer{model: model, identity: "ai-" + file, name: name}, nil
}

func (p *ScoreModelPlayer) Identity() string { return p.identity }
func (p *ScoreModelPlayer) Name() string     { return p.name }

// Move encodes the board canonically, runs the MLP, and returns the valid move with the highest
// predicted score, or a random one among the best moves for a sampled player.
func (p *ScoreModelPlayer) Move(game tictactoe.Game) (int, int, error) {
	validMoves := tictactoe.AvailableMoves(game.Board)
	if len(validMoves) == 0 {
		return 0, 0, fmt.Errorf("no moves available")
	}
	inputs := GameToScoreModelInput(game)
	var pos int
	if p.rng == nil {
		pos = BestPositionFromValues(p.model, validMoves, inputs)[0]
	} else {
		top := topPositionsFromValues(p.model, validMoves, inputs, p.topK, p.margin)
		pos = top[p.rng.Intn(len(top))]
	}
	return pos % game.Size, pos / game.Size, nil
}
