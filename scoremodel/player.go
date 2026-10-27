package scoremodel

import (
	"autograd/nn"
	"autograd/tictactoe"
	"fmt"
	"path/filepath"
	"strings"
)

// ScoreModelPlayer is an AIPlayer backed by a trained score-regression MLP.
// Move selection is 1-ply: the MLP predicts a score for each cell and the valid move with the
// highest score is played.
type ScoreModelPlayer struct {
	model    *nn.MLP
	identity string // unique identity string, one per trained model
	name     string // human-readable display name
}

// NewScoreModelPlayerFromModel wraps an already-loaded MLP as a ready-to-use player.
func NewScoreModelPlayerFromModel(model *nn.MLP) *ScoreModelPlayer {
	return &ScoreModelPlayer{
		model: model, identity: tictactoe.IdentityAIScoreModel, name: "Score MLP"}
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
// predicted score.
func (p *ScoreModelPlayer) Move(game tictactoe.Game) (int, int, error) {
	validMoves := tictactoe.AvailableMoves(game.Board)
	if len(validMoves) == 0 {
		return 0, 0, fmt.Errorf("no moves available")
	}
	bestPos := BestPositionFromValues(p.model, validMoves, GameToScoreModelInput(game))[0]
	return bestPos % game.Size, bestPos / game.Size, nil
}
