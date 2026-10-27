package scoremodel

import (
	"autograd/nn"
	"autograd/tictactoe"
	"fmt"
	"math/rand"
	"testing"
)

const playerMatchupTestSeed = 42

// randomPlayer plays a uniformly random legal move. It stands in for a weak opponent, so the
// matchup test can check that the model converts wins rather than only that it avoids losses.
type randomPlayer struct {
	rand *rand.Rand
}

func (randomPlayer) Identity() string { return "random" }
func (randomPlayer) Name() string     { return "Random" }
func (p randomPlayer) Move(g tictactoe.Game) (int, int, error) {
	moves := tictactoe.AvailableMoves(g.Board)
	if len(moves) == 0 {
		return 0, 0, fmt.Errorf("no moves available")
	}
	pos := moves[p.rand.Intn(len(moves))]
	return pos % g.Size, pos / g.Size, nil
}

// TestScoreModelVsRandom pits the ScoreModel MLP against a random opponent over several games and
// asserts that ScoreModel wins more games than it loses.
func TestScoreModelVsRandom(t *testing.T) {
	model, err := nn.NewMLPFromFile(
		"../models/scoremodel_3x3_hinge_18-32-32-9.json")
	if err != nil {
		t.Fatalf("model not available: %v", err)
	}

	opponent := randomPlayer{rand: rand.New(rand.NewSource(playerMatchupTestSeed))}

	const numGames = 250
	ps := EvaluateMatchUp(model, opponent, numGames)

	t.Logf("ScoreModel vs Random (%d games): wins=%d draws=%d losses=%d",
		numGames, ps.Wins, ps.Draws, ps.Losses)
	if ps.Wins <= ps.Losses {
		t.Errorf("expected ScoreModel to win more games than it loses: wins=%d losses=%d",
			ps.Wins, ps.Losses)
	}
}

// TestScoreModelVsPerfect pits the ScoreModel MLP against unlimited minimax.
// Perfect minimax never loses, so ScoreModel cannot win; the test asserts it at least draws more
// often than it loses.
func TestScoreModelVsPerfect(t *testing.T) {
	model, err := nn.NewMLPFromFile(
		"../models/scoremodel_3x3_hinge_18-32-32-9.json")
	if err != nil {
		t.Fatalf("model not available: %v", err)
	}

	perfect := tictactoe.MinimaxPlayer{
		Rand:          rand.New(rand.NewSource(playerMatchupTestSeed)),
		MinimaxConfig: tictactoe.DefaultMinimaxConfig()}

	const numGames = 100
	ps := EvaluateMatchUp(model, perfect, numGames)

	t.Logf("ScoreModel vs Perfect (%d games): wins=%d draws=%d losses=%d",
		numGames, ps.Wins, ps.Draws, ps.Losses)
	if ps.Wins > 0 {
		t.Errorf("ScoreModel should not be able to beat perfect minimax: wins=%d", ps.Wins)
	}
	if ps.Draws <= ps.Losses {
		t.Errorf("expected ScoreModel to draw more than it loses: draws=%d losses=%d",
			ps.Draws, ps.Losses)
	}
	if ps.Draws < 85 {
		t.Errorf("expected ScoreModel to draw each time: draws=%d losses=%d",
			ps.Draws, ps.Losses)
	}
}
