package scoremodel

import (
	"autograd/nn"
	"autograd/tictactoe"
	"math/rand"
	"testing"
)

// makeGame returns a minimal Game with the given board and active player.
func makeGame(board tictactoe.Board, active tictactoe.Player) tictactoe.Game {
	return tictactoe.Game{Board: board, Size: 3, ActivePlayer: active}
}

// TestNewScoreModelPlayerFromModel verifies that the constructor produces a functional player.
func TestNewScoreModelPlayerFromModel(t *testing.T) {
	model := NewScoreModel(rand.New(rand.NewSource(1)), []int{16}, nn.ReLU, nn.Tanh)
	p := NewScoreModelPlayerFromModel(model)
	if p == nil {
		t.Fatal("expected non-nil player")
	}
	if p.Identity() != tictactoe.IdentityAIScoreModel {
		t.Errorf("Identity: got %q, want %q", p.Identity(), tictactoe.IdentityAIScoreModel)
	}
	board := make(tictactoe.Board, 9)
	_, _, err := p.Move(makeGame(board, tictactoe.X))
	if err != nil {
		t.Errorf("Move on empty board: %v", err)
	}
}

// TestNewScoreModelPlayer checks that a player loaded from a weights file is identified by the
// file name and displays the loss it was trained with.
func TestNewScoreModelPlayer(t *testing.T) {
	p, err := NewScoreModelPlayer("../models/scoremodel_3x3_hinge_18-32-32-9.json")
	if err != nil {
		t.Fatalf("model not available: %v", err)
	}
	if want := "ai-scoremodel_3x3_hinge_18-32-32-9"; p.Identity() != want {
		t.Errorf("Identity: got %q, want %q", p.Identity(), want)
	}
	if want := "Score MLP (hinge)"; p.Name() != want {
		t.Errorf("Name: got %q, want %q", p.Name(), want)
	}
}

// fixedScoreMLP returns a model that predicts the given scores for the 9 cells on every board.
func fixedScoreMLP(scores ...float64) *nn.MLP {
	model := nn.NewMLP(rand.New(rand.NewSource(0)), []int{18, 9}, []nn.ActivationFunc{nn.Linear})
	for i, n := range model.Layers[0].Neurons {
		for _, w := range n.Weights {
			w.Data = 0
		}
		n.Bias.Data = scores[i]
	}
	return model
}

func TestSampledScoreModelPlayer(t *testing.T) {
	game := makeGame(make(tictactoe.Board, 9), tictactoe.X)
	// playedCells plays 200 moves on the empty board and returns which cells were chosen.
	playedCells := func(p *ScoreModelPlayer) map[int]bool {
		cells := map[int]bool{}
		for range 200 {
			col, row, err := p.Move(game)
			if err != nil {
				t.Fatalf("Move: %v", err)
			}
			cells[row*3+col] = true
		}
		return cells
	}

	tests := []struct {
		name   string
		scores []float64
		want   []int // cells that are all played, and no others
	}{
		{"clearly best move", []float64{0.9, 0.1, 0.2, 0.3, 0, 0, 0, 0, 0}, []int{0}},
		{"top 3 within margin", []float64{0.9, 0.85, 0.82, 0.81, 0, 0, 0, 0, 0}, []int{0, 1, 2}},
		{"only moves within margin", []float64{0.9, 0.85, 0.5, 0, 0, 0, 0, 0, 0}, []int{0, 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newSampledScoreModelPlayer(fixedScoreMLP(tt.scores...),
				rand.New(rand.NewSource(1)), 3, 0.1)
			cells := playedCells(p)
			if len(cells) != len(tt.want) {
				t.Errorf("played cells %v, want %v", cells, tt.want)
			}
			for _, c := range tt.want {
				if !cells[c] {
					t.Errorf("cell %d never played, want all of %v", c, tt.want)
				}
			}
		})
	}

	// The same seed gives the same choices.
	model := fixedScoreMLP(0.9, 0.85, 0.82, 0, 0, 0, 0, 0, 0)
	a := newSampledScoreModelPlayer(model, rand.New(rand.NewSource(7)), 3, 0.1)
	b := newSampledScoreModelPlayer(model, rand.New(rand.NewSource(7)), 3, 0.1)
	for i := range 20 {
		colA, rowA, _ := a.Move(game)
		colB, rowB, _ := b.Move(game)
		if colA != colB || rowA != rowB {
			t.Fatalf("move %d: (%d, %d) vs (%d, %d) with the same seed", i, colA, rowA, colB, rowB)
		}
	}

	// Without sampling, the player always plays the best move.
	if col, row, _ := NewScoreModelPlayerFromModel(model).Move(game); col != 0 || row != 0 {
		t.Errorf("deterministic player: (%d, %d), want (0, 0)", col, row)
	}
}
