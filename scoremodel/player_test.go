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
