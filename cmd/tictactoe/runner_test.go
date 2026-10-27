package main

import (
	"autograd/tictactoe"
	"fmt"
	"strings"
	"testing"
)

// firstCellPlayer always picks the first empty cell -> predictable for tests.
type firstCellPlayer struct{}

func (firstCellPlayer) Identity() string { return tictactoe.IdentityAIMinimax }
func (firstCellPlayer) Name() string     { return "Test AI" }
func (firstCellPlayer) Move(g tictactoe.Game) (int, int, error) {
	for pos, cell := range g.Board {
		if cell == tictactoe.Empty {
			return pos % g.Size, pos / g.Size, nil
		}
	}
	return 0, 0, fmt.Errorf("no moves available")
}

func newTestRunner(input string) Runner {
	return Runner{
		in:        strings.NewReader(input),
		out:       &strings.Builder{},
		ai:        firstCellPlayer{},
		boardSize: 3,
		connectN:  3,
	}
}

// Human X plays column 0 top-to-bottom; AI fills row 0 left-to-right. Moves are entered as
// "row col". After 3 human moves X wins column 0.
//
// Turn sequence (row, col):
// Human X: (0,0)  AI O: (0,1)
// Human X: (1,0)  AI O: (0,2)
// Human X: (2,0)  -> X wins
func TestPlay_HumanWins(t *testing.T) {
	r := newTestRunner("0 0\n1 0\n2 0\n")
	game, err := r.Play(tictactoe.X)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if game.Status != tictactoe.Won {
		t.Fatalf("expected Won, got %v", game.Status)
	}
	if game.Winner != tictactoe.X {
		t.Errorf("expected winner X, got %v", game.Winner)
	}
}

// Human O plays out of the way; AI X fills row 0 (positions 0,1,2) and wins.
// Turn sequence (row, col):
// AI X: (0,0)  Human O: (2,2)
// AI X: (0,1)  Human O: (2,1)
// AI X: (0,2)  -> X wins row 0
func TestPlay_AIWins(t *testing.T) {
	r := newTestRunner("2 2\n2 1\n")
	game, err := r.Play(tictactoe.O)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if game.Status != tictactoe.Won {
		t.Fatalf("expected Won, got %v", game.Status)
	}
	if game.Winner != tictactoe.X {
		t.Errorf("expected winner X (AI), got %v", game.Winner)
	}
}

// Invalid and occupied inputs are rejected; the game continues normally.
func TestPlay_InvalidInputRetried(t *testing.T) {
	// "abc" -> not two numbers.
	// "0 0" twice -> second is occupied.
	// Valid subsequent moves complete the game.
	input := "abc\n0 0\n0 0\n1 0\n2 0\n"
	r := newTestRunner(input)
	game, err := r.Play(tictactoe.X)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if game.Status != tictactoe.Won {
		t.Fatalf("expected Won, got %v", game.Status)
	}
	if game.Winner != tictactoe.X {
		t.Errorf("expected winner X, got %v", game.Winner)
	}
}
