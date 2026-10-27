package tictactoe

import (
	"math/rand"
	"testing"
)

// TestMinimaxPlayer_Move checks that the player plays the winning move when one exists. The seeded
// Rand makes the pick among equally-optimal moves deterministic.
// Board (X to move, optimal win at (2,0)):
//
//	X X -
//	O O -
//	- - -
func TestMinimaxPlayer_Move(t *testing.T) {
	game := NewGame(3)
	game, _ = PlaceMark(game, 0, 0) // X
	game, _ = PlaceMark(game, 0, 1) // O
	game, _ = PlaceMark(game, 1, 0) // X
	game, _ = PlaceMark(game, 1, 1) // O

	p := MinimaxPlayer{Rand: rand.New(rand.NewSource(42)), MinimaxConfig: DefaultMinimaxConfig()}
	x, y, err := p.Move(game)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if x != 2 || y != 0 {
		t.Errorf("expected the winning move (2,0), got (%d,%d)", x, y)
	}
}
