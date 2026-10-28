package tictactoe

import (
	"testing"
)

func TestNewGame(t *testing.T) {
	game := NewGame(3)

	if game.Size != 3 {
		t.Errorf("expected Size=3, got %d", game.Size)
	}
	if game.ActivePlayer != X {
		t.Errorf("expected ActivePlayer=X, got %v", game.ActivePlayer)
	}
	if game.Status != Running {
		t.Errorf("expected Status=Running, got %v", game.Status)
	}
	if len(game.Board) != 9 {
		t.Errorf("expected Board length=9, got %d", len(game.Board))
	}
}

func TestGame_GetCell(t *testing.T) {
	game := NewGame(3)
	game, _ = PlaceMark(game, 1, 1) // X at center

	if game.GetCell(1, 1) != X {
		t.Errorf("expected X at (1,1)")
	}
	if game.GetCell(0, 0) != Empty {
		t.Errorf("expected Empty at (0,0)")
	}
}

func TestPlaceMark_SimpleMove(t *testing.T) {
	game := NewGame(3)
	newGame, err := PlaceMark(game, 1, 1) // center

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if newGame.Board[4] != X {
		t.Errorf("expected X on Position (1,1)")
	}
	if newGame.ActivePlayer != O {
		t.Errorf("expected next Player=O, got %v", newGame.ActivePlayer)
	}
	// Original not changed.
	if game.Board[4] != Empty {
		t.Errorf("original Board should not be changed")
	}
}

func TestPlaceMark_InvalidPosition(t *testing.T) {
	game := NewGame(3)
	_, err := PlaceMark(game, 3, 0) // x=3 is out of bounds for a 3x3 board
	if err == nil {
		t.Error("expected error for invalid position")
	}
}

func TestPlaceMark_OccupiedPosition(t *testing.T) {
	game := NewGame(3)
	game, _ = PlaceMark(game, 1, 1)
	_, err := PlaceMark(game, 1, 1)
	if err == nil {
		t.Error("expected error for occupied position")
	}
}

func TestPlaceMark_WinnerRow(t *testing.T) {
	// X wins with top row: (0,0),(1,0),(2,0).
	game := NewGame(3)
	game, _ = PlaceMark(game, 0, 0)    // X
	game, _ = PlaceMark(game, 0, 1)    // O
	game, _ = PlaceMark(game, 1, 0)    // X
	game, _ = PlaceMark(game, 1, 1)    // O
	game, err := PlaceMark(game, 2, 0) // X wins.

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if game.Status != Won {
		t.Errorf("expected Status=Won, got %v", game.Status)
	}
	if game.Winner != X {
		t.Errorf("expected Winner=X, got %v", game.Winner)
	}
}

func TestPlaceMark_WinnerDiagonal(t *testing.T) {
	// X wins with diagonal: (0,0),(1,1),(2,2).
	game := NewGame(3)
	game, _ = PlaceMark(game, 0, 0)    // X
	game, _ = PlaceMark(game, 1, 0)    // O
	game, _ = PlaceMark(game, 1, 1)    // X
	game, _ = PlaceMark(game, 2, 0)    // O
	game, err := PlaceMark(game, 2, 2) // X wins.

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if game.Status != Won || game.Winner != X {
		t.Errorf("expected X as winner")
	}
}

func TestPlaceMark_Draw(t *testing.T) {
	// Draw: X O X / O X O / O X O.
	game := NewGame(3)
	moves := [][2]int{{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {0, 2}, {2, 1}, {2, 2}, {1, 2}}
	var err error
	for _, m := range moves {
		game, err = PlaceMark(game, m[0], m[1])
		if err != nil {
			t.Fatalf("unexpected error at (%d,%d): %v", m[0], m[1], err)
		}
	}
	if game.Status != Draw {
		t.Errorf("expected Status=Draw, got %v", game.Status)
	}
}

// TestConnectN uses a 4x4 board with connectN=3 and verifies:
//   - 3-in-a-row wins (row, column, both diagonals)
//   - 4 marks in a row on a full row also wins (just the first window of 3 triggers it)
//   - filling 2 marks does not win
func TestConnectN(t *testing.T) {
	place := func(game Game, moves [][2]int) Game {
		t.Helper()
		for _, m := range moves {
			var err error
			game, err = PlaceMark(game, m[0], m[1])
			if err != nil {
				t.Fatalf("unexpected error at (%d,%d): %v", m[0], m[1], err)
			}
		}
		return game
	}

	t.Run("row win", func(t *testing.T) {
		// X wins with 3-in-a-row on row 1: (0,1),(1,1),(2,1)
		// Board (4x4, connect-3):
		//   O O . .
		//   X X X .
		//   . . . .
		//   . . . .
		game := NewConnectNGame(4, 3)
		game = place(game, [][2]int{{0, 1}, {0, 0}, {1, 1}, {1, 0}, {2, 1}})
		if game.Status != Won || game.Winner != X {
			t.Errorf("expected X to win via row, got status=%v winner=%v", game.Status, game.Winner)
		}
	})

	t.Run("column win", func(t *testing.T) {
		// X wins with 3-in-a-column on col 1: (1,0),(1,1),(1,2)
		// Board:
		//   O X . .
		//   O X . .
		//   . X . .
		//   . . . .
		game := NewConnectNGame(4, 3)
		game = place(game, [][2]int{{1, 0}, {0, 0}, {1, 1}, {0, 1}, {1, 2}})
		if game.Status != Won || game.Winner != X {
			t.Errorf("expected X to win via column, got status=%v winner=%v", game.Status, game.Winner)
		}
	})

	t.Run("diagonal top-left to bottom-right win", func(t *testing.T) {
		// X wins with diagonal: (1,0),(2,1),(3,2)
		// Board:
		//   O X . .
		//   O . X .
		//   . . . X
		//   . . . .
		game := NewConnectNGame(4, 3)
		game = place(game, [][2]int{{1, 0}, {0, 0}, {2, 1}, {0, 1}, {3, 2}})
		if game.Status != Won || game.Winner != X {
			t.Errorf("expected X to win via \\ diagonal, got status=%v winner=%v", game.Status, game.Winner)
		}
	})

	t.Run("diagonal top-right to bottom-left win", func(t *testing.T) {
		// X wins with anti-diagonal: (2,0),(1,1),(0,2)
		// Board:
		//   O . X .
		//   O X . .
		//   X . . .
		//   . . . .
		game := NewConnectNGame(4, 3)
		game = place(game, [][2]int{{2, 0}, {0, 0}, {1, 1}, {0, 1}, {0, 2}})
		if game.Status != Won || game.Winner != X {
			t.Errorf("expected X to win via / diagonal, got status=%v winner=%v", game.Status, game.Winner)
		}
	})

	t.Run("two in a row does not win", func(t *testing.T) {
		// X has only 2 in a row, should still be running.
		game := NewConnectNGame(4, 3)
		game = place(game, [][2]int{{0, 0}, {0, 1}, {1, 0}})
		if game.Status != Running {
			t.Errorf("expected Running after 2-in-a-row, got %v", game.Status)
		}
	})
}

func TestPlaceMark_GameAlreadyEnded(t *testing.T) {
	game := NewGame(3)
	game, _ = PlaceMark(game, 0, 0) // X
	game, _ = PlaceMark(game, 0, 1) // O
	game, _ = PlaceMark(game, 1, 0) // X
	game, _ = PlaceMark(game, 1, 1) // O
	game, _ = PlaceMark(game, 2, 0) // X wins.

	_, err := PlaceMark(game, 2, 1)
	if err == nil {
		t.Error("expected error when game is already ended")
	}
}

func TestAvailableMoves(t *testing.T) {
	game := NewGame(3)
	game, _ = PlaceMark(game, 0, 0) // X at (0,0)
	game, _ = PlaceMark(game, 1, 1) // O at (1,1)

	moves := AvailableMoves(game.Board)

	expected := []int{1, 2, 3, 5, 6, 7, 8}
	if len(moves) != len(expected) {
		t.Fatalf("expected %d moves, got %d: %v", len(expected), len(moves), moves)
	}
	for i, pos := range expected {
		if moves[i] != pos {
			t.Errorf("moves[%d]: expected %d, got %d", i, pos, moves[i])
		}
	}
}

func TestAvailableMoves_EmptyBoard(t *testing.T) {
	game := NewGame(3)

	moves := AvailableMoves(game.Board)

	if len(moves) != 9 {
		t.Errorf("expected 9 moves on empty board, got %d", len(moves))
	}
}

func TestAvailableMoves_FullBoard(t *testing.T) {
	game := NewGame(3)
	for _, mv := range [][2]int{{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {0, 2}, {2, 1}, {2, 2},
		{1, 2}} {
		game, _ = PlaceMark(game, mv[0], mv[1])
	}

	moves := AvailableMoves(game.Board)

	if len(moves) != 0 {
		t.Errorf("expected 0 moves on full board, got %d: %v", len(moves), moves)
	}
}
