package scoremodel

import (
	"autograd/tictactoe"
	"testing"
)

// TestBoardToScoreModelInput checks the dataset encoding: on a perspective-canonical board the
// mover (X) lands in plane 0 and the opponent (O) in plane 1.
func TestBoardToScoreModelInput(t *testing.T) {
	board := make(tictactoe.Board, 9)
	board[0] = tictactoe.X
	board[4] = tictactoe.O
	ab := tictactoe.AnnotatedBoard{Board: board, Player: tictactoe.X}

	input := BoardToScoreModelInput(&ab)

	if len(input) != 18 {
		t.Fatalf("input width for a 3x3 board: got %d, want 18", len(input))
	}
	if input[0].Data != 1.0 {
		t.Errorf("plane0[0] (mover piece): got %g, want 1.0", input[0].Data)
	}
	if input[9+4].Data != 1.0 {
		t.Errorf("plane1[4] (opponent piece): got %g, want 1.0", input[9+4].Data)
	}
	for _, pos := range []int{1, 2, 3, 5, 6, 7, 8} {
		if input[pos].Data != 0.0 {
			t.Errorf("plane0[%d]: expected 0.0, got %g", pos, input[pos].Data)
		}
		if input[9+pos].Data != 0.0 {
			t.Errorf("plane1[%d]: expected 0.0, got %g", pos, input[9+pos].Data)
		}
	}
}

// TestGameToScoreModelInput_XToMove checks that X pieces land in plane 0 and O pieces land in
// plane 1 when X is the active player.
func TestGameToScoreModelInput_XToMove(t *testing.T) {
	// Board: X at position 0, O at position 4, rest empty.
	board := make(tictactoe.Board, 9)
	board[0] = tictactoe.X
	board[4] = tictactoe.O
	game := makeGame(board, tictactoe.X)

	input := GameToScoreModelInput(game)

	if input[0].Data != 1.0 {
		t.Errorf("plane0[0] (X piece, X to move): got %g, want 1.0", input[0].Data)
	}
	if input[9+4].Data != 1.0 {
		t.Errorf("plane1[4] (O piece, X to move): got %g, want 1.0", input[9+4].Data)
	}
	// Empty cells must be zero in both planes.
	for _, pos := range []int{1, 2, 3, 5, 6, 7, 8} {
		if input[pos].Data != 0.0 {
			t.Errorf("plane0[%d]: expected 0.0, got %g", pos, input[pos].Data)
		}
		if input[9+pos].Data != 0.0 {
			t.Errorf("plane1[%d]: expected 0.0, got %g", pos, input[9+pos].Data)
		}
	}
}

// TestGameToScoreModelInput_LargerBoard checks that the encoding follows the board size instead of
// assuming 3x3: the planes are as wide as the board has cells.
func TestGameToScoreModelInput_LargerBoard(t *testing.T) {
	// 4x4 board: X at position 0, O at position 15.
	board := make(tictactoe.Board, 16)
	board[0] = tictactoe.X
	board[15] = tictactoe.O
	game := tictactoe.Game{Board: board, Size: 4, ConnectN: 4, ActivePlayer: tictactoe.X}

	input := GameToScoreModelInput(game)

	if len(input) != 32 {
		t.Fatalf("input width for a 4x4 board: got %d, want 32", len(input))
	}
	if input[0].Data != 1.0 {
		t.Errorf("plane0[0] (X piece, X to move): got %g, want 1.0", input[0].Data)
	}
	if input[16+15].Data != 1.0 {
		t.Errorf("plane1[15] (O piece, X to move): got %g, want 1.0", input[16+15].Data)
	}
}

// TestGameToScoreModelInput_OToMove checks that when O is the active player the encoding swaps: O
// pieces go to plane 0 and X pieces go to plane 1.
func TestGameToScoreModelInput_OToMove(t *testing.T) {
	board := make(tictactoe.Board, 9)
	board[0] = tictactoe.X
	board[4] = tictactoe.O
	game := makeGame(board, tictactoe.O)

	input := GameToScoreModelInput(game)

	// O is active player -> O pieces in plane 0, X pieces in plane 1.
	if input[4].Data != 1.0 {
		t.Errorf("plane0[4] (O piece, O to move): got %g, want 1.0", input[4].Data)
	}
	if input[9+0].Data != 1.0 {
		t.Errorf("plane1[0] (X piece, O to move): got %g, want 1.0", input[9+0].Data)
	}
	if input[0].Data != 0.0 {
		t.Errorf("plane0[0] should be empty when X piece is in plane 1: got %g", input[0].Data)
	}
}
