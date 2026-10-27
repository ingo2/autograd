package tictactoe

import (
	"slices"
	"testing"
)

func TestD4Perms_3x3(t *testing.T) {
	// Verify D4Perms(3) against the previously hardcoded permutation arrays.
	want := [][]int{
		{2, 5, 8, 1, 4, 7, 0, 3, 6}, // r090: (r,c) -> (c, n-1-r)
		{8, 7, 6, 5, 4, 3, 2, 1, 0}, // r180: (r,c) -> (n-1-r, n-1-c)
		{6, 3, 0, 7, 4, 1, 8, 5, 2}, // r270: (r,c) -> (n-1-c, r)
		{2, 1, 0, 5, 4, 3, 8, 7, 6}, // fh:   (r,c) -> (r, n-1-c)  flip left-right
		{6, 7, 8, 3, 4, 5, 0, 1, 2}, // fv:   (r,c) -> (n-1-r, c)  flip top-bottom
		{0, 3, 6, 1, 4, 7, 2, 5, 8}, // d+:   (r,c) -> (c, r)      main diagonal
		{8, 5, 2, 7, 4, 1, 6, 3, 0}, // d-:   (r,c) -> (n-1-c, n-1-r)  anti-diagonal
	}
	got := D4Perms(3)
	if len(got) != len(want) {
		t.Fatalf("len(D4Perms(3)) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !slices.Equal(got[i], want[i]) {
			t.Errorf("D4Perms(3)[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestD4Augment_Count(t *testing.T) {
	// Both boards have a full D4 orbit of size 8 (no symmetry maps either board to itself), and the
	// two orbits don't overlap, so the result has 8*len(boards) entries.
	boards := []AnnotatedBoard{
		{Board: Board{1, 0, 2, 0, 0, 0, 0, 0, 0}, Player: X},
		{Board: Board{1, 2, 0, 0, 0, 0, 0, 0, 0}, Player: X},
	}

	result := D4Augment(boards)

	if len(result) != 8*len(boards) {
		t.Errorf("len(result) = %d, want %d", len(result), 8*len(boards))
	}
}

func TestD4Augment_EmptyInput(t *testing.T) {
	result := D4Augment(nil)
	if len(result) != 0 {
		t.Errorf("len(result) = %d, want 0", len(result))
	}
}

func TestD4Augment_IdentityIsFirst(t *testing.T) {
	orig := AnnotatedBoard{
		Board: Board{
			1, 0, 2,
			0, 0, 0,
			0, 0, 0},
		Player: X,
		Scores: []ScoredMove{{Pos: 0, Score: 1.0}, {Pos: 2, Score: -1.0}},
	}

	result := D4Augment([]AnnotatedBoard{orig})

	if !slices.Equal(result[0].Board, orig.Board) {
		t.Errorf("result[0].Board = %v, want %v (unchanged)", result[0].Board, orig.Board)
	}
	if result[0].Player != orig.Player {
		t.Errorf("result[0].Player = %v, want %v", result[0].Player, orig.Player)
	}
	if !slices.Equal(result[0].Scores, orig.Scores) {
		t.Errorf("result[0].Scores = %v, want %v (unchanged)", result[0].Scores, orig.Scores)
	}
}

// TestD4Augment_R090Transform checks the exact board and Scores produced by the first symmetry (a
// 90 degree rotation) against hand-computed values.
func TestD4Augment_R090Transform(t *testing.T) {
	orig := AnnotatedBoard{
		Board: Board{
			1, 0, 2,
			0, 0, 0,
			0, 0, 0},
		Player: X,
		Scores: []ScoredMove{{Pos: 0, Score: 1.0}, {Pos: 2, Score: -1.0}},
	}

	result := D4Augment([]AnnotatedBoard{orig})

	wantBoard := Board{
		2, 0, 0,
		0, 0, 0,
		1, 0, 0}
	if !slices.Equal(result[1].Board, wantBoard) {
		t.Errorf("result[1].Board = %v, want %v", result[1].Board, wantBoard)
	}

	wantScores := []ScoredMove{{Pos: 6, Score: 1.0}, {Pos: 0, Score: -1.0}}
	if !slices.Equal(result[1].Scores, wantScores) {
		t.Errorf("result[1].Scores = %v, want %v", result[1].Scores, wantScores)
	}
}

// TestD4Augment_PieceAndScorePositionsAgree checks, for every one of the 8 symmetries, that a
// candidate move's Scores.Pos always points at the same cell that the corresponding piece ends up
// in. This is the property that the buggy Scores.Pos transform for the 4-cycle rotations violated.
func TestD4Augment_PieceAndScorePositionsAgree(t *testing.T) {
	for pos := range 9 {
		board := make(Board, 9)
		board[pos] = X
		orig := AnnotatedBoard{
			Board:  board,
			Player: X,
			Scores: []ScoredMove{{Pos: pos, Score: 0.42}},
		}

		result := D4Augment([]AnnotatedBoard{orig})

		// The center cell (4) is fixed by every D4 symmetry, so all 8 candidates are identical and
		// dedup leaves just 1. Every other cell has an orbit of size 4 under D4.
		wantLen := 4
		if pos == 4 {
			wantLen = 1
		}
		if len(result) != wantLen {
			t.Fatalf("orig pos %d: len(result) = %d, want %d", pos, len(result), wantLen)
		}

		for i, ab := range result {
			piecePos := slices.Index(ab.Board, X)
			if piecePos < 0 {
				t.Fatalf("orig pos %d, result %d: no X piece found in transformed board", pos, i)
			}
			if ab.Scores[0].Pos != piecePos {
				t.Errorf("orig pos %d, result %d: Scores[0].Pos = %d, want %d (X piece position)",
					pos, i, ab.Scores[0].Pos, piecePos)
			}
			if ab.Scores[0].Score != 0.42 {
				t.Errorf("orig pos %d, result %d: Scores[0].Score = %g, want 0.42",
					pos, i, ab.Scores[0].Score)
			}
		}
	}
}

// TestD4Augment_PlayerAndCellsPreserved checks that every symmetry keeps Player unchanged and
// produces a board that is a permutation of the original board's cells.
func TestD4Augment_PlayerAndCellsPreserved(t *testing.T) {
	orig := AnnotatedBoard{
		Board: Board{
			1, 0, 2,
			0, 1, 0,
			2, 0, 0},
		Player: O,
	}

	result := D4Augment([]AnnotatedBoard{orig})

	wantCells := slices.Clone(orig.Board)
	slices.Sort(wantCells)

	for i, ab := range result {
		if ab.Player != orig.Player {
			t.Errorf("symmetry %d: Player = %v, want %v", i, ab.Player, orig.Player)
		}
		gotCells := slices.Clone(ab.Board)
		slices.Sort(gotCells)
		if !slices.Equal(gotCells, wantCells) {
			t.Errorf("symmetry %d: Board cells = %v, want a permutation of %v", i, ab.Board, wantCells)
		}
	}
}

// TestD4Augment_MultipleBoards checks that results from multiple input boards don't overlap and
// that each board's identity copy appears at the expected offset.
func TestD4Augment_MultipleBoards(t *testing.T) {
	board0 := AnnotatedBoard{
		Board:  Board{1, 0, 0, 0, 0, 0, 0, 0, 0},
		Player: X}
	board1 := AnnotatedBoard{
		Board:  Board{0, 0, 0, 0, 0, 0, 0, 0, 2},
		Player: O}

	result := D4Augment([]AnnotatedBoard{board0, board1})

	// A single corner piece has an orbit of size 4 under D4 (corner cells form one orbit of size 4),
	// so each board contributes 4 unique entries after dedup.
	if len(result) != 8 {
		t.Fatalf("len(result) = %d, want 8", len(result))
	}
	if !slices.Equal(result[0].Board, board0.Board) {
		t.Errorf("result[0].Board = %v, want %v (board0 identity)", result[0].Board, board0.Board)
	}
	if !slices.Equal(result[4].Board, board1.Board) {
		t.Errorf("result[4].Board = %v, want %v (board1 identity)", result[4].Board, board1.Board)
	}
}

// TestD4Augment_Deduplication checks that boards which are invariant under one or more D4
// symmetries don't appear more than once in the result, with the original (untransformed) board
// kept as the surviving copy.
func TestD4Augment_Deduplication(t *testing.T) {
	tests := []struct {
		name    string
		board   Board
		wantLen int
	}{
		{"empty board, fixed by all symmetries", make(Board, 9), 1},
		{"center piece, fixed by all symmetries", Board{0, 0, 0, 0, 1, 0, 0, 0, 0}, 1},
		{"corner piece, orbit of size 4", Board{1, 0, 0, 0, 0, 0, 0, 0, 0}, 4},
		{"edge piece, orbit of size 4", Board{0, 1, 0, 0, 0, 0, 0, 0, 0}, 4},
		{"two pieces, full orbit of size 8", Board{1, 0, 2, 0, 0, 0, 0, 0, 0}, 8},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			orig := AnnotatedBoard{Board: tc.board, Player: X}

			result := D4Augment([]AnnotatedBoard{orig})

			if len(result) != tc.wantLen {
				t.Fatalf("len(result) = %d, want %d", len(result), tc.wantLen)
			}
			if !slices.Equal(result[0].Board, tc.board) {
				t.Errorf("result[0].Board = %v, want %v (original board kept first)",
					result[0].Board, tc.board)
			}
			seen := make(map[BoardKey]struct{}, len(result))
			for i, ab := range result {
				key := ab.Board.ToKey()
				if _, ok := seen[key]; ok {
					t.Errorf("result[%d].Board = %v is a duplicate of an earlier entry", i, ab.Board)
				}
				seen[key] = struct{}{}
			}
		})
	}
}

func TestRemoveOverlap(t *testing.T) {
	board := func(b Board) AnnotatedBoard {
		return AnnotatedBoard{Board: b, Player: X}
	}

	a := board(Board{1, 0, 0, 0, 0, 0, 0, 0, 0})
	b := board(Board{0, 1, 0, 0, 0, 0, 0, 0, 0})
	c := board(Board{0, 0, 1, 0, 0, 0, 0, 0, 0})

	result := RemoveOverlap([]AnnotatedBoard{a, b, c}, []AnnotatedBoard{b})

	if len(result) != 2 {
		t.Fatalf("len(result) = %d, want 2", len(result))
	}
	if !slices.Equal(result[0].Board, a.Board) || !slices.Equal(result[1].Board, c.Board) {
		t.Errorf("result = %v, want [%v, %v]", result, a.Board, c.Board)
	}
}

func TestRemoveOverlap_MultipleExcludeSets(t *testing.T) {
	board := func(b Board) AnnotatedBoard {
		return AnnotatedBoard{Board: b, Player: X}
	}

	a := board(Board{1, 0, 0, 0, 0, 0, 0, 0, 0})
	b := board(Board{0, 1, 0, 0, 0, 0, 0, 0, 0})
	c := board(Board{0, 0, 1, 0, 0, 0, 0, 0, 0})

	result := RemoveOverlap([]AnnotatedBoard{a, b, c},
		[]AnnotatedBoard{a}, []AnnotatedBoard{c})

	if len(result) != 1 {
		t.Fatalf("len(result) = %d, want 1", len(result))
	}
	if !slices.Equal(result[0].Board, b.Board) {
		t.Errorf("result = %v, want [%v]", result, b.Board)
	}
}

func TestRemoveOverlap_NoOverlap(t *testing.T) {
	a := AnnotatedBoard{
		Board:  Board{1, 0, 0, 0, 0, 0, 0, 0, 0},
		Player: X}
	b := AnnotatedBoard{
		Board:  Board{0, 1, 0, 0, 0, 0, 0, 0, 0},
		Player: X}
	exclude := AnnotatedBoard{
		Board:  Board{0, 0, 1, 0, 0, 0, 0, 0, 0},
		Player: X}

	result := RemoveOverlap([]AnnotatedBoard{a, b}, []AnnotatedBoard{exclude})

	if len(result) != 2 {
		t.Fatalf("len(result) = %d, want 2", len(result))
	}
}
