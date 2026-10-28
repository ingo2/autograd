package tictactoe

import (
	"fmt"
	"maps"
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

// applyPerm returns the board p produces from board: result[cell] = board[p[cell]].
func applyPerm(board Board, p []int) Board {
	result := make(Board, len(board))
	for cell := range board {
		result[cell] = board[p[cell]]
	}
	return result
}

// boardLines returns the rows, columns, and both diagonals of an nxn board as sorted cell lists.
func boardLines(n int) [][]int {
	var lines [][]int
	diag, anti := make([]int, n), make([]int, n)
	for i := range n {
		row, col := make([]int, n), make([]int, n)
		for j := range n {
			row[j] = i*n + j
			col[j] = j*n + i
		}
		lines = append(lines, row, col)
		diag[i] = i*n + i
		anti[i] = i*n + n - 1 - i
	}
	slices.Sort(anti)
	return append(lines, diag, anti)
}

// TestD4Perms_GroupProperties checks for several board sizes that the identity together with
// D4Perms forms the dihedral group D4: 8 distinct permutations, closed under composition, and each
// one maps every row, column, and diagonal onto a row, column, or diagonal.
func TestD4Perms_GroupProperties(t *testing.T) {
	for n := 2; n <= 5; n++ {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			identity := make([]int, n*n)
			for i := range identity {
				identity[i] = i
			}
			group := append([][]int{identity}, D4Perms(n)...)

			members := map[string]bool{}
			for i, p := range group {
				if !slices.Equal(slices.Sorted(slices.Values(p)), identity) {
					t.Fatalf("group[%d] = %v is not a permutation of 0..%d", i, p, n*n-1)
				}
				members[fmt.Sprint(p)] = true
			}
			if len(members) != 8 {
				t.Fatalf("got %d distinct permutations, want 8", len(members))
			}

			for i, p := range group {
				for j, q := range group {
					composed := make([]int, n*n)
					for cell := range composed {
						composed[cell] = q[p[cell]]
					}
					if !members[fmt.Sprint(composed)] {
						t.Errorf("group[%d] after group[%d] = %v is not in the group", i, j, composed)
					}
				}
			}

			lines := map[string]bool{}
			for _, line := range boardLines(n) {
				lines[fmt.Sprint(line)] = true
			}
			for i, p := range group {
				for _, line := range boardLines(n) {
					image := make([]int, len(line))
					for k, cell := range line {
						image[k] = p[cell]
					}
					slices.Sort(image)
					if !lines[fmt.Sprint(image)] {
						t.Errorf("group[%d] maps line %v to %v, which is not a line", i, line, image)
					}
				}
			}
		})
	}
}

// TestCanonicalKey_ReachableBoards checks, for every reachable 3x3 board, that all of its D4
// variants share one CanonicalKey and that this key is the smallest of the variants' own keys.
// The reachable boards must fall into the 765 well-known symmetry classes of tic-tac-toe.
func TestCanonicalKey_ReachableBoards(t *testing.T) {
	classes := map[BoardKey]struct{}{}
	for _, board := range Enumerate3x3Boards() {
		want := board.ToKey()
		for _, p := range D4Perms(3) {
			want = min(want, applyPerm(board, p).ToKey())
		}
		if got := CanonicalKey(board); got != want {
			t.Fatalf("CanonicalKey(%v) = %d, want %d (smallest key of its variants)", board, got, want)
		}
		for _, p := range D4Perms(3) {
			variant := applyPerm(board, p)
			if got := CanonicalKey(variant); got != want {
				t.Fatalf("CanonicalKey(%v) = %d, want %d (same as its variant %v)",
					variant, got, want, board)
			}
		}
		classes[want] = struct{}{}
	}
	if len(classes) != 765 {
		t.Errorf("got %d symmetry classes, want 765", len(classes))
	}
}

// TestD4CanonicalPerm_RepMatchesPerm checks the documented contract rep[cell] = board[perm[cell]].
func TestD4CanonicalPerm_RepMatchesPerm(t *testing.T) {
	for _, board := range Enumerate3x3Boards() {
		rep, perm := d4CanonicalPerm(board)
		if got := applyPerm(board, perm); !slices.Equal(got, rep) {
			t.Fatalf("d4CanonicalPerm(%v): applying perm %v gives %v, want rep %v",
				board, perm, got, rep)
		}
	}
}

// TestD4Augment_ScoresMatchMinimax checks on every dataset position that each variant D4Augment
// produces carries exactly the scores minimax computes for that variant directly.
func TestD4Augment_ScoresMatchMinimax(t *testing.T) {
	boards := FilterBoards(Enumerate3x3Boards())
	annotated := AnnotateCanonicalBoards(boards, X, DefaultMinimaxConfig())
	byKey := make(map[BoardKey]AnnotatedBoard, len(annotated))
	for _, ab := range annotated {
		byKey[ab.Board.ToKey()] = ab
	}

	for _, ab := range annotated {
		for _, variant := range D4Augment([]AnnotatedBoard{ab}) {
			want, ok := byKey[variant.Board.ToKey()]
			if !ok {
				t.Fatalf("variant %v of %v is not a reachable dataset board", variant.Board, ab.Board)
			}
			if !maps.Equal(scoresByPos(variant.Scores), scoresByPos(want.Scores)) {
				t.Fatalf("variant %v of %v: Scores = %v, want %v",
					variant.Board, ab.Board, variant.Scores, want.Scores)
			}
		}
	}
}

func scoresByPos(scores []ScoredMove) map[int]float64 {
	m := make(map[int]float64, len(scores))
	for _, s := range scores {
		m[s.Pos] = s.Score
	}
	return m
}

// TestD4Augment_DatasetAlreadyComplete checks that the 3x3 dataset boards already contain every D4
// variant of each position, so augmenting them adds nothing.
func TestD4Augment_DatasetAlreadyComplete(t *testing.T) {
	boards := FilterBoards(Enumerate3x3Boards())
	annotated := make([]AnnotatedBoard, len(boards))
	for i, b := range boards {
		annotated[i] = AnnotatedBoard{Board: b, Player: X}
	}

	if got := len(D4Augment(annotated)); got != len(boards) {
		t.Errorf("len(D4Augment(dataset)) = %d, want %d", got, len(boards))
	}
}
