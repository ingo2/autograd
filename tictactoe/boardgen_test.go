package tictactoe

import (
	"slices"
	"testing"
)

func TestEnumerateBoards_Count(t *testing.T) {
	const want = 5478
	if got := len(Enumerate3x3Boards()); got != want {
		t.Errorf("got %d boards, want %d", got, want)
	}
}

// TestEnumerateBoards_CanonicalInvariant checks: in canonical form count(2)-count(1) must be 0 or
// 1 for every board.
func TestEnumerate3x3Boards_CanonicalInvariant(t *testing.T) {
	for i, b := range Enumerate3x3Boards() {
		var count1, count2 int
		for _, p := range b {
			switch p {
			case X: // value 1 = mover in canonical form
				count1++
			case O: // value 2 = opponent
				count2++
			}
		}
		if d := count2 - count1; d != 0 && d != 1 {
			t.Errorf("board %d: count(2)-count(1)=%d, want 0 or 1: %v", i, d, b)
		}
	}
}

func TestEnumerate3x3Boards_NoDuplicates(t *testing.T) {
	seen := map[[9]Player]struct{}{}
	for _, b := range Enumerate3x3Boards() {
		var key [9]Player
		copy(key[:], b)
		if _, dup := seen[key]; dup {
			t.Errorf("duplicate board: %v", b)
			return
		}
		seen[key] = struct{}{}
	}
}

// TestEnumerateBoards_WorkedExamples checks that the six board positions all appear in the output
// (canonical form).
func TestEnumerate3x3Boards_WorkedExamples(t *testing.T) {
	examples := [][]int{
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 2, 0, 0, 0, 0},
		{1, 1, 0, 2, 2, 0, 0, 0, 0},
		{1, 2, 1, 2, 1, 2, 0, 2, 0},
		{2, 2, 0, 2, 1, 0, 0, 0, 1},
		{2, 2, 2, 1, 1, 0, 0, 0, 1},
	}
	boards := Enumerate3x3Boards()
	for _, ex := range examples {
		want := make(Board, len(ex))
		for i, v := range ex {
			want[i] = Player(v)
		}
		if !slices.ContainsFunc(boards, func(b Board) bool { return slices.Equal(b, want) }) {
			t.Errorf("board not found: %v", ex)
		}
	}
}

func TestDeduplicateBoards(t *testing.T) {
	a := Board{0, 0, 0, 0, 0, 0, 0, 0, 0}
	b1 := Board{1, 0, 0, 0, 0, 0, 0, 0, 0}
	b2 := Board{1, 0, 0, 0, 0, 0, 0, 0, 0}

	got := DeduplicateBoards([]Board{a, b1, b2})
	if len(got) != 2 {
		t.Fatalf("expected 2 boards, got %d", len(got))
	}
	if !slices.Equal(got[0], a) {
		t.Errorf("[0] unexpected board: %v", got[0])
	}
	if !slices.Equal(got[1], b2) {
		t.Errorf("[1] expected last occurrence")
	}
	if got2 := DeduplicateBoards([]Board{a, b1}); len(got2) != 2 {
		t.Errorf("no duplicates: expected 2, got %d", len(got2))
	}
	DeduplicateBoards(nil)
	DeduplicateBoards([]Board{})
}

func TestSortBoards(t *testing.T) {
	boards := []Board{
		{1, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0},
	}
	SortBoards(boards)
	if !slices.Equal(boards[0], Board{0, 0, 0, 0, 0, 0, 0, 0, 0}) {
		t.Errorf("[0] got %v", boards[0])
	}
	if !slices.Equal(boards[1], Board{0, 1, 0, 0, 0, 0, 0, 0, 0}) {
		t.Errorf("[1] got %v", boards[1])
	}
	if !slices.Equal(boards[2], Board{1, 0, 0, 0, 0, 0, 0, 0, 0}) {
		t.Errorf("[2] got %v", boards[2])
	}
	SortBoards(nil)
	SortBoards([]Board{})
}
