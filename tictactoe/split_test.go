package tictactoe

import (
	"math/rand"
	"testing"
)

func makeBoards(n int) []AnnotatedBoard {
	boards := make([]AnnotatedBoard, n)
	for i := range boards {
		boards[i] = AnnotatedBoard{Board: Board{Player(i % 3)}}
	}
	return boards
}

func TestShuffle_ReturnsDifferentOrder(t *testing.T) {
	boards := makeBoards(20)
	shuffled := Shuffle(boards, rand.New(rand.NewSource(42)))
	if len(shuffled) != len(boards) {
		t.Fatalf("length changed: got %d, want %d", len(shuffled), len(boards))
	}
	same := true
	for i := range boards {
		if boards[i].Board[0] != shuffled[i].Board[0] {
			same = false
			break
		}
	}
	if same {
		t.Error("shuffled slice has same order as original")
	}
}

func TestShuffle_DoesNotModifyOriginal(t *testing.T) {
	boards := makeBoards(10)
	original := make([]AnnotatedBoard, len(boards))
	copy(original, boards)
	Shuffle(boards, rand.New(rand.NewSource(7)))
	for i := range boards {
		if boards[i].Board[0] != original[i].Board[0] {
			t.Errorf("original modified at index %d", i)
		}
	}
}

func TestSplit_Sizes(t *testing.T) {
	boards := Shuffle(makeBoards(100), rand.New(rand.NewSource(42)))
	train, test, val, err := Split(boards, 0.2, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if len(test) != 20 {
		t.Errorf("test size: got %d, want 20", len(test))
	}
	if len(val) != 10 {
		t.Errorf("validation size: got %d, want 10", len(val))
	}
	if len(train) != 70 {
		t.Errorf("train size: got %d, want 70", len(train))
	}
	if len(train)+len(test)+len(val) != len(boards) {
		t.Errorf("total mismatch: %d+%d+%d != %d", len(train), len(test), len(val), len(boards))
	}
}

func TestSplit_NoValidation(t *testing.T) {
	boards := Shuffle(makeBoards(50), rand.New(rand.NewSource(1)))
	train, test, val, err := Split(boards, 0.2, 0.0)
	if err != nil {
		t.Fatal(err)
	}
	if len(val) != 0 {
		t.Errorf("expected empty validation set, got %d", len(val))
	}
	if len(test) != 10 {
		t.Errorf("test size: got %d, want 10", len(test))
	}
	if len(train) != 40 {
		t.Errorf("train size: got %d, want 40", len(train))
	}
}

func TestSplit_InvalidArgs(t *testing.T) {
	boards := makeBoards(10)
	cases := [][2]float64{
		{-0.1, 0.0}, // testFraction < 0
		{1.0, 0.0},  // testFraction >= 1
		{0.5, -0.1}, // validationFraction < 0
		{0.5, 0.6},  // sum > 1
	}
	for _, c := range cases {
		if _, _, _, err := Split(boards, c[0], c[1]); err == nil {
			t.Errorf("Split(%g, %g): expected error, got nil", c[0], c[1])
		}
	}
}

// makeDistinctBoards returns n boards each with a unique key (unlike makeBoards which cycles 3).
func makeDistinctBoards(n int) []AnnotatedBoard {
	boards := make([]AnnotatedBoard, n)
	for i := range boards {
		boards[i] = AnnotatedBoard{Board: Board{Player(i)}}
	}
	return boards
}

func TestD4Split_Sizes(t *testing.T) {
	// 100 boards each with a unique key (single-board groups).
	boards := makeDistinctBoards(100)
	train, test, val, err := D4Split(boards, 0.2, 0.1, rand.New(rand.NewSource(42)))
	if err != nil {
		t.Fatal(err)
	}
	if len(train)+len(test)+len(val) != len(boards) {
		t.Errorf("total mismatch: %d+%d+%d != %d", len(train), len(test), len(val), len(boards))
	}
	// Greedy first-fit with single-board groups hits the targets exactly.
	if len(test) != 20 {
		t.Errorf("test size: got %d, want 20", len(test))
	}
	if len(val) != 10 {
		t.Errorf("validation size: got %d, want 10", len(val))
	}
	if len(train) != 70 {
		t.Errorf("train size: got %d, want 70", len(train))
	}
}

func TestD4Split_NoValidation(t *testing.T) {
	boards := makeDistinctBoards(50)
	train, test, val, err := D4Split(boards, 0.2, 0.0, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if len(val) != 0 {
		t.Errorf("expected empty validation set, got %d", len(val))
	}
	if len(test) != 10 {
		t.Errorf("test size: got %d, want 10", len(test))
	}
	if len(train) != 40 {
		t.Errorf("train size: got %d, want 40", len(train))
	}
}

func TestD4Split_GroupsStayTogether(t *testing.T) {
	// makeBoards(12) produces 3 groups of 4 boards (keyed by Player(i%3)).
	boards := makeBoards(12)
	train, test, val, err := D4Split(boards, 0.25, 0.25, rand.New(rand.NewSource(7)))
	if err != nil {
		t.Fatal(err)
	}
	if len(train)+len(test)+len(val) != len(boards) {
		t.Errorf("total mismatch: %d+%d+%d != %d", len(train), len(test), len(val), len(boards))
	}
	// Each board key must appear in exactly one partition.
	type counts struct{ train, test, val int }
	groupCounts := map[BoardKey]*counts{}
	for _, b := range train {
		k := b.Board.ToKey()
		if groupCounts[k] == nil {
			groupCounts[k] = &counts{}
		}
		groupCounts[k].train++
	}
	for _, b := range test {
		k := b.Board.ToKey()
		if groupCounts[k] == nil {
			groupCounts[k] = &counts{}
		}
		groupCounts[k].test++
	}
	for _, b := range val {
		k := b.Board.ToKey()
		if groupCounts[k] == nil {
			groupCounts[k] = &counts{}
		}
		groupCounts[k].val++
	}
	for key, c := range groupCounts {
		parts := 0
		if c.train > 0 {
			parts++
		}
		if c.test > 0 {
			parts++
		}
		if c.val > 0 {
			parts++
		}
		if parts > 1 {
			t.Errorf("group %v appears in %d partitions (train=%d test=%d val=%d)",
				key, parts, c.train, c.test, c.val)
		}
	}
}

func TestD4Split_InvalidArgs(t *testing.T) {
	boards := makeBoards(10)
	cases := [][2]float64{
		{-0.1, 0.0}, // testFraction < 0
		{1.0, 0.0},  // testFraction >= 1
		{0.5, -0.1}, // validationFraction < 0
		{0.5, 0.6},  // sum > 1
	}
	for _, c := range cases {
		if _, _, _, err := D4Split(boards, c[0], c[1], rand.New(rand.NewSource(0))); err == nil {
			t.Errorf("D4Split(%g, %g): expected error, got nil", c[0], c[1])
		}
	}
}

func TestSplit_ZeroFractions(t *testing.T) {
	boards := makeBoards(50)
	train, test, val, err := Split(boards, 0.0, 0.0)
	if err != nil {
		t.Fatal(err)
	}
	if len(test) != 0 {
		t.Errorf("test size: got %d, want 0", len(test))
	}
	if len(val) != 0 {
		t.Errorf("validation size: got %d, want 0", len(val))
	}
	if len(train) != len(boards) {
		t.Errorf("train size: got %d, want %d", len(train), len(boards))
	}
}
