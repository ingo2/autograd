package tictactoe

import (
	"path/filepath"
	"slices"
	"testing"
)

var fullSearchConfig = MinimaxConfig{
	Gamma: 0.9,
}

func makeBoard(vals ...int) Board {
	b := make(Board, len(vals))
	for i, v := range vals {
		b[i] = Player(v)
	}
	return b
}

// Tests for AnnotateCanonicalBoard.

func TestAnnotateCanonicalBoard_ImmediateWin(t *testing.T) {
	// Mover has stones at 0,1; position 2 completes top row.
	b := makeBoard(1, 1, 0, 2, 2, 0, 0, 0, 0)
	result := AnnotateCanonicalBoard(b, X, fullSearchConfig)
	if len(result.Scores) == 0 {
		t.Fatal("expected scores, got none")
	}
	if result.Scores[0].Pos != 2 {
		t.Errorf("best position: got %d, want 2", result.Scores[0].Pos)
	}
	if result.Scores[0].Score <= 0 {
		t.Errorf("best score should be positive (win), got %v", result.Scores[0].Score)
	}
}

func TestAnnotateCanonicalBoard_Terminal(t *testing.T) {
	// Opponent completed top row, no moves left.
	b := makeBoard(2, 2, 2, 1, 1, 0, 0, 0, 1)
	result := AnnotateCanonicalBoard(b, X, fullSearchConfig)
	if len(result.Scores) != 0 {
		t.Errorf("terminal board: expected empty scores, got %v", result.Scores)
	}
}

func TestAnnotateCanonicalBoard_EmptyBoard(t *testing.T) {
	// All 9 moves draw under perfect play, so every score must be 0.
	b := makeBoard(0, 0, 0, 0, 0, 0, 0, 0, 0)
	result := AnnotateCanonicalBoard(b, X, fullSearchConfig)
	if result.Player != X {
		t.Errorf("Player: got %v, want X", result.Player)
	}
	if len(result.Scores) != 9 {
		t.Errorf("empty board: expected 9 scores, got %d", len(result.Scores))
	}
	for _, sm := range result.Scores {
		if sm.Score != 0 {
			t.Errorf("position %d: score %v, want 0 (draw)", sm.Pos, sm.Score)
		}
	}
}

func TestAnnotateCanonicalBoards(t *testing.T) {
	shallow := MinimaxConfig{MaxDepth: 1, Gamma: 0.9}
	want := DeduplicateBoards(FilterBoards(Enumerate3x3Boards()))

	got := AnnotateCanonicalBoards(slices.Clone(want), X, shallow)
	if len(got) != len(want) {
		t.Fatalf("count: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !slices.Equal(got[i].Board, want[i]) {
			t.Fatalf("[%d] Board: got %v, want %v", i, got[i].Board, want[i])
		}
		if got[i].Player != X {
			t.Errorf("[%d] Player: got %v, want X", i, got[i].Player)
		}
	}
}

// Tests for annotateAbsoluteBoard.

func TestAnnotateAbsoluteBoard_PlayerX_MatchesCanonical(t *testing.T) {
	// When player is X, absolute == canonical, so results must be identical.
	b := makeBoard(1, 1, 0, 2, 2, 0, 0, 0, 0)
	abs := annotateAbsoluteBoard(b, X, fullSearchConfig)
	can := AnnotateCanonicalBoard(b, X, fullSearchConfig)
	if len(abs.Scores) != len(can.Scores) {
		t.Fatalf("score count: abs=%d can=%d", len(abs.Scores), len(can.Scores))
	}
	for i := range abs.Scores {
		if abs.Scores[i] != can.Scores[i] {
			t.Errorf("Scores[%d]: abs=%v can=%v", i, abs.Scores[i], can.Scores[i])
		}
	}
}

func TestAnnotateAbsoluteBoard_PlayerO_RelabelsCorrectly(t *testing.T) {
	// Absolute board after X plays center, O to move: [0,0,0,0,1,0,0,0,0].
	// Canonical equivalent: [0,0,0,0,2,0,0,0,0].
	// Against a center opening, O draws only from corners; edges lose.
	absBoard := makeBoard(0, 0, 0, 0, 1, 0, 0, 0, 0)
	result := annotateAbsoluteBoard(absBoard, O, fullSearchConfig)
	best := BestPositions(result.Scores)
	slices.Sort(best)
	if !slices.Equal(best, []int{0, 2, 6, 8}) {
		t.Errorf("best positions: got %v, want [0 2 6 8]", best)
	}
}

func TestAnnotateAbsoluteBoard_PlayerO_PreservesOriginalBoard(t *testing.T) {
	// Board and Player in the result must reflect the original call, not the relabeled state.
	absBoard := makeBoard(0, 0, 0, 0, 1, 0, 0, 0, 0)
	result := annotateAbsoluteBoard(absBoard, O, fullSearchConfig)
	if !slices.Equal(result.Board, absBoard) {
		t.Errorf("Board: got %v, want %v", result.Board, absBoard)
	}
	if result.Player != O {
		t.Errorf("Player: got %v, want O", result.Player)
	}
}

// Tests for WriteAnnotatedBoards and ReadAnnotatedBoards.

func TestWriteReadAnnotatedBoards_Roundtrip(t *testing.T) {
	want := []AnnotatedBoard{
		AnnotateCanonicalBoard(makeBoard(0, 0, 0, 0, 0, 0, 0, 0, 0), X, fullSearchConfig),
		AnnotateCanonicalBoard(makeBoard(1, 1, 0, 2, 2, 0, 0, 0, 0), X, fullSearchConfig),
		AnnotateCanonicalBoard(makeBoard(2, 2, 2, 1, 1, 0, 0, 0, 1), X, fullSearchConfig), // terminal
	}
	path := filepath.Join(t.TempDir(), "boards.ndjson")
	if err := WriteAnnotatedBoards(path, want, false); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := ReadAnnotatedBoards(path, 0, -1, false)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("count: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !slices.Equal(got[i].Board, want[i].Board) {
			t.Errorf("[%d] Board: got %v, want %v", i, got[i].Board, want[i].Board)
		}
		if got[i].Player != want[i].Player {
			t.Errorf("[%d] Player: got %v, want %v", i, got[i].Player, want[i].Player)
		}
		if len(got[i].Scores) != len(want[i].Scores) {
			t.Errorf("[%d] Scores length: got %d, want %d", i, len(got[i].Scores), len(want[i].Scores))
		}
	}
}

func TestWriteAnnotatedBoards_Append(t *testing.T) {
	path := filepath.Join(t.TempDir(), "boards.ndjson")
	batch1 := []AnnotatedBoard{
		AnnotateCanonicalBoard(makeBoard(1, 1, 0, 2, 2, 0, 0, 0, 0), X, fullSearchConfig)}
	batch2 := []AnnotatedBoard{
		AnnotateCanonicalBoard(makeBoard(0, 0, 0, 0, 0, 0, 0, 0, 0), X, fullSearchConfig)}
	if err := WriteAnnotatedBoards(path, batch1, false); err != nil {
		t.Fatalf("first write: %v", err)
	}
	if err := WriteAnnotatedBoards(path, batch2, false); err != nil {
		t.Fatalf("second write: %v", err)
	}
	got, err := ReadAnnotatedBoards(path, 0, -1, false)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != 2 {
		t.Errorf("got %d boards after append, want 2", len(got))
	}
}

func TestWriteAnnotatedBoards_Overwrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "boards.ndjson")
	batch1 := []AnnotatedBoard{
		AnnotateCanonicalBoard(makeBoard(1, 1, 0, 2, 2, 0, 0, 0, 0), X, fullSearchConfig),
		AnnotateCanonicalBoard(makeBoard(0, 0, 0, 0, 0, 0, 0, 0, 0), X, fullSearchConfig),
	}
	batch2 := []AnnotatedBoard{
		AnnotateCanonicalBoard(makeBoard(2, 2, 2, 1, 1, 0, 0, 0, 1), X, fullSearchConfig),
	}
	if err := WriteAnnotatedBoards(path, batch1, false); err != nil {
		t.Fatalf("first write: %v", err)
	}
	if err := WriteAnnotatedBoards(path, batch2, true); err != nil {
		t.Fatalf("overwrite: %v", err)
	}
	got, err := ReadAnnotatedBoards(path, 0, -1, false)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != 1 {
		t.Errorf("got %d boards after overwrite, want 1", len(got))
	}
}

func TestReadAnnotatedBoards_Chunk(t *testing.T) {
	// Write 5 boards distinguishable by a stone at position i, then read lines [1,4).
	var boards []AnnotatedBoard
	for i := range 5 {
		b := makeBoard(0, 0, 0, 0, 0, 0, 0, 0, 0)
		b[i] = X
		boards = append(boards, AnnotateCanonicalBoard(b, X, fullSearchConfig))
	}
	path := filepath.Join(t.TempDir(), "boards.ndjson")
	if err := WriteAnnotatedBoards(path, boards, false); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := ReadAnnotatedBoards(path, 1, 4, false)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("chunk [1,4): got %d boards, want 3", len(got))
	}
	for i, g := range got {
		if !slices.Equal(g.Board, boards[i+1].Board) {
			t.Errorf("chunk[%d] Board: got %v, want %v", i, g.Board, boards[i+1].Board)
		}
	}
}

func TestReadAnnotatedBoards_EmptyRange(t *testing.T) {
	path := filepath.Join(t.TempDir(), "boards.ndjson")
	if err := WriteAnnotatedBoards(path, []AnnotatedBoard{
		AnnotateCanonicalBoard(makeBoard(0, 0, 0, 0, 0, 0, 0, 0, 0), X, fullSearchConfig),
	}, false); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := ReadAnnotatedBoards(path, 3, 3, false)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %d boards for empty range, want 0", len(got))
	}
}

func TestReadAnnotatedBoards_AsOutcomeClasses(t *testing.T) {
	// Mover has stones at 0,1; position 2 wins (positive score), the rest are non-winning replies.
	b := makeBoard(1, 1, 0, 2, 2, 0, 0, 0, 0)
	boards := []AnnotatedBoard{AnnotateCanonicalBoard(b, X, fullSearchConfig)}
	path := filepath.Join(t.TempDir(), "boards.ndjson")
	if err := WriteAnnotatedBoards(path, boards, false); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := ReadAnnotatedBoards(path, 0, -1, true)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d boards, want 1", len(got))
	}
	if len(got[0].Scores) != len(boards[0].Scores) {
		t.Fatalf("Scores length: got %d, want %d", len(got[0].Scores), len(boards[0].Scores))
	}
	for i, sm := range got[0].Scores {
		want := 0.0
		switch {
		case boards[0].Scores[i].Score > 0:
			want = 1
		case boards[0].Scores[i].Score < 0:
			want = -1
		}
		if sm.Score != want {
			t.Errorf("[%d] Score: got %v, want %v (original %v)",
				i, sm.Score, want, boards[0].Scores[i].Score)
		}
		if sm.Pos != boards[0].Scores[i].Pos {
			t.Errorf("[%d] Pos: got %d, want %d", i, sm.Pos, boards[0].Scores[i].Pos)
		}
	}
}
