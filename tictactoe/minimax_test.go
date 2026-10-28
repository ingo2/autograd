package tictactoe

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

func TestMinimax_TerminalStates(t *testing.T) {
	// AI is X (X moves first, so ActivePlayer=X at game start).
	baseGame := NewGame(3)
	m := newMinimax(baseGame, MinimaxConfig{}, nil)

	// X wins: top row
	// X X X
	// O O -
	// - - -
	xWins := NewGame(3)
	xWins, _ = PlaceMark(xWins, 0, 0) // X
	xWins, _ = PlaceMark(xWins, 0, 1) // O
	xWins, _ = PlaceMark(xWins, 1, 0) // X
	xWins, _ = PlaceMark(xWins, 1, 1) // O
	xWins, _ = PlaceMark(xWins, 2, 0) // X wins
	if score := m.evaluate(xWins, 0, -math.MaxFloat64, math.MaxFloat64); score <= 0 {
		t.Errorf("expected positive score for AI (X) victory, got %v", score)
	}

	// O wins:
	// - - X
	// O O O
	// X - -
	oWins := NewGame(3)
	oWins, _ = PlaceMark(oWins, 0, 2) // X
	oWins, _ = PlaceMark(oWins, 0, 1) // O
	oWins, _ = PlaceMark(oWins, 2, 0) // X
	oWins, _ = PlaceMark(oWins, 1, 1) // O
	oWins, _ = PlaceMark(oWins, 2, 2) // X
	oWins, _ = PlaceMark(oWins, 2, 1) // O wins
	if score := m.evaluate(oWins, 0, -math.MaxFloat64, math.MaxFloat64); score >= 0 {
		t.Errorf("expected negative score for AI (X) defeat, got %v", score)
	}

	// Draw:
	// X O X
	// O X O
	// O X O
	draw := NewGame(3)
	for _, mv := range [][2]int{{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {0, 2}, {2, 1}, {2, 2},
		{1, 2}} {
		draw, _ = PlaceMark(draw, mv[0], mv[1])
	}
	if score := m.evaluate(draw, 0, -math.MaxFloat64, math.MaxFloat64); score != 0 {
		t.Errorf("expected 0 for draw, got %v", score)
	}
}

// A win found sooner (lower depth) must score higher than a later win.
func TestMinimax_WinScoreDecreasesWithDepth(t *testing.T) {
	game := NewGame(3)
	m := newMinimax(game, MinimaxConfig{Gamma: 0.9}, nil)

	xWins := NewGame(3)
	xWins, _ = PlaceMark(xWins, 0, 0) // X
	xWins, _ = PlaceMark(xWins, 0, 1) // O
	xWins, _ = PlaceMark(xWins, 1, 0) // X
	xWins, _ = PlaceMark(xWins, 1, 1) // O
	xWins, _ = PlaceMark(xWins, 2, 0) // X wins

	early, _ := m.terminalScore(xWins, 1)
	late, _ := m.terminalScore(xWins, 5)
	if early <= late {
		t.Errorf("win at depth 1 (%v) should score higher than win at depth 5 (%v)", early, late)
	}
}

// A loss found later (higher depth) must score higher than an early loss.
func TestMinimax_LossScoreIncreasesWithDepth(t *testing.T) {
	game := NewGame(3)
	m := newMinimax(game, MinimaxConfig{Gamma: 0.9}, nil)

	oWins := NewGame(3)
	oWins, _ = PlaceMark(oWins, 0, 2) // X
	oWins, _ = PlaceMark(oWins, 0, 1) // O
	oWins, _ = PlaceMark(oWins, 2, 0) // X
	oWins, _ = PlaceMark(oWins, 1, 1) // O
	oWins, _ = PlaceMark(oWins, 2, 2) // X
	oWins, _ = PlaceMark(oWins, 2, 1) // O wins

	early, _ := m.terminalScore(oWins, 1)
	late, _ := m.terminalScore(oWins, 5)
	if late <= early {
		t.Errorf("loss at depth 5 (%v) should score higher than loss at depth 1 (%v)", late, early)
	}
}

func TestBestMoves_TakesWin(t *testing.T) {
	// X to play, wins at (2,0) = pos 2.
	// X X -
	// O O -
	// - - -
	game := NewGame(3)
	game, _ = PlaceMark(game, 0, 0) // X
	game, _ = PlaceMark(game, 0, 1) // O
	game, _ = PlaceMark(game, 1, 0) // X
	game, _ = PlaceMark(game, 1, 1) // O
	// X's turn.
	res, err := BestMoves(game, MinimaxConfig{Gamma: 0.9})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	best := BestPositions(res.Moves)
	if len(best) != 1 || best[0] != 2 {
		t.Errorf("expected only winning move pos 2, got %v", best)
	}
	if res.Nodes == 0 {
		t.Error("Nodes must be > 0")
	}
	if res.Duration < 0 {
		t.Error("Duration must be non-negative")
	}
}

func TestBestMoves_BlocksWin(t *testing.T) {
	// X to play, must block O at (2,0) = pos 2.
	// X has no immediate winning move, so it must block.
	// O O -
	// X - -
	// - - X
	game := NewGame(3)
	game, _ = PlaceMark(game, 0, 1) // X
	game, _ = PlaceMark(game, 0, 0) // O
	game, _ = PlaceMark(game, 2, 2) // X
	game, _ = PlaceMark(game, 1, 0) // O
	// X's turn: O threatens (2,0) to complete top row.
	res, err := BestMoves(game, MinimaxConfig{Gamma: 0.9})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !slices.Contains(BestPositions(res.Moves), 2) {
		t.Errorf("expected blocking move pos 2 in optimal set, got %v", BestPositions(res.Moves))
	}
	if res.Nodes == 0 {
		t.Error("Nodes must be > 0")
	}
	if res.Duration < 0 {
		t.Error("Duration must be non-negative")
	}
}

// With maxDepth=1 the AI can only see one move ahead, so it takes an immediate
// win but cannot look further to find non-obvious threats.
func TestBestMoves_MaxDepthLimitsSearch(t *testing.T) {
	// X to play; (2,0) = pos 2 wins immediately.
	// X X -
	// O O -
	// - - -
	game := NewGame(3)
	game, _ = PlaceMark(game, 0, 0) // X
	game, _ = PlaceMark(game, 0, 1) // O
	game, _ = PlaceMark(game, 1, 0) // X
	game, _ = PlaceMark(game, 1, 1) // O

	res, err := BestMoves(game, MinimaxConfig{MaxDepth: 1, Gamma: 0.9})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Immediate win must still be found even with maxDepth=1.
	best := BestPositions(res.Moves)
	if len(best) != 1 || best[0] != 2 {
		t.Errorf("expected only winning move pos 2, got %v", best)
	}
	if res.Depth != 1 {
		t.Errorf("Depth: got %d, want 1 (MaxDepth was 1)", res.Depth)
	}
	// 5 empty cells, depth-1: one evaluate() call per candidate move.
	if res.Nodes != 5 {
		t.Errorf("Nodes: got %d, want 5 (one per empty cell at depth 1)", res.Nodes)
	}
}

// Two perfect players must always draw on a 3x3 board.
func TestBestMoves_PerfectPlayAlwaysDraws(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	game := NewGame(3)
	cfg := MinimaxConfig{Gamma: 0.9}
	for game.Status == Running {
		res, err := BestMoves(game, cfg)
		if err != nil {
			t.Fatalf("BestMoves error: %v", err)
		}
		best := BestPositions(res.Moves)
		pos := best[rng.Intn(len(best))]
		game, _ = PlaceMark(game, pos%game.Size, pos/game.Size)
	}
	if game.Status != Draw {
		t.Errorf("expected draw with perfect play on both sides, got status=%v winner=%v",
			game.Status, game.Winner)
	}
}

func TestEstimatedNodeCount(t *testing.T) {
	tests := []struct {
		name     string
		size     int
		maxDepth int
		want     float64
	}{
		{"3x3 unlimited", 3, 0, 986_409},
		{"4x4 depth-6", 4, 6, 6_337_216},
		{"5x5 depth-6", 5, 6, 134_205_625},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			game := NewGame(tc.size)
			got := EstimatedNodeCount(game, MinimaxConfig{MaxDepth: tc.maxDepth})
			if got != tc.want {
				t.Errorf("got %.0f, want %.0f", got, tc.want)
			}
		})
	}
}

// TestMinimax_NodeCounting verifies that evaluate() increments the node counter.
// With MaxDepth=1 on an empty 3x3 board each of the 9 candidate moves triggers exactly one
// evaluate() call that hits the depth cap immediately -> 9 nodes total.
// A search at MaxDepth=2 must visit strictly more nodes than MaxDepth=1.
func TestMinimax_NodeCounting(t *testing.T) {
	game := NewGame(3)
	cfg := MinimaxConfig{Gamma: 0.9}

	countNodes := func(maxDepth int) int {
		var nodes int
		cfg.MaxDepth = maxDepth
		m := newMinimax(game, cfg, &nodes)
		for _, pos := range AvailableMoves(game.Board) {
			x, y := game.posToXY(pos)
			next, _ := PlaceMark(game, x, y)
			m.evaluate(next, 1, -math.MaxFloat64, math.MaxFloat64)
		}
		return nodes
	}

	d1 := countNodes(1)
	d2 := countNodes(2)

	if d1 != 9 {
		t.Errorf("MaxDepth=1 on empty board: expected 9 nodes, got %d", d1)
	}
	if d2 <= d1 {
		t.Errorf("MaxDepth=2 (%d nodes) should visit more nodes than MaxDepth=1 (%d nodes)", d2, d1)
	}
}

// TestBestMoves_NodeBudget uses the blocking position from TestBestMoves_BlocksWin:
//
//	O O -
//	X - -
//	- - X
//
// 5 empty cells remain. At depth 1 all five score as draw (O's threat at pos 2 is invisible).
// At depth 2+ the search sees O would win immediately at pos 2 -> only pos 2 is returned.
func TestBestMoves_NodeBudget(t *testing.T) {
	setup := func() Game {
		game := NewGame(3)
		game, _ = PlaceMark(game, 0, 1) // X
		game, _ = PlaceMark(game, 0, 0) // O
		game, _ = PlaceMark(game, 2, 2) // X
		game, _ = PlaceMark(game, 1, 0) // O
		return game
	}
	cfg := MinimaxConfig{Gamma: 0.9}

	// Budget = 5: exactly the nodes consumed by depth-1 (one evaluate call per empty cell).
	// After depth 1 the budget is exhausted; depth 2 is skipped.
	shallow, err :=
		BestMoves(setup(), MinimaxConfig{Gamma: 0.9, NodeBudget: 5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(shallow.Moves) != 5 {
		t.Errorf("depth-1 budget: expected all 5 available moves, got %v", shallow.Moves)
	}
	if shallow.Depth != 1 {
		t.Errorf("depth-1 budget: Depth got %d, want 1", shallow.Depth)
	}
	if shallow.Nodes != 5 {
		t.Errorf("depth-1 budget: Nodes got %d, want 5", shallow.Nodes)
	}

	// Budget = 1000000: deeper than any 3x3 search; result must match unlimited.
	unlimited, _ := BestMoves(setup(), cfg)
	large, err :=
		BestMoves(setup(), MinimaxConfig{Gamma: 0.9, NodeBudget: 1_000_000})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !slices.Equal(BestPositions(unlimited.Moves), BestPositions(large.Moves)) {
		t.Errorf("large budget should match unlimited: unlimited=%v large=%v",
			BestPositions(unlimited.Moves), BestPositions(large.Moves))
	}
	if large.Depth <= 1 {
		t.Errorf("large budget: Depth got %d, want > 1", large.Depth)
	}
	if large.Nodes == 0 {
		t.Error("large budget: Nodes must be > 0")
	}
}

func TestBestMoves_NoMovesAvailable(t *testing.T) {
	game := NewGame(3)
	for _, mv := range [][2]int{{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {0, 2}, {2, 1}, {2, 2},
		{1, 2}} {
		game, _ = PlaceMark(game, mv[0], mv[1])
	}

	_, err := BestMoves(game, MinimaxConfig{Gamma: 0.9})

	if err == nil {
		t.Error("expected error when no moves available")
	}
}

// TestMinimaxResult_Sanity checks all four MinimaxResult fields and verifies that iterative-
// deepening depth grows monotonically with budget, using a 5-cell position where node counts
// are exact and predictable.
//
// Position (X to move, 5 empty cells):
//
//	O O -
//	X - -
//	- - X
func TestMinimaxResult_Sanity(t *testing.T) {
	setup := func() Game {
		game := NewGame(3)
		game, _ = PlaceMark(game, 0, 1) // X
		game, _ = PlaceMark(game, 0, 0) // O
		game, _ = PlaceMark(game, 2, 2) // X
		game, _ = PlaceMark(game, 1, 0) // O
		return game
	}
	cfg1 := MinimaxConfig{MaxDepth: 1, Gamma: 0.9}
	cfgUnlim := MinimaxConfig{Gamma: 0.9}

	// Direct search, MaxDepth=1: one evaluate() call per empty cell -> 5 nodes.
	t.Run("direct/depth=1/nodes=5", func(t *testing.T) {
		res, err := BestMoves(setup(), cfg1)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.Depth != 1 {
			t.Errorf("Depth: got %d, want 1", res.Depth)
		}
		if res.Nodes != 5 {
			t.Errorf("Nodes: got %d, want 5 (one per empty cell at depth 1)", res.Nodes)
		}
		if len(res.Moves) == 0 {
			t.Error("Moves must not be empty")
		}
		if res.Duration < 0 {
			t.Error("Duration must be non-negative")
		}
	})

	// Direct search, MaxDepth=0 (unlimited): Depth reports the board size as theoretical ceiling.
	t.Run("direct/unlimited/depth=board-size", func(t *testing.T) {
		res, err := BestMoves(setup(), cfgUnlim)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.Depth != 9 { // 3x3 board has 9 cells
			t.Errorf("Depth: got %d, want 9 (board size)", res.Depth)
		}
		if res.Nodes == 0 {
			t.Error("Nodes must be > 0")
		}
	})

	// Iterative deepening: depth must increase monotonically with budget.
	// Budget = 5: depth-1 consumes exactly 5 nodes; the pre-depth-2 check sees nodes(5) >= budget(5)
	// and stops => Depth = 1.
	// Budget = 6: the pre-depth-2 check sees nodes(5) < budget(6) => depth 2 starts and completes
	// => Depth >= 2.
	t.Run("iter/depth-monotone", func(t *testing.T) {
		shallow, err :=
			BestMoves(setup(), MinimaxConfig{NodeBudget: 5, Gamma: 0.9})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if shallow.Depth != 1 {
			t.Errorf("budget=5: want Depth=1, got %d", shallow.Depth)
		}
		if shallow.Nodes != 5 {
			t.Errorf("budget=5: want Nodes=5, got %d", shallow.Nodes)
		}

		deeper, err :=
			BestMoves(setup(), MinimaxConfig{NodeBudget: 6, Gamma: 0.9})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if deeper.Depth < 2 {
			t.Errorf("budget=6: want Depth>=2, got %d (Nodes=%d)", deeper.Depth, deeper.Nodes)
		}
		if shallow.Depth >= deeper.Depth {
			t.Errorf("depth should increase with budget: budget=5->Depth=%d, budget=6->Depth=%d",
				shallow.Depth, deeper.Depth)
		}
	})

	// MinDepth overrides the budget check: with budget=5 the search normally stops after depth 1
	// (5 nodes consumed == budget). Setting MinDepth=2 must force depth 2 to complete regardless.
	t.Run("iter/min-depth-overrides-budget", func(t *testing.T) {
		withoutMin, err :=
			BestMoves(setup(), MinimaxConfig{NodeBudget: 5, Gamma: 0.9})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if withoutMin.Depth != 1 {
			t.Errorf("baseline: budget=5 MinDepth=0: want Depth=1, got %d", withoutMin.Depth)
		}

		withMin, err :=
			BestMoves(setup(),
				MinimaxConfig{NodeBudget: 5, MinDepth: 2, Gamma: 0.9})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if withMin.Depth < 2 {
			t.Errorf("budget=5 MinDepth=2: want Depth>=2, got %d (Nodes=%d)", withMin.Depth, withMin.Nodes)
		}
		if withMin.Nodes <= withoutMin.Nodes {
			t.Errorf("MinDepth=2 should consume more nodes than MinDepth=0: got %d vs %d",
				withMin.Nodes, withoutMin.Nodes)
		}
	})

	// Iterative deepening, large budget: must reach depth > 1 and agree with unlimited search.
	t.Run("iter/large-budget/matches-unlimited", func(t *testing.T) {
		unlimited, _ := BestMoves(setup(), cfgUnlim)
		res, err :=
			BestMoves(setup(), MinimaxConfig{NodeBudget: 1_000_000, Gamma: 0.9})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.Depth <= 1 {
			t.Errorf("Depth: got %d, want > 1 with large budget", res.Depth)
		}
		if !slices.Equal(BestPositions(unlimited.Moves), BestPositions(res.Moves)) {
			t.Errorf("Moves mismatch: unlimited=%v budgeted=%v",
				BestPositions(unlimited.Moves), BestPositions(res.Moves))
		}
	})
}
