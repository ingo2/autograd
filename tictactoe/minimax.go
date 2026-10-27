package tictactoe

import (
	"fmt"
	"math"
	"time"
)

// MinimaxConfig holds the pure search parameters for the minimax algorithm.
//
// Gamma defines the depth-discount formula:
//
//	win score at depth d  =  math.Pow(Gamma, d)  // in (0, 1]
//	loss score at depth d = -math.Pow(Gamma, d)  // in [-1, 0)
//
// So faster wins score higher and later losses score higher.
type MinimaxConfig struct {
	MinDepth   int     // 0 = no minimum; NodeBudget is not checked before this depth is reached
	MaxDepth   int     // 0 = unlimited
	NodeBudget int     // 0 = infinite budget (iterative deepening)
	Gamma      float64 // per-ply discount rate (plan default 0.9); 1.0 = no discount
}

// DefaultMinimaxConfig returns an exhaustive search with the plan's default discount rate.
func DefaultMinimaxConfig() MinimaxConfig {
	return MinimaxConfig{Gamma: 0.9}
}

// MinimaxResult is returned by BestMoves and carries the search outcome alongside diagnostics.
type MinimaxResult struct {
	Moves    []ScoredMove  // all moves sorted best-to-worst; ties at the top are equally optimal
	Nodes    int           // total evaluate() calls made during the search
	Depth    int           // deepest completed search (MaxDepth cap or last iter-deepening depth)
	Duration time.Duration // wall-clock time of the search
}

// ScoredMove pairs a board position with its minimax score.
type ScoredMove struct {
	Pos   int
	Score float64
}

// minimax holds the reference game state and search parameters.
type minimax struct {
	game Game
	MinimaxConfig
	nodes *int // counts evaluate() calls; nil = no tracking
}

func newMinimax(game Game, config MinimaxConfig, nodes *int) minimax {
	return minimax{
		game:          game,
		MinimaxConfig: config,
		nodes:         nodes}
}

// EstimatedNodeCount returns a rough upper-bound on the number of evaluate() calls that BestMoves
// would make for the given game and config. It sums the falling permutations
// P(n, d) = n*(n-1)*...*(n-d+1) for each depth level d, where n is the number of empty cells. This
// assumes neither alpha-beta pruning nor early game termination, so the actual count will be
// significantly lower.
func EstimatedNodeCount(game Game, cfg MinimaxConfig) float64 {
	n := 0
	for _, cell := range game.Board {
		if cell == Empty {
			n++
		}
	}
	depth := n
	if cfg.MaxDepth > 0 && cfg.MaxDepth < depth {
		depth = cfg.MaxDepth
	}
	var total, product float64
	product = 1
	for d := 1; d <= depth; d++ {
		product *= float64(n - d + 1)
		total += product
	}
	return total
}

// BestMoves returns all flat board positions that are equally optimal for the active player.
// When NodeBudget > 0 it uses iterative deepening: it searches depth 1, 2, 3, ... and stops before
// starting a new depth once the cumulative node count reaches the budget. The result from the last
// fully searched depth is returned, so the output is always consistent.
func BestMoves(game Game, cfg MinimaxConfig) (MinimaxResult, error) {
	moves := AvailableMoves(game.Board)
	if len(moves) == 0 {
		return MinimaxResult{}, fmt.Errorf("no moves available")
	}
	if cfg.NodeBudget == 0 {
		return searchDirect(game, cfg, moves), nil
	}
	return searchIterative(game, cfg, moves), nil
}

// BestPositions returns all positions tied at the highest score in a sorted ScoredMove slice.
func BestPositions(scored []ScoredMove) []int {
	best := make([]int, 0, len(scored))
	for _, sm := range scored {
		if sm.Score < scored[0].Score {
			break
		}
		best = append(best, sm.Pos)
	}
	return best
}

// searchDirect runs a single minimax search to the configured depth (or unlimited if MaxDepth=0).
func searchDirect(game Game, cfg MinimaxConfig, moves []int) MinimaxResult {
	start := time.Now()
	var nodes int
	m := newMinimax(game, cfg, &nodes)
	scored := m.scoreMoves(moves)
	depth := cfg.MaxDepth
	if depth == 0 {
		depth = len(game.Board) // Maximum possible depth that can be reached on the given board.
	}
	return MinimaxResult{Moves: scored, Nodes: nodes, Depth: depth, Duration: time.Since(start)}
}

// searchIterative runs iterative deepening: it searches depth 1, 2, 3, ... and stops before
// starting a new depth once the cumulative node count reaches the budget. The result from the last
// fully searched depth is returned, so the output is always consistent.
func searchIterative(game Game, cfg MinimaxConfig, moves []int) MinimaxResult {
	start := time.Now()
	var nodes int
	maxDepth := cfg.MaxDepth
	if maxDepth == 0 {
		maxDepth = len(game.Board) // Absolute ceiling: one move per cell.
	}
	minDepth := cfg.MinDepth
	if minDepth >= maxDepth {
		minDepth = maxDepth - 1 // Budget check must fire at least at the last depth.
	}
	var scored []ScoredMove
	var reachedDepth int
	// orderedMoves tracks the position sequence sorted by the previous depth's scores, giving
	// alpha-beta ideally a tight initial bound at each new depth.
	orderedMoves := moves
	for depth := 1; depth <= maxDepth; depth++ {
		if depth > minDepth && nodes >= cfg.NodeBudget {
			break
		}
		iterCfg := MinimaxConfig{MaxDepth: depth, Gamma: cfg.Gamma}
		m := newMinimax(game, iterCfg, &nodes)
		scored = m.scoreMoves(orderedMoves)
		reachedDepth = depth
		orderedMoves = make([]int, len(scored))
		for i, sm := range scored {
			orderedMoves[i] = sm.Pos
		}
	}
	return MinimaxResult{Moves: scored, Nodes: nodes, Depth: reachedDepth, Duration: time.Since(start)}
}

// scoreMoves evaluates every candidate position and returns them sorted best-to-worst.
func (m minimax) scoreMoves(moves []int) []ScoredMove {
	scored := make([]ScoredMove, len(moves))
	for i, pos := range moves {
		x, y := m.game.posToXY(pos)
		nextGame, _ := PlaceMark(m.game, x, y)
		scored[i] = ScoredMove{pos, m.evaluate(nextGame, 1, -math.MaxFloat64, math.MaxFloat64)}
	}
	// Insertion sort in descending order.
	for i := 1; i < len(scored); i++ {
		for j := i; j > 0 && scored[j].Score > scored[j-1].Score; j-- {
			scored[j], scored[j-1] = scored[j-1], scored[j]
		}
	}
	return scored
}

// evaluate recursively scores all possible game states from the maximizing player's perspective.
// depth tracks how many moves deep we are, so the AI can prefer quicker wins and delayed losses
// if specified so in the MinimaxConfig. When the MaxDepth is reached without a terminal state, the
// position is scored as a draw. Worst case is O(k!) where k is the number of empty cells.
//
// Returns the minimax score of the position: positive = maximizer wins, negative = minimizer wins,
// zero = draw or depth-limit cutoff. Scores are in [-1, 1].
//
// Alpha-beta pruning: alpha and beta track the best scores each player can already guarantee
// elsewhere in the tree. Any result outside [alpha, beta] is irrelevant - one of the players
// would have steered the game to a better option already found. The window only ever shrinks
// (alpha rises, beta falls); once it collapses (beta <= alpha) the subtree is pruned.
//
// alpha: best score the maximizer can already guarantee on some other branch.
// beta:  best score the minimizer can already guarantee on some other branch.
func (m minimax) evaluate(game Game, depth int, alpha, beta float64) float64 {
	if m.nodes != nil {
		*m.nodes++
	}
	// Terminal check must come before the depth limit: a won/lost/drawn position must always be
	// scored correctly even if we hit the search horizon at the same depth.
	if score, ok := m.terminalScore(game, depth); ok {
		return score
	}
	if m.MaxDepth > 0 && depth >= m.MaxDepth {
		return 0
	}

	moves := AvailableMoves(game.Board)
	if len(moves) == 0 {
		return 0
	}

	if game.ActivePlayer == m.game.ActivePlayer {
		// Maximizing player: raise alpha, prune when alpha >= beta.
		best := -math.MaxFloat64
		for _, pos := range moves {
			x, y := game.posToXY(pos)
			nextGame, e := PlaceMark(game, x, y)
			if e != nil {
				panic(fmt.Sprintf("unexpected error placing mark at (%d,%d): %v", x, y, e))
			}
			best = max(best, m.evaluate(nextGame, depth+1, alpha, beta))
			alpha = max(alpha, best)
			if alpha >= beta {
				// Minimizer has a better option in an already visited sub-tree.
				break
			}
		}
		return best
	}

	// Minimizing player: lower beta, prune when beta <= alpha.
	best := math.MaxFloat64
	for _, pos := range moves {
		x, y := game.posToXY(pos)
		nextGame, e := PlaceMark(game, x, y)
		if e != nil {
			panic(fmt.Sprintf("unexpected error placing mark at (%d,%d): %v", x, y, e))
		}
		best = min(best, m.evaluate(nextGame, depth+1, alpha, beta))
		beta = min(beta, best)
		if beta <= alpha {
			// Maximizer has a better option in an already visited sub-tree.
			break
		}
	}
	return best
}

// terminalScore checks if the game is finished and returns the depth-discounted score,
// computed as +/-math.Pow(Gamma, depth).
func (m minimax) terminalScore(game Game, depth int) (float64, bool) {
	switch game.Status {
	case Won:
		if game.Winner == m.game.ActivePlayer {
			return math.Pow(m.Gamma, float64(depth)), true
		}
		return -math.Pow(m.Gamma, float64(depth)), true
	case Draw:
		return 0, true
	}
	return 0, false
}
