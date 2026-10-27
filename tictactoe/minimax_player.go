package tictactoe

import (
	"math/rand"
)

// MinimaxPlayer is an AIPlayer backed by minimax search. With an unlimited search it never loses
// and always wins if possible, goes for fast wins and delays losses as much as possible. Set
// NodeBudget on boards larger than 3x3, where unlimited depth is intractable.
type MinimaxPlayer struct {
	Rand *rand.Rand // nil = use a new auto-seeded Rand per call
	MinimaxConfig
}

func (p MinimaxPlayer) Identity() string { return IdentityAIMinimax }
func (p MinimaxPlayer) Name() string     { return "Deep Thought" }

// rng returns the configured Rand, falling back to a new auto-seeded Rand.
func (p MinimaxPlayer) rng() *rand.Rand {
	if p.Rand != nil {
		return p.Rand
	}
	return rand.New(rand.NewSource(rand.Int63()))
}

// Move returns the AI's chosen move for the given game state, picked at random from the full set
// of equally-optimal moves.
func (p MinimaxPlayer) Move(game Game) (int, int, error) {
	rng := p.rng()
	result, err := BestMoves(game, p.MinimaxConfig)
	if err != nil {
		return 0, 0, err
	}
	best := BestPositions(result.Moves)
	pos := best[rng.Intn(len(best))]
	return pos % game.Size, pos / game.Size, nil
}
