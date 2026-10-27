package tictactoe

// Unique identity strings for the available AI players.
const (
	IdentityAIMinimax    = "ai-minimax"
	IdentityAIScoreModel = "ai-scoremodel-mlp"
)

// AIPlayer is the common interface for all AI implementations.
// Any type that can pick a move for a given game state satisfies this interface, whether backed
// by minimax, a value network, or any other method.
type AIPlayer interface {
	// Identity returns the unique identity string of this AI, used in logs and stats.
	Identity() string
	// Name returns the human-readable display name shown in menus and logs.
	Name() string
	// Move returns the column and row the AI wants to play on the given game state.
	Move(game Game) (int, int, error)
}
