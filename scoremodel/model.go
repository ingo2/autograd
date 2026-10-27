package scoremodel

import (
	"autograd/nn"
	"autograd/tictactoe"
	"math/rand"
)

// Score Regression using a masked MLP.
const (
	inputLayerSize  = 18
	outputLayerSize = 9
)

// NewScoreModel returns an MLP for score regression with the given hidden layer sizes.
// The input layer is always 18 (two one-hot planes) and the output layer is always 9 (one score
// per cell). Example: hiddenSizes = []int{32} gives architecture [18, 32, 9].
// activationHidden is used for every hidden layer, activationOut for the output layer.
func NewScoreModel(rng *rand.Rand, hiddenSizes []int, activationHidden, activationOut nn.ActivationFunc) *nn.MLP {
	sizes := make([]int, 0, 2+len(hiddenSizes))
	sizes = append(sizes, inputLayerSize)
	sizes = append(sizes, hiddenSizes...)
	sizes = append(sizes, outputLayerSize)
	activations := make([]nn.ActivationFunc, len(sizes)-1)
	for i := range activations {
		activations[i] = activationHidden
	}
	activations[len(activations)-1] = activationOut
	return nn.NewMLP(rng, sizes, activations)
}

// boardToInput encodes a board as two one-hot planes, each as wide as the board has cells (so
// 0-8 and 9-17 for a 3x3 board). The pieces of sideToMove go in plane 0, the opponent's pieces in
// plane 1. Keeping the side to move in plane 0 is the canonical encoding the model is trained on.
func boardToInput(board tictactoe.Board, sideToMove tictactoe.Player) []*nn.Value {
	cells := len(board)
	input := make([]*nn.Value, 2*cells)
	for k := range input {
		input[k] = nn.NewValue(0.0)
	}
	opponent := tictactoe.Player(3 - int(sideToMove)) // X(1)<->O(2)
	for pos, cell := range board {
		switch cell {
		case sideToMove:
			input[pos] = nn.NewValue(1.0)
		case opponent:
			input[cells+pos] = nn.NewValue(1.0)
		}
	}
	return input
}

// BoardToScoreModelInput encodes a dataset board. Dataset boards are perspective-canonical, so the
// side to move is always X and its pieces land in plane 0.
func BoardToScoreModelInput(ab *tictactoe.AnnotatedBoard) []*nn.Value {
	return boardToInput(ab.Board, tictactoe.X)
}

// BoardsToScoreModelInputs calls BoardToScoreModelInput for each board.
func BoardsToScoreModelInputs(annotatedBoards []tictactoe.AnnotatedBoard) [][]*nn.Value {
	inputs := make([][]*nn.Value, 0, len(annotatedBoards))
	for _, ab := range annotatedBoards {
		inputs = append(inputs, BoardToScoreModelInput(&ab))
	}
	return inputs
}

// GameToScoreModelInput encodes a live game board, where X and O are absolute: the active player's
// pieces go in plane 0, so the encoding matches the canonical one the model was trained on.
func GameToScoreModelInput(game tictactoe.Game) []*nn.Value {
	return boardToInput(game.Board, game.ActivePlayer)
}

// BoardToScoreModelOutput returns a length-9 target vector with the minimax score at each valid
// move position and 0.0 at occupied cells.
func BoardToScoreModelOutput(ab *tictactoe.AnnotatedBoard) []float64 {
	output := make([]float64, outputLayerSize)
	for k := range outputLayerSize {
		output[k] = 0.0
	}
	for _, scoredMove := range ab.Scores {
		output[scoredMove.Pos] = scoredMove.Score
	}
	return output
}

// BoardsToScoreModelOutputs calls BoardToScoreModelOutput for each board.
func BoardsToScoreModelOutputs(annotatedBoards []tictactoe.AnnotatedBoard) [][]float64 {
	outputs := make([][]float64, 0, len(annotatedBoards))
	for _, ab := range annotatedBoards {
		outputs = append(outputs, BoardToScoreModelOutput(&ab))
	}
	return outputs
}
