package tictactoe

import (
	"fmt"
	"slices"
	"strings"
)

// Player represents a player or an empty cell.
type Player int

const (
	Empty Player = 0
	X     Player = 1
	O     Player = 2
)

func (p Player) String() string {
	switch p {
	case X:
		return "X"
	case O:
		return "O"
	default:
		return "-"
	}
}

// GameStatus represents the current game status.
type GameStatus int

const (
	Running GameStatus = iota
	Won
	Draw
)

// Board is a flat array of size size*size, e.g. 9 for a 3x3 board.
type Board []Player

// Less returns true if b is lexicographically smaller the other.
func (b Board) LessThan(other Board) bool {
	for k := range len(b) {
		if b[k] != other[k] {
			return b[k] < other[k]
		}
	}
	return false
}

// BoardKey is a compact, order-independent encoding of a Board's raw cell values, suitable for use
// as a map key. Two boards with identical cell values always produce the same key. Packing one
// base-3 digit per cell fits in a uint64 for boards up to 40 cells (3^40 < 2^64), far beyond
// anything this project reaches.
type BoardKey uint64

// ToKey returns a BoardKey uniquely identifying b, suitable for use as a map key.
func (b Board) ToKey() BoardKey {
	var key BoardKey
	for _, c := range b {
		key = key*3 + BoardKey(c)
	}
	return key
}

// Game represents the complete state of a game.
type Game struct {
	Board        Board
	Size         int
	ConnectN     int // number of marks in a line needed to win; defaults to Size.
	ActivePlayer Player
	Status       GameStatus
	Winner       Player
}

// NewGame creates a new game where connectN equals the board size (standard rules).
func NewGame(size int) Game {
	return NewConnectNGame(size, size)
}

// NewConnectNGame creates a new game with a custom connectN win condition.
func NewConnectNGame(size, connectN int) Game {
	return Game{
		Board:        make(Board, size*size),
		Size:         size,
		ConnectN:     connectN,
		ActivePlayer: X, // X always goes first.
		Status:       Running,
		Winner:       Empty,
	}
}

// GetCell returns the player at the given (x, y) coordinate. (0,0) is top-left.
func (g Game) GetCell(x, y int) Player {
	return g.Board[y*g.Size+x]
}

// posToXY converts a flat board position to (x, y) coordinates.
func (g Game) posToXY(pos int) (x, y int) {
	return pos % g.Size, pos / g.Size
}

// setCell returns a copy of the board with the given (x, y) cell set to p.
func (g Game) setCell(x, y int, p Player) Board {
	next := make(Board, len(g.Board))
	copy(next, g.Board)
	next[y*g.Size+x] = p
	return next
}

// PlaceMark executes a move and returns the new game state. The original state is not modified.
// (0,0) is left-top.
func PlaceMark(game Game, x, y int) (Game, error) {
	if game.Status != Running {
		return game, fmt.Errorf("game already ended")
	}
	if x < 0 || x >= game.Size || y < 0 || y >= game.Size {
		return game, fmt.Errorf("position (%d,%d) invalid (must be 0-%d for each coordinate)",
			x, y, game.Size-1)
	}
	if game.Board[y*game.Size+x] != Empty {
		return game, fmt.Errorf("position (%d,%d) is already occupied", x, y)
	}

	newBoard := game.setCell(x, y, game.ActivePlayer)

	newStatus := Running
	newWinner := Empty
	if winner, ok := checkWinner(newBoard, game.Size, game.ConnectN); ok {
		newStatus = Won
		newWinner = winner
	} else if isDraw(newBoard) {
		newStatus = Draw
	}

	// Next player.
	nextPlayer := O
	if game.ActivePlayer == O {
		nextPlayer = X
	}

	return Game{
		Board:        newBoard,
		Size:         game.Size,
		ConnectN:     game.ConnectN,
		ActivePlayer: nextPlayer,
		Status:       newStatus,
		Winner:       newWinner,
	}, nil
}

// AvailableMoves returns all positions on the board that are not yet occupied.
func AvailableMoves(board Board) []int {
	var moves []int
	for pos, cell := range board {
		if cell == Empty {
			moves = append(moves, pos)
		}
	}
	return moves
}

// checkWinner checks if any player has connectN marks in a line.
func checkWinner(board Board, size, connectN int) (Player, bool) {
	for row := range size {
		for col := 0; col <= size-connectN; col++ {
			if allEqual(board, row*size+col, 1, connectN) {
				return board[row*size+col], true
			}
		}
	}

	for col := range size {
		for row := 0; row <= size-connectN; row++ {
			if allEqual(board, row*size+col, size, connectN) {
				return board[row*size+col], true
			}
		}
	}

	// Diagonals top-left to bottom-right.
	for row := 0; row <= size-connectN; row++ {
		for col := 0; col <= size-connectN; col++ {
			if allEqual(board, row*size+col, size+1, connectN) {
				return board[row*size+col], true
			}
		}
	}

	// Diagonals top-right to bottom-left.
	for row := 0; row <= size-connectN; row++ {
		for col := connectN - 1; col < size; col++ {
			if allEqual(board, row*size+col, size-1, connectN) {
				return board[row*size+col], true
			}
		}
	}

	return Empty, false
}

// allEqual checks if 'count' cells starting at 'start' with step size 'step' are all equal and
// non-empty.
func allEqual(board Board, start int, step int, count int) bool {
	first := board[start]
	if first == Empty {
		return false
	}
	for i := 1; i < count; i++ {
		if board[start+i*step] != first {
			return false
		}
	}
	return true
}

// isDraw checks if the game is a draw (all cells filled, no winner).
func isDraw(board Board) bool {
	return !slices.Contains(board, Empty)
}

// BoardString returns the board as a human-readable string.
func BoardString(game Game) string {
	var result strings.Builder
	for i, cell := range game.Board {
		result.WriteString(cell.String())
		if (i+1)%game.Size == 0 {
			result.WriteString("\n")
		} else {
			result.WriteString(" ")
		}
	}
	return result.String()
}
