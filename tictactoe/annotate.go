package tictactoe

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// AnnotatedBoard holds a board position together with the minimax scores for every legal move from
// that position.
type AnnotatedBoard struct {
	Board  Board
	Player Player // the side to move in absolute terms (X or O)
	Scores []ScoredMove
}

// AnnotateCanonicalBoard runs minimax on a canonical board (mover=1, opponent=2, as returned by
// EnumerateBoards). Terminal boards return empty Scores.
func AnnotateCanonicalBoard(board Board, player Player, config MinimaxConfig) AnnotatedBoard {
	return annotate(board, board, player, config)
}

// annotateAbsoluteBoard runs minimax on an absolute board (X=1, O=2 always, as in Game.Board).
// When player==O the board is relabeled to canonical form before searching. Terminal boards return
// empty Scores.
func annotateAbsoluteBoard(board Board, player Player, config MinimaxConfig) AnnotatedBoard {
	return annotate(board, RelabelToMover(board, player), player, config)
}

// RelabelToMover returns board with mover's marks recoded as X (1) and the opponent's as O (2),
// i.e. the canonical "side to move" encoding used throughout this package. If mover is not O,
// board is returned unchanged (no copy made) -- mover is expected to be X or O; any other value is
// treated like X (no relabeling), matching this function's original inline form.
func RelabelToMover(board Board, mover Player) Board {
	if mover != O {
		return board
	}
	relabeled := make(Board, len(board))
	for i, p := range board {
		if p != Empty {
			relabeled[i] = switchPlayer(p)
		}
	}
	return relabeled
}

// AnnotateCanonicalBoards annotates each board with minimax scores for player using config.
func AnnotateCanonicalBoards(boards []Board, player Player, config MinimaxConfig) []AnnotatedBoard {
	annotated := make([]AnnotatedBoard, len(boards))
	for i, b := range boards {
		annotated[i] = AnnotateCanonicalBoard(b, player, config)
	}
	return annotated
}

func annotate(original, canonical Board, player Player, config MinimaxConfig) AnnotatedBoard {
	size := int(math.Round(math.Sqrt(float64(len(canonical)))))
	status, winner := Running, Empty
	if w, won := checkWinner(canonical, size, size); won {
		status, winner = Won, w
	} else if isDraw(canonical) {
		status = Draw
	}
	if status != Running {
		return AnnotatedBoard{Board: original, Player: player}
	}
	game := Game{
		Board:        canonical,
		Size:         size,
		ConnectN:     size,
		ActivePlayer: X, // canonical: mover is always 1 = X
		Status:       status,
		Winner:       winner,
	}
	result, _ := BestMoves(game, config)
	return AnnotatedBoard{Board: original, Player: player, Scores: result.Moves}
}

// WriteAnnotatedBoards writes boards as NDJSON to path, one board per line.
// By default it appends to an existing file; pass overwrite=true to truncate first.
func WriteAnnotatedBoards(path string, boards []AnnotatedBoard, overwrite bool) error {
	flags := os.O_APPEND | os.O_CREATE | os.O_WRONLY
	if overwrite {
		flags = os.O_TRUNC | os.O_CREATE | os.O_WRONLY
	}
	f, err := os.OpenFile(path, flags, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, b := range boards {
		if err := enc.Encode(b); err != nil {
			return err
		}
	}
	return nil
}

// ReadAnnotatedBoards reads AnnotatedBoard NDJSON from path and returns entries in [from, to).
// Lines are 0-indexed; pass to=-1 to read until EOF.
// When asOutcomeClasses is true, each score is reduced to its outcome class via its sign: 1 for a
// win (score > 0), -1 for a loss (score < 0), 0 for a draw (score == 0), turning the regression
// target into a coarse win/draw/loss target.
func ReadAnnotatedBoards(path string, from, to int, asOutcomeClasses bool) ([]AnnotatedBoard, error) {
	if to >= 0 && from >= to {
		return nil, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var boards []AnnotatedBoard
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	i := 0
	for scanner.Scan() {
		if to >= 0 && i >= to {
			break
		}
		if i >= from {
			var ab AnnotatedBoard
			if err := json.Unmarshal(scanner.Bytes(), &ab); err != nil {
				return nil, fmt.Errorf("line %d: %w", i, err)
			}
			if asOutcomeClasses {
				toOutcomeClasses(ab.Scores)
			}
			boards = append(boards, ab)
		}
		i++
	}
	return boards, scanner.Err()
}

// toOutcomeClasses replaces each score in place with its outcome class (its sign: 1, -1, or 0).
func toOutcomeClasses(scores []ScoredMove) {
	for i, s := range scores {
		switch {
		case s.Score > 0:
			scores[i].Score = 1
		case s.Score < 0:
			scores[i].Score = -1
		default:
			scores[i].Score = 0
		}
	}
}
