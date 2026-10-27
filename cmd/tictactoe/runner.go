package main

import (
	"autograd/tictactoe"
	"bufio"
	"fmt"
	"io"
	"strings"
)

// Runner plays a single game of Tic-Tac-Toe, reading human moves from in and writing output to out.
// ai is called each turn to produce the AI's move.
type Runner struct {
	in        io.Reader
	out       io.Writer
	ai        tictactoe.AIPlayer
	boardSize int
	connectN  int
}

// NewRunner returns a Runner backed by the given AI player.
func NewRunner(in io.Reader, out io.Writer, ai tictactoe.AIPlayer, boardSize, connectN int) Runner {
	return Runner{
		in:        in,
		out:       out,
		ai:        ai,
		boardSize: boardSize,
		connectN:  connectN,
	}
}

// Play runs the game loop for the given human player mark and returns the finished game.
func (r Runner) Play(humanPlayer tictactoe.Player) (tictactoe.Game, error) {
	game := tictactoe.NewConnectNGame(r.boardSize, r.connectN)
	scanner := bufio.NewScanner(r.in)

	fmt.Fprintf(r.out, "Enter moves as: row col  (both 0-%d, left-top is 0 0)\n", r.boardSize-1)

	for game.Status == tictactoe.Running {
		r.printBoard(game)

		if game.ActivePlayer == humanPlayer {
			next, err := r.readHumanMove(game, scanner)
			if err != nil {
				return game, err
			}
			game = next
		} else {
			x, y, err := r.ai.Move(game)
			if err != nil {
				return game, fmt.Errorf("AI error: %w", err)
			}
			fmt.Fprintf(r.out, "AI plays: %d %d\n", y, x)
			next, err := tictactoe.PlaceMark(game, x, y)
			if err != nil {
				return game, fmt.Errorf("AI placed invalid move: %w", err)
			}
			game = next
		}
	}

	return game, nil
}

// readHumanMove prompts until the player enters a valid move and returns the resulting game state.
func (r Runner) readHumanMove(game tictactoe.Game, scanner *bufio.Scanner) (tictactoe.Game, error) {
	for {
		fmt.Fprint(r.out, "Your move (row col): ")
		if !scanner.Scan() {
			return game, fmt.Errorf("input closed")
		}
		parts := strings.Fields(scanner.Text())
		if len(parts) != 2 {
			fmt.Fprintln(r.out, "Invalid input, enter two numbers, e.g.: 0 1")
			continue
		}
		var x, y int
		if _, err := fmt.Sscan(parts[0], &y); err != nil {
			fmt.Fprintln(r.out, "Invalid row:", parts[0])
			continue
		}
		if _, err := fmt.Sscan(parts[1], &x); err != nil {
			fmt.Fprintln(r.out, "Invalid column:", parts[1])
			continue
		}
		next, err := tictactoe.PlaceMark(game, x, y)
		if err != nil {
			fmt.Fprintln(r.out, "Try again:", err)
			continue
		}
		return next, nil
	}
}

func (r Runner) printBoard(game tictactoe.Game) {
	fmt.Fprintln(r.out)
	// Column header.
	fmt.Fprint(r.out, "  ")
	for x := range game.Size {
		fmt.Fprintf(r.out, " %d", x)
	}
	fmt.Fprintln(r.out)
	// Rows.
	for y := range game.Size {
		fmt.Fprintf(r.out, "%d ", y)
		for x := range game.Size {
			fmt.Fprintf(r.out, " %s", game.GetCell(x, y))
		}
		fmt.Fprintln(r.out)
	}
	fmt.Fprintln(r.out)
}
