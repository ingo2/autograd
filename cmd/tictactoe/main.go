package main

import (
	"autograd/scoremodel"
	"autograd/tictactoe"
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// defaultModelDir walks upward from the working directory to find the repo root
// (identified by go.mod), then returns the absolute path to the score-reg model directory.
// Falls back to a CWD-relative path when the repo root cannot be found.
func defaultModelDir() string {
	const rel = "models"
	dir, err := os.Getwd()
	if err != nil {
		return rel
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return filepath.Join(dir, rel)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return rel
		}
		dir = parent
	}
}

func main() {
	boardSize := flag.Int("board-size", 3, "board size (e.g. 3 for 3x3)")
	connectN := flag.Int("connect-n", 0, "marks in a line needed to win (default: board-size)")
	nodeBudget := flag.Int("node-budget", 0, "max minimax nodes per move via iterative deepening (0 = unlimited)")
	modelDir := flag.String("model-dir", defaultModelDir(),
		"directory of score-reg MLP weights JSON files, all offered as opponents")
	flag.Parse()
	if *connectN == 0 {
		*connectN = *boardSize
	}

	// Single shared reader for all stdin input, avoids double-buffering when
	// we hand it off to the runner below.
	stdin := bufio.NewReader(os.Stdin)

	players := []tictactoe.AIPlayer{
		tictactoe.MinimaxPlayer{MinimaxConfig: tictactoe.DefaultMinimaxConfig()}}
	players = append(players, modelPlayers(*modelDir)...)

	fmt.Println("Choose your opponent:")
	for i, p := range players {
		fmt.Printf("  [%d] %s\n", i+1, p.Name())
	}
	fmt.Print("Enter number: ")

	var ai tictactoe.AIPlayer
	for {
		line, err := stdin.ReadString('\n')
		if err != nil {
			os.Exit(0)
		}
		n, err := strconv.Atoi(strings.TrimSpace(line))
		if err != nil || n < 1 || n > len(players) {
			fmt.Printf("Please enter a number between 1 and %d: ", len(players))
			continue
		}
		ai = players[n-1]
		break
	}
	// NodeBudget override only applies to the minimax player.
	if minimax, ok := ai.(tictactoe.MinimaxPlayer); ok {
		minimax.NodeBudget = *nodeBudget
		ai = minimax
	}

	fmt.Println("Tic-Tac-Toe! X always goes first.")
	fmt.Print("Choose your mark (X/O): ")
	var humanPlayer tictactoe.Player
	for {
		line, err := stdin.ReadString('\n')
		if err != nil {
			os.Exit(0)
		}
		switch strings.ToUpper(strings.TrimSpace(line)) {
		case "X":
			humanPlayer = tictactoe.X
		case "O":
			humanPlayer = tictactoe.O
		default:
			fmt.Print("Please enter X or O: ")
			continue
		}
		break
	}

	runner := NewRunner(stdin, os.Stdout, ai, *boardSize, *connectN)
	game, err := runner.Play(humanPlayer)
	if err != nil {
		log.Fatal(err)
	}

	runner.printBoard(game)
	switch game.Status {
	case tictactoe.Won:
		if game.Winner == humanPlayer {
			fmt.Println("You win!")
		} else {
			fmt.Println("AI wins!")
		}
	case tictactoe.Draw:
		fmt.Println("It's a draw!")
	}
}

// modelPlayers loads every *.json file in dir as a score-reg MLP player, sorted by file name.
// Files that fail to load are logged and skipped.
func modelPlayers(dir string) []tictactoe.AIPlayer {
	if dir == "" {
		return nil
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		log.Printf("no MLP players, cannot read model directory %s: %v", dir, err)
		return nil
	}
	var players []tictactoe.AIPlayer
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		path := filepath.Join(dir, e.Name())
		p, err := scoremodel.NewScoreModelPlayer(path)
		if err != nil {
			log.Printf("skipping %s: %v", path, err)
			continue
		}
		players = append(players, p)
	}
	return players
}
