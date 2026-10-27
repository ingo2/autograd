package main

import (
	"autograd/tictactoe"
	"flag"
	"fmt"
	"log"
)

func main() {
	out := flag.String("out", "boards.ndjson", "output NDJSON file")
	overwrite := flag.Bool("overwrite", false, "overwrite output file (default: append)")
	gamma := flag.Float64("gamma", 0.9, "minimax gamma discount rate (0-1)")
	maxDepth := flag.Int("max-depth", 0, "maximum minimax depth (0=unlimited)")
	nodeBudget := flag.Int("node-budget", 0, "minimax node budget per move via iterative deepening (0=unlimited)")
	flag.Parse()

	cfg := tictactoe.MinimaxConfig{
		MaxDepth:   *maxDepth,
		NodeBudget: *nodeBudget,
		Gamma:      *gamma,
	}

	boards := tictactoe.DeduplicateBoards(tictactoe.FilterBoards(tictactoe.Enumerate3x3Boards()))
	tictactoe.SortBoards(boards)
	annotated := tictactoe.AnnotateCanonicalBoards(boards, tictactoe.X, cfg)
	fmt.Printf("Annotated %d boards.\n", len(annotated))

	if err := tictactoe.WriteAnnotatedBoards(*out, annotated, *overwrite); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Written %d annotated boards to %s.\n", len(annotated), *out)
}
