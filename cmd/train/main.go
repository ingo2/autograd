package main

import (
	"autograd/nn"
	"autograd/scoremodel"
	"autograd/tictactoe"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const seed int64 = 42

// evalMatchUpGames is the number of games in the matchup against minimax after training. It is
// fixed so that the results stay comparable between models.
const evalMatchUpGames = 1000

// parseActivation maps a name to the corresponding nn.ActivationFunc.
func parseActivation(name string) (nn.ActivationFunc, error) {
	switch name {
	case "linear":
		return nn.Linear, nil
	case "relu":
		return nn.ReLU, nil
	case "tanh":
		return nn.Tanh, nil
	default:
		return nn.ActivationFunc{},
			fmt.Errorf("unknown activation %q: valid values are linear, relu, tanh", name)
	}
}

// parseHidden parses a comma-separated list of hidden layer sizes (e.g. "32" or "64,32").
func parseHidden(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	sizes := make([]int, len(parts))
	for i, p := range parts {
		n, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("invalid hidden layer size %q", p)
		}
		sizes[i] = n
	}
	return sizes, nil
}

// saveResults writes the model weights (with info embedded) and dataset splits alongside outPath.
func saveResults(outPath string, model *nn.MLP, info scoremodel.ModelInfo, train, test []tictactoe.AnnotatedBoard) {
	if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
		log.Fatalf("failed to create output directory: %v", err)
	}
	if err := model.SaveWeightsWithInfo(outPath, info); err != nil {
		log.Fatalf("failed to save model: %v", err)
	}
	log.Printf("model saved to %s", outPath)

	base := strings.TrimSuffix(outPath, ".json")
	for _, split := range []struct {
		name   string
		boards []tictactoe.AnnotatedBoard
	}{
		{"train", train},
		{"test", test},
	} {
		if len(split.boards) == 0 {
			continue
		}
		path := base + "_" + split.name + ".ndjson"
		if err := tictactoe.WriteAnnotatedBoards(path, split.boards, true); err != nil {
			log.Fatalf("failed to write %s split: %v", split.name, err)
		}
		log.Printf("%s split (%d samples) saved to %s", split.name, len(split.boards), path)
	}
}

func main() {
	dataPath := flag.String("data", "", "path to annotated NDJSON file")
	outPath := flag.String("out", "", "path to save trained model weights as JSON (optional)")
	maxEpochs := flag.Int64("epochs", 300, "maximum number of training epochs")
	testFraction := flag.Float64("test-fraction", 0.1,
		"fraction of the dataset removed from training, only evaluated after training")
	learningRate := flag.Float64("learning-rate", 0.075, "SGD learning rate")
	weightDecay := flag.Float64("weight-decay", 0.0, "weight decay factor pulling weights towards zero")
	matchUpInterval := flag.Int64("matchup-interval", 0,
		"play a matchup every this many epochs, revert to the best model if not better; 0 = disabled")
	matchUpPatience := flag.Int64("matchup-patience", 0,
		"stop when the matchup score has not improved for this many matchups; 0 = disabled")
	matchUpGames := flag.Int("matchup-games", 100, "number of games per matchup, alternating sides")
	matchUpOpponentName := flag.String("matchup-opponent", "minimax",
		"matchup opponent: minimax, or self (the best model so far)")
	lossName := flag.String("loss", "mse", "loss function: mse, hinge, or crossentropy")
	hingeMargin := flag.Float64("hinge-margin", 0.1, "margin for hinge loss (ignored for mse)")
	activationHiddenName := flag.String("activation-hidden", "relu", "hidden layer activation: linear, relu, tanh")
	activationOutName := flag.String("activation-out", "tanh", "output layer activation: linear, relu, tanh")
	hiddenStr := flag.String("hidden", "32",
		"hidden layer sizes only, comma-separated; input and output layers are added (e.g. \"64,32\")")
	outcomeClasses := flag.Bool("outcome-classes", false,
		"train on win/draw/loss outcome classes (-1/0/1) instead of minimax scores")
	flag.Parse()

	if *dataPath == "" {
		log.Fatal("usage: train -data <path> [-out <path>]")
	}
	hiddenSizes, err := parseHidden(*hiddenStr)
	if err != nil {
		log.Fatalf("invalid -hidden: %v", err)
	}

	log.Printf("loading dataset...")
	annotatedBoards, err := tictactoe.ReadAnnotatedBoards(*dataPath, 0, -1, *outcomeClasses)
	if err != nil {
		log.Fatalf("failed to load dataset: %v", err)
	}
	log.Printf("loaded %d samples", len(annotatedBoards))

	// The split only removes boards from training. D4Split keeps the symmetric variants of a board
	// together, so no removed board is seen in training in a rotated or mirrored form. No validation
	// split is used.
	train, test, _, err :=
		tictactoe.D4Split(annotatedBoards, *testFraction, 0.0, rand.New(rand.NewSource(seed)))
	if err != nil {
		log.Fatalf("failed to split dataset: %v", err)
	}
	log.Printf("training set size: %d, test set size: %d", len(train), len(test))

	params := scoremodel.DefaultTrainParams()
	params.MaxEpochs = *maxEpochs
	params.LearningRate = *learningRate
	params.WeightDecay = *weightDecay
	params.MatchUpInterval = *matchUpInterval
	params.MatchUpPatience = *matchUpPatience
	params.MatchUpGames = *matchUpGames
	params.HingeMargin = *hingeMargin
	switch *lossName {
	case "mse":
		params.LossFunc = scoremodel.MSELoss
	case "hinge":
		params.LossFunc = scoremodel.NewHingeLoss(params.HingeMargin)
	case "crossentropy":
		params.LossFunc = scoremodel.CrossEntropyLoss
	default:
		log.Fatalf("unknown -loss %q: valid values are mse, hinge, crossentropy", *lossName)
	}
	var matchUpOpponent scoremodel.MatchUpOpponent
	switch *matchUpOpponentName {
	case "minimax":
		matchUpOpponent = scoremodel.MinimaxOpponent(seed)
	case "self":
		matchUpOpponent = scoremodel.SelfPlayOpponent(seed)
	default:
		log.Fatalf("unknown -matchup-opponent %q: valid values are minimax, self", *matchUpOpponentName)
	}
	if act, err := parseActivation(*activationHiddenName); err != nil {
		log.Fatalf("invalid -activation-hidden: %v", err)
	} else {
		params.ActivationHidden = act
	}
	if act, err := parseActivation(*activationOutName); err != nil {
		log.Fatalf("invalid -activation-out: %v", err)
	} else {
		params.ActivationOut = act
	}
	archFull := make([]int, 0, 2+len(hiddenSizes))
	archFull = append(archFull, 18)
	archFull = append(archFull, hiddenSizes...)
	archFull = append(archFull, 9)
	log.Printf("training configuration:")
	log.Printf("  outcome-classes:   %t", *outcomeClasses)
	log.Printf("  architecture:      %v", archFull)
	log.Printf("  loss:              %s", params.LossFunc)
	log.Printf("  activation-hidden: %s", params.ActivationHidden)
	log.Printf("  activation-out:    %s", params.ActivationOut)
	log.Printf("  learning-rate:     %g", params.LearningRate)
	log.Printf("  weight-decay:      %g", params.WeightDecay)
	log.Printf("  epochs:            %d", params.MaxEpochs)
	log.Printf("  matchup-interval:  %d", params.MatchUpInterval)
	log.Printf("  matchup-patience:  %d", params.MatchUpPatience)
	log.Printf("  matchup-games:     %d", params.MatchUpGames)
	log.Printf("  matchup-opponent:  %s", *matchUpOpponentName)

	rng := rand.New(rand.NewSource(seed))
	model := scoremodel.NewScoreModel(rng, hiddenSizes, params.ActivationHidden, params.ActivationOut)

	trainStart := time.Now()
	stats, err := scoremodel.TrainScoreModel(model, params, train, matchUpOpponent, rng)
	if err != nil {
		log.Fatalf("training failed: %v", err)
	}
	trainDuration := time.Since(trainStart)
	hours := int(trainDuration.Hours())
	minutes := int(trainDuration.Minutes()) % 60
	seconds := int(trainDuration.Seconds()) % 60
	log.Printf("training time: %d:%02d:%02d", hours, minutes, seconds)
	log.Printf("training stats: loss(total): %g, epochs: %d",
		stats.TotalLoss[len(stats.TotalLoss)-1], stats.Epochs)
	if len(stats.MatchUpScore) > 0 {
		kept := 0
		for _, k := range stats.MatchUpKept {
			if k {
				kept++
			}
		}
		log.Printf("training stats: matchups: %d, kept: %d", len(stats.MatchUpScore), kept)
	}

	// Fraction of boards where the model's chosen move preserves the optimal outcome class.
	log.Printf("===== Game tree evaluation =====")
	trainGameTree := scoremodel.EvaluateGameTree(model, train)
	log.Printf("train game tree: overall=%g critical=%g neverLoses=%g winConversion=%g",
		trainGameTree.OverallAcc, trainGameTree.CriticalPositionAcc,
		trainGameTree.NeverLosesAcc, trainGameTree.WinConversionAcc)

	testGameTree := scoremodel.EvaluateGameTree(model, test)
	log.Printf("test  game tree: overall=%g critical=%g neverLoses=%g winConversion=%g",
		testGameTree.OverallAcc, testGameTree.CriticalPositionAcc,
		testGameTree.NeverLosesAcc, testGameTree.WinConversionAcc)

	// Exact play-strength breakdown over the full reachable board space, scored with perfect minimax.
	log.Printf("===== Perfection =====")
	perfection := scoremodel.EvaluatePerfection(model)
	log.Printf("perfection: overall=%g critical=%g neverLoses=%g winConversion=%g",
		perfection.OverallAcc, perfection.CriticalPositionAcc,
		perfection.NeverLosesAcc, perfection.WinConversionAcc)

	// Games played against a perfect minimax opponent.
	log.Printf("===== Matchup Strength =====")
	opponent := tictactoe.MinimaxPlayer{
		Rand: rand.New(rand.NewSource(seed)), MinimaxConfig: tictactoe.DefaultMinimaxConfig()}
	matchUp := scoremodel.EvaluateMatchUp(model, opponent, evalMatchUpGames)
	log.Printf("matchup vs %s (%d games): %d-%d-%d (score=%g)",
		opponent.Identity(), matchUp.NumGames, matchUp.Wins, matchUp.Draws, matchUp.Losses,
		matchUp.Score)

	if *outPath != "" {
		info := scoremodel.ModelInfo{
			Dataset: scoremodel.DatasetInfo{
				File:         filepath.Base(*dataPath),
				TrainSamples: len(train),
				TestSamples:  len(test),
			},
			Config: scoremodel.TrainConfig{
				Seed:            seed,
				Architecture:    archFull,
				MatchUpOpponent: *matchUpOpponentName,
				TrainParams:     *params,
			},
			TrainStats:    stats,
			TrainGameTree: trainGameTree,
			TestGameTree:  testGameTree,
			Perfection:    perfection,
			MatchUp:       matchUp,
		}
		saveResults(*outPath, model, info, train, test)
	}
}
