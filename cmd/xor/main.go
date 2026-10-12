// Trains a small MLP on XOR several times from different random initializations, reports how
// often it solves XOR and prints the best and the worst net.
//
// Example:
//
//	go run ./cmd/xor -runs 100 -hidden 3 -act tanh -epochs 2000 -lr 0.05 -seed 42
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"autograd"
)

var activations = map[string]autograd.ActivationFunc{
	"linear": autograd.Linear,
	"relu":   autograd.ReLU,
	"tanh":   autograd.Tanh,
}

type sample struct {
	inputs []*autograd.Value
	target float64
}

// result describes one training run.
type result struct {
	seed   int64
	mlp    *autograd.MLP
	losses []float64 // average loss per epoch
	loss   float64   // average loss of the trained net
	solved bool
}

func main() {
	runs := flag.Int("runs", 100, "number of training runs")
	epochs := flag.Int("epochs", 1000, "number of training epochs per run")
	hidden := flag.Int("hidden", 2, "number of neurons in the hidden layer")
	actFuncName := flag.String("act", "tanh", "hidden layer activation (linear, relu, tanh)")
	learningRate := flag.Float64("lr", 0.1, "learning rate")
	seed := flag.Int64("seed", 0, "base random seed; run k is initialized with seed+k")
	flag.Parse()

	if *runs <= 0 {
		fail("-runs must be > 0, got %d", *runs)
	}
	if *epochs <= 0 {
		fail("-epochs must be > 0, got %d", *epochs)
	}
	if *hidden <= 0 {
		fail("-hidden must be > 0, got %d", *hidden)
	}
	act, ok := activations[*actFuncName]
	if !ok {
		fail("-act: unknown activation %q", *actFuncName)
	}

	layerSizes := []int{2, *hidden, 1}
	layerActs := []autograd.ActivationFunc{act, autograd.Tanh}

	// XOR with targets in [-1, 1]:
	// 0 0 => -1
	// 0 1 =>  1
	// 1 0 =>  1
	// 1 1 => -1
	samples := []sample{
		{inputs: values(0, 0), target: -1.0},
		{inputs: values(0, 1), target: 1.0},
		{inputs: values(1, 0), target: 1.0},
		{inputs: values(1, 1), target: -1.0},
	}

	fmt.Printf("Architecture: %v, activations: %v\n", layerSizes, layerActs)
	fmt.Printf("Runs: %d, epochs: %d, learning rate: %g, seeds: %d..%d\n\n",
		*runs, *epochs, *learningRate, *seed, *seed+int64(*runs)-1)

	var best, worst *result
	numSolved := 0
	for k := range *runs {
		runSeed := *seed + int64(k)
		mlp := autograd.NewMLP(rand.New(rand.NewSource(runSeed)), layerSizes, layerActs)
		losses := train(mlp, samples, *epochs, *learningRate)
		r := &result{
			seed:   runSeed,
			mlp:    mlp,
			losses: losses,
			loss:   avgLoss(mlp, samples),
			solved: solved(mlp, samples),
		}
		if r.solved {
			numSolved++
		}

		if best == nil || better(r, best) {
			best = r
		}
		if worst == nil || better(worst, r) {
			worst = r
		}
	}

	fmt.Printf("Solved: %d, not solved: %d (%.1f%% solved)\n", numSolved, *runs-numSolved,
		100*float64(numSolved)/float64(*runs))

	printNet("Best", best, samples)
	printNet("Worst", worst, samples)
}

func better(a, b *result) bool {
	if a.solved != b.solved {
		return a.solved
	}
	return a.loss < b.loss
}

func printNet(title string, r *result, samples []sample) {
	fmt.Printf("\n%s net (seed %d, loss %.6f):\n", title, r.seed, r.loss)
	printLosses(r.losses)
	printResults(r.mlp, samples)
	printParams(r.mlp)
}

func train(mlp *autograd.MLP, samples []sample, epochs int, learningRate float64) []float64 {
	losses := make([]float64, epochs)
	for epoch := range epochs {
		totalLoss := 0.0
		for _, s := range samples {
			// Forward pass.
			score := mlp.Call(s.inputs, false)
			loss := autograd.NewValueMSE(score, []float64{s.target})
			totalLoss += loss.Data

			// Backward pass.
			mlp.ZeroGrad()
			loss.Backward()

			// Update parameters.
			for _, param := range mlp.Params() {
				param.Data -= param.Grad * learningRate
			}
		}
		losses[epoch] = totalLoss / float64(len(samples))
	}
	return losses
}

func avgLoss(mlp *autograd.MLP, samples []sample) float64 {
	total := 0.0
	for _, s := range samples {
		total += autograd.NewValueMSE(mlp.Call(s.inputs, true), []float64{s.target}).Data
	}
	return total / float64(len(samples))
}

func solved(mlp *autograd.MLP, samples []sample) bool {
	for _, s := range samples {
		out := mlp.Call(s.inputs, true)[0].Data
		var class float64
		switch {
		case out < 0:
			class = -1
		case out > 0:
			class = 1
		}
		if class != s.target {
			return false
		}
	}
	return true
}

func printLosses(losses []float64) {
	epochs := len(losses)
	// Prints per-epoch losses roughly 10 times, plus the first and the last epoch.
	logEvery := max(1, epochs/10)
	epochWidth := len(strconv.Itoa(epochs))
	fmt.Println("  Training:")
	for k, loss := range losses {
		epoch := k + 1
		if epoch == 1 || epoch%logEvery == 0 || epoch == epochs {
			fmt.Printf("    epoch %*d  avg loss %.6f\n", epochWidth, epoch, loss)
		}
	}
}

func printResults(mlp *autograd.MLP, samples []sample) {
	fmt.Println("  Results:")
	for _, s := range samples {
		out := mlp.Call(s.inputs, true)[0].Data
		fmt.Printf("    %g XOR %g => %+.4f  (target %+g)\n",
			s.inputs[0].Data, s.inputs[1].Data, out, s.target)
	}
}

func printParams(mlp *autograd.MLP) {
	fmt.Println("  Learned parameters:")
	for i, layer := range mlp.Layers {
		name := "hidden"
		if i == len(mlp.Layers)-1 {
			name = "output"
		}
		fmt.Printf("    Layer %d (%s, %s):\n", i, name, layer.Neurons[0].Activation)
		for j, n := range layer.Neurons {
			weights := make([]string, len(n.Weights))
			for k, w := range n.Weights {
				weights[k] = fmt.Sprintf("%+.4f", w.Data)
			}
			fmt.Printf("      neuron %d: w=[%s] b=%+.4f\n",
				j, strings.Join(weights, ", "), n.Bias.Data)
		}
	}
}

func values(xs ...float64) []*autograd.Value {
	vs := make([]*autograd.Value, len(xs))
	for i, x := range xs {
		vs[i] = autograd.NewValue(x)
	}
	return vs
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	flag.Usage()
	os.Exit(2)
}
