package tictactoe

import (
	"fmt"
	"math/rand"
	"slices"
)

// Shuffle returns a new slice with the boards in random order determined by rng.
func Shuffle(boards []AnnotatedBoard, rng *rand.Rand) []AnnotatedBoard {
	shuffled := slices.Clone(boards)
	rng.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	return shuffled
}

// Split partitions boards into train, test, and validation slices according to the given
// fractions. The caller is responsible for shuffling before splitting if randomness is needed.
func Split(boards []AnnotatedBoard, testFraction, validationFraction float64) (train, test, validation []AnnotatedBoard, err error) {
	if testFraction < 0 || testFraction >= 1 || validationFraction < 0 || validationFraction > 1 ||
		testFraction+validationFraction > 1 {
		return nil, nil, nil,
			fmt.Errorf("got invalid split ratios: (test: %g, validation: %g)",
				testFraction, validationFraction)
	}
	testSize := int(float64(len(boards)) * testFraction)
	validationSize := int(float64(len(boards)) * validationFraction)
	return boards[testSize+validationSize:],
		boards[:testSize],
		boards[testSize : testSize+validationSize], nil
}

// D4Split partitions boards into train, test, and validation slices while keeping symmetry groups
// together: all boards sharing the same Board key are always assigned to the same partition.
// This prevents D4-augmented training examples from leaking into the held-out sets. Sizes are
// approximate (greedy first-fit on groups), so actual fractions may overshoot slightly. The caller
// is responsible for supplying a seeded rng for reproducibility.
func D4Split(boards []AnnotatedBoard, testFraction, validationFraction float64, rng *rand.Rand) (
	train, test, validation []AnnotatedBoard, err error) {
	if testFraction < 0 || testFraction >= 1 || validationFraction < 0 || validationFraction > 1 ||
		testFraction+validationFraction > 1 {
		return nil, nil, nil,
			fmt.Errorf("got invalid split ratios: (test: %g, validation: %g)",
				testFraction, validationFraction)
	}

	groupedBoards := map[BoardKey][]AnnotatedBoard{}
	for _, ab := range boards {
		key := ab.Board.ToKey()
		groupedBoards[key] = append(groupedBoards[key], ab)
	}

	keys := make([]BoardKey, 0, len(groupedBoards))
	for key := range groupedBoards {
		keys = append(keys, key)
	}
	// Enforce deterministic order before shuffling.
	slices.Sort(keys)
	rng.Shuffle(len(keys), func(i, j int) { keys[i], keys[j] = keys[j], keys[i] })

	testSize := int(testFraction * float64(len(boards)))
	validationSize := int(validationFraction * float64(len(boards)))
	for _, key := range keys {
		gb := groupedBoards[key]
		switch {
		case len(test) < testSize:
			test = append(test, gb...)
		case len(validation) < validationSize:
			validation = append(validation, gb...)
		default:
			train = append(train, gb...)
		}
	}
	return train, test, validation, nil
}
