package scoremodel

import (
	"math"
	"testing"
)

const tol = 1e-9

func TestAccuracy(t *testing.T) {
	tests := []struct {
		name           string
		tp, tn, fp, fn int
		want           float64
	}{
		{"all correct", 5, 5, 0, 0, 1.0},
		{"all wrong", 0, 0, 5, 5, 0.0},
		{"mixed", 3, 3, 1, 1, 0.75},
		{"empty", 0, 0, 0, 0, 0.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := accuracy(tt.tp, tt.tn, tt.fp, tt.fn); math.Abs(got-tt.want) > tol {
				t.Errorf("accuracy(%d,%d,%d,%d) = %g, want %g", tt.tp, tt.tn, tt.fp, tt.fn, got, tt.want)
			}
		})
	}
}

func TestPrecision(t *testing.T) {
	tests := []struct {
		name   string
		tp, fp int
		want   float64
	}{
		{"all correct", 5, 0, 1.0},
		{"all wrong", 0, 5, 0.0},
		{"mixed", 3, 1, 0.75},
		{"no positives predicted", 0, 0, 0.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := precision(tt.tp, tt.fp); math.Abs(got-tt.want) > tol {
				t.Errorf("precision(%d,%d) = %g, want %g", tt.tp, tt.fp, got, tt.want)
			}
		})
	}
}

func TestRecall(t *testing.T) {
	tests := []struct {
		name   string
		tp, fn int
		want   float64
	}{
		{"all found", 5, 0, 1.0},
		{"none found", 0, 5, 0.0},
		{"mixed", 3, 1, 0.75},
		{"no actual positives", 0, 0, 0.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := recall(tt.tp, tt.fn); math.Abs(got-tt.want) > tol {
				t.Errorf("recall(%d,%d) = %g, want %g", tt.tp, tt.fn, got, tt.want)
			}
		})
	}
}

func TestF1Score(t *testing.T) {
	tests := []struct {
		name              string
		precision, recall float64
		want              float64
	}{
		{"perfect", 1.0, 1.0, 1.0},
		{"both zero", 0.0, 0.0, 0.0},
		{"precision zero", 0.0, 1.0, 0.0},
		{"recall zero", 1.0, 0.0, 0.0},
		{"harmonic mean", 1.0, 0.01, 2 * 1.0 * 0.01 / (1.0 + 0.01)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := f1Score(tt.precision, tt.recall); math.Abs(got-tt.want) > tol {
				t.Errorf("f1Score(%g,%g) = %g, want %g", tt.precision, tt.recall, got, tt.want)
			}
		})
	}
}
