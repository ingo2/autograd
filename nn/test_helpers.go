package nn

import (
	"math"
	"testing"
)

func approx(t *testing.T, label string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %f, want %f (tol %g)", label, got, want, tol)
	}
}
