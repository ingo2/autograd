package scoremodel

import (
	"autograd/nn"
	"encoding/json"
	"math/rand"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLoadModelInfo(t *testing.T) {
	model := nn.NewMLP(rand.New(rand.NewSource(0)), []int{2, 3, 1},
		[]nn.ActivationFunc{nn.ReLU, nn.Tanh})

	want := ModelInfo{
		Dataset: DatasetInfo{File: "boards.ndjson", TrainSamples: 100, TestSamples: 10},
		Config: TrainConfig{
			Seed:         42,
			Architecture: []int{2, 3, 1},
			TrainParams: TrainParams{
				LossFunc:         MSELoss,
				ActivationHidden: nn.ReLU,
				ActivationOut:    nn.Tanh,
			},
		},
		TrainStats: TrainStats{
			TotalLoss: []float64{0.5, 0.4},
			Epochs:    2,
		},
	}

	pathWithInfo := filepath.Join(t.TempDir(), "model.json")
	if err := model.SaveWeightsWithInfo(pathWithInfo, want); err != nil {
		t.Fatalf("SaveWeightsWithInfo: %v", err)
	}

	got, err := LoadModelInfo(pathWithInfo)
	if err != nil {
		t.Fatalf("LoadModelInfo: %v", err)
	}
	// LossFunc and ActivationFunc hold unexported func fields, which reflect.DeepEqual never
	// considers equal (even to themselves), so compare the JSON representations instead.
	gotJSON, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("Marshal(got): %v", err)
	}
	wantJSON, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("Marshal(want): %v", err)
	}
	if string(gotJSON) != string(wantJSON) {
		t.Errorf("LoadModelInfo() = %s, want %s", gotJSON, wantJSON)
	}

	pathNoInfo := filepath.Join(t.TempDir(), "model_no_info.json")
	if err := model.SaveWeights(pathNoInfo); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	got, err = LoadModelInfo(pathNoInfo)
	if err != nil {
		t.Fatalf("LoadModelInfo (no info): %v", err)
	}
	if !reflect.DeepEqual(got, ModelInfo{}) {
		t.Errorf("LoadModelInfo() for file without info = %+v, want zero value", got)
	}
}
