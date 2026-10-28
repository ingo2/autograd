package scoremodel

import (
	"encoding/json"
	"os"
)

// DatasetInfo records dataset provenance for a saved model.
type DatasetInfo struct {
	File         string `json:"file"`
	TrainSamples int    `json:"trainSamples"`
	TestSamples  int    `json:"testSamples"`
}

// TrainConfig records the architecture and hyperparameters used to train a saved model.
type TrainConfig struct {
	Seed         int64 `json:"seed"`
	Architecture []int `json:"architecture"`
	// Name of the opponent for matchups during training, see TrainParams.MatchUpInterval.
	MatchUpOpponent string `json:"matchUpOpponent,omitempty"`
	TrainParams
}

// ModelInfo is embedded as metadata in a saved model JSON file (see nn.SaveWeightsWithInfo),
// recording how the model was trained and how it performed.
type ModelInfo struct {
	Dataset       DatasetInfo   `json:"dataset"`
	Config        TrainConfig   `json:"config"`
	TrainGameTree GameTreeStats `json:"trainGameTree"`
	TestGameTree  GameTreeStats `json:"testGameTree"`
	Perfection    GameTreeStats `json:"perfection"`
	MatchUp       MatchUpStats  `json:"matchUp"`
	TrainStats    TrainStats    `json:"trainStats"`
}

// LoadModelInfo reads the "info" metadata embedded in a model JSON file saved with
// nn.SaveWeightsWithInfo. Returns the zero ModelInfo if the file has no info field.
func LoadModelInfo(path string) (ModelInfo, error) {
	var file struct {
		Info json.RawMessage `json:"info"`
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return ModelInfo{}, err
	}
	if err := json.Unmarshal(data, &file); err != nil {
		return ModelInfo{}, err
	}

	var info ModelInfo
	if len(file.Info) == 0 {
		return info, nil
	}
	if err := json.Unmarshal(file.Info, &info); err != nil {
		return ModelInfo{}, err
	}
	return info, nil
}
