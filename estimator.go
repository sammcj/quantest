package quantest

import (
	"fmt"
	"strings"

	"github.com/sammcj/gollama/logging"
)

// EstimateVRAM calculates VRAM usage for a given model configuration.
//
// Parameters:
//   - modelName: A pointer to a string representing the model name (Huggingface/ModelID or Ollama:modelName).
//   - contextSize: An integer representing the context size.
//   - kvCacheQuant: The KV cache quantization level.
//   - availableVRAM: A float64 representing the available VRAM in GB.
//   - quantLevel: A string representing the quantization level.
//
// Returns:
//   - *VRAMEstimation: A pointer to a VRAMEstimation struct containing the estimation results.
//   - error: An error if the estimation fails.
//
// Example:
//
//	estimation, err := quantest.EstimateVRAM(
//		&modelName,
//		8192,
//		quantest.KVCacheFP16,
//		24.0,
//		"Q4_K_M",
//	)
//	if err != nil {
//		log.Fatal(err)
//	}
//	fmt.Printf("Max Context Size: %d\n", estimation.MaxContextSize)
func EstimateVRAM(
	modelName *string,
	contextSize int,
	kvCacheQuant KVCacheQuantisation,
	availableVRAM float64,
	quantLevel string,
) (*VRAMEstimation, error) {
	var ollamaModelInfo *OllamaModelInfo
	var err error

	logging.DebugLogger.Println("Estimating VRAM for", *modelName)

	// Check if the modelName is an Ollama model
	if strings.Contains(*modelName, ":") {
		logging.DebugLogger.Println("Fetching Ollama model info for", *modelName)
		ollamaModelInfo, err = FetchOllamaModelInfo(*modelName)
		if err != nil {
			logging.ErrorLogger.Println("Error fetching Ollama model info:", err)
			return nil, fmt.Errorf("error fetching Ollama model info: %v", err)
		}
		logging.DebugLogger.Printf("Ollama model info: %+v", ollamaModelInfo)
	}

	// Use default values if not provided
	if contextSize == 0 {
		contextSize = DefaultContextSize
	}
	if availableVRAM == 0 {
		availableVRAM = DefaultVRAM
	}
	if quantLevel == "" {
		quantLevel = DefaultQuantLevel
	}

	bpw, err := ParseBPWOrQuant(quantLevel)
	if err != nil {
		return nil, err
	}

	modelConfig, _ := GetModelConfig(*modelName)

	estimatedVRAM, err := CalculateVRAM(modelConfig, bpw, contextSize, kvCacheQuant)
	if err != nil {
		return nil, err
	}

	maxContextSize, err := CalculateContext(modelConfig, availableVRAM, bpw, kvCacheQuant)
	if err != nil {
		return nil, err
	}

	maximumQuant, recommendations, err := CalculateBPW(modelConfig, availableVRAM, contextSize, kvCacheQuant, "gguf")
	if err != nil {
		return nil, err
	}

	return &VRAMEstimation{
		ModelName:       *modelName,
		ContextSize:     contextSize,
		KVCacheQuant:    kvCacheQuant,
		AvailableVRAM:   availableVRAM,
		QuantLevel:      quantLevel,
		EstimatedVRAM:   estimatedVRAM,
		FitsAvailable:   estimatedVRAM <= availableVRAM,
		MaxContextSize:  maxContextSize,
		MaximumQuant:    maximumQuant.(string),
		Recommendations: recommendations.Recommendations,
		ollamaModelInfo: ollamaModelInfo,
	}, nil
}
