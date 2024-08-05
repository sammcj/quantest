package quantest

import (
	"fmt"
	"strings"
)

// Version can be set at build time
var Version string

func GetModelConfig(modelName string) (ModelConfig, error) {
	if strings.Contains(modelName, ":") {
		return GetOllamaModelConfig(modelName)
	}
	return GetHFModelConfig(modelName)
}

func EstimateVRAMForModel(modelName string, vram float64, contextSize int, quantLevel, kvQuant string) (*VRAMEstimation, error) {

	modelConfig, err := GetModelConfig(modelName)
	if err != nil {
		return nil, fmt.Errorf("error getting model config: %w", err)
	}

  // If quantLevel is not provided, and it's an ollama model, check the model's config
  if quantLevel == "" && modelConfig.IsOllama {
    quantLevel = modelConfig.QuantLevel
  } else if quantLevel == "" {
    fmt.Println("Quant level not provided, and model is not an Ollama model. Defaulting to q4_k_m...")
    quantLevel = "q4_k_m"
  }

	// Parse BPW from quantLevel
	bpw, err := ParseBPWOrQuant(quantLevel)
	if err != nil {
		return nil, fmt.Errorf("error parsing quantisation level: %w", err)
	}

	// Calculate VRAM usage
	estimatedVRAM, err := CalculateVRAM(modelConfig, bpw, contextSize, KVCacheQuantisation(kvQuant))
	if err != nil {
		return nil, fmt.Errorf("error calculating VRAM: %w", err)
	}

	// Calculate maximum context size
	maxContextSize, err := CalculateContext(modelConfig, vram, bpw, KVCacheQuantisation(kvQuant))
	if err != nil {
		maxContextSize = 0 // Set to 0 if calculation fails
	}

	// Calculate best BPW
	bestBPW, recommendations, err := CalculateBPW(modelConfig, vram, contextSize, KVCacheQuantisation(kvQuant), "gguf")
	if err != nil {
		bestBPW = "Unknown"
		recommendations = QuantRecommendations{Recommendations: make(map[int]string)}
	}

	return &VRAMEstimation{
		ModelName:       modelName,
		ModelConfig:     modelConfig, // Add this line
		ContextSize:     contextSize,
		KVCacheQuant:    KVCacheQuantisation(kvQuant),
		AvailableVRAM:   vram,
		QuantLevel:      quantLevel,
		EstimatedVRAM:   estimatedVRAM,
		FitsAvailable:   estimatedVRAM <= vram,
		MaxContextSize:  maxContextSize,
		MaximumQuant:    fmt.Sprintf("%v", bestBPW),
		Recommendations: recommendations.Recommendations,
	}, nil
}
