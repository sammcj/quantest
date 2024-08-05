package quantest

import (
	"fmt"
	"os"
	"strings"

	"github.com/sammcj/gollama/logging"
)

// Version can be set at build time
var Version string

// EstimateVRAMForModel is the main function that other packages can call
func EstimateVRAMForModel(modelName string, vram float64, contextSize int, quantLevel, kvQuant string) (*VRAMEstimation, error) {
	if modelName == "" {
		return nil, fmt.Errorf("error: Model name is required")
	}

	if strings.Contains(modelName, ":") {
		ollamaHost := os.Getenv("OLLAMA_HOST")
		if ollamaHost == "" {
			logging.InfoLogger.Println("Warning: OLLAMA_HOST environment variable is not set. Using default http://localhost:11434")
			os.Setenv("OLLAMA_HOST", "http://localhost:11434")
		}
	}

	// Use default values if not provided
	if vram == 0 {
		vram = DefaultVRAM
	}
	if contextSize == 0 {
		contextSize = DefaultContextSize
	}
	if quantLevel == "" {
		quantLevel = DefaultQuantLevel
	}
	if kvQuant == "" {
		kvQuant = "fp16"
	}

	// Estimate the VRAM
	estimation, err := EstimateVRAM(
		&modelName,
		contextSize,
		KVCacheQuantisation(kvQuant),
		vram,
		quantLevel,
	)
	if err != nil {
		return nil, fmt.Errorf("error estimating VRAM: %v", err)
	}

	return estimation, nil
}

// GenerateQuantTableForModel generates a quant table for a model
func GenerateQuantTableForModel(modelName string, vram float64) (QuantResultTable, error) {
	var ollamaModelInfo *OllamaModelInfo
	var err error

	if strings.Contains(modelName, ":") {
		ollamaModelInfo, err = FetchOllamaModelInfo(modelName)
		if err != nil {
			return QuantResultTable{}, fmt.Errorf("error fetching Ollama model info: %v", err)
		}
	}

	table, err := GenerateQuantTable(modelName, vram, ollamaModelInfo)
	if err != nil {
		return QuantResultTable{}, fmt.Errorf("error generating quant table: %v", err)
	}
	return table, nil
}

// GetRecommendations returns the quant recommendations for different context sizes
func GetRecommendations(estimation *VRAMEstimation) map[int]string {
	return estimation.Recommendations
}

// GetMaxContextSize returns the maximum context size for the given VRAM and quantization
func GetMaxContextSize(estimation *VRAMEstimation) int {
	return estimation.MaxContextSize
}

// GetMaximumQuant returns the maximum quantization level for the given context size
func GetMaximumQuant(estimation *VRAMEstimation) string {
	return estimation.MaximumQuant
}

// PrintEstimationResults formats and returns the estimation results as a string
func PrintEstimationResults(estimation *VRAMEstimation) string {
	return fmt.Sprintf("Estimation Results:\n"+
		"Model: %s\n"+
		"Estimated vRAM Required For A Context Size Of %d: %.2f GB\n"+
		"Model Fits In Available vRAM (%.2f GB): %t\n"+
		"Max Context Size For vRAM At Supplied Quant (BPW: %s): %d\n"+
		"Maximum Quantisation For Provided Context Size Of %d: %s\n",
		estimation.ModelName,
		estimation.ContextSize,
		estimation.EstimatedVRAM,
		estimation.AvailableVRAM,
		estimation.FitsAvailable,
		estimation.QuantLevel,
		estimation.MaxContextSize,
		estimation.ContextSize,
		estimation.MaximumQuant)
}
