package main

import (
	"fmt"
	"strings"

	"github.com/sammcj/gollama/logging"
)

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

	estimatedVRAM, err := CalculateVRAM(*modelName, bpw, contextSize, kvCacheQuant, ollamaModelInfo)
	if err != nil {
		return nil, err
	}

	maxContextSize, err := CalculateContext(*modelName, availableVRAM, bpw, kvCacheQuant, ollamaModelInfo)
	if err != nil {
		return nil, err
	}

	maximumQuant, recommendations, err := CalculateBPW(*modelName, availableVRAM, contextSize, kvCacheQuant, "gguf", ollamaModelInfo)
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
