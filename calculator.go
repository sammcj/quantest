// File: vramestimator/calculator.go

package main

import (
	"fmt"
	"math"
	"slices"
	"sort"

	"github.com/sammcj/gollama/logging"
)

// CalculateBPW calculates the best BPW for a given memory and context constraint
func CalculateBPW(modelID string, memory float64, context int, kvCacheQuant KVCacheQuantisation, quantType string, ollamaModelInfo *OllamaModelInfo) (interface{}, error) {
  logging.DebugLogger.Println("Calculating BPW...")

  contextSizes := []int{2048, 8192, 16384, 32768, 49152, 65536}
  if !slices.Contains(contextSizes, context) {
      contextSizes = append(contextSizes, context)
      sort.Ints(contextSizes)
  }
  bestQuants := make(map[int]string)

  // Function to find best quant for a given context size
  findBestQuant := func(ctxSize int) string {
      var bestQuant string
      maxBPW := 0.0

      for quantName, bpw := range GGUFMapping {
          vram, err := CalculateVRAM(modelID, bpw, ctxSize, kvCacheQuant, ollamaModelInfo)
          if err != nil {
              logging.ErrorLogger.Printf("Error calculating VRAM for %s: %v", quantName, err)
              continue
          }

          if vram <= memory && bpw > maxBPW {
              maxBPW = bpw
              bestQuant = quantName
          }
      }

      return bestQuant
  }

  // Find best quants for all context sizes
  for _, ctxSize := range contextSizes {
      bestQuants[ctxSize] = findBestQuant(ctxSize)
  }

  // Print best quants for each context size
  fmt.Println("\nRecommended quantizations for different context sizes:")
  for _, ctxSize := range contextSizes {
      if quant, ok := bestQuants[ctxSize]; ok && quant != "" {
          if ctxSize == context {
              fmt.Printf("Context %d: %s (User Specified)\n", ctxSize, quant)
          } else {
              fmt.Printf("Context %d: %s\n", ctxSize, quant)
          }
      } else {
          if ctxSize == context {
              fmt.Printf("Context %d: No suitable quantization found\n", ctxSize)
          } else {
              fmt.Printf("Context %d: No suitable quantization found\n", ctxSize)
          }
      }
  }

  if bestQuant, ok := bestQuants[context]; ok && bestQuant != "" {
      return bestQuant, nil
  }

  return nil, fmt.Errorf("no suitable BPW found for the given memory constraint and context size")
}

// CalculateVRAM calculates the VRAM usage for a given model and configuration
func CalculateVRAM(modelID string, bpw float64, context int, kvCacheQuant KVCacheQuantisation, ollamaModelInfo *OllamaModelInfo) (float64, error) {
	logging.DebugLogger.Println("Calculating VRAM usage...")

	var config ModelConfig
	var err error

	if ollamaModelInfo != nil {
		// Use Ollama model information
		config = ModelConfig{
			NumParams:             float64(ollamaModelInfo.ModelInfo.ParameterCount) / 1e9, // Convert to billions
			MaxPositionEmbeddings: ollamaModelInfo.ModelInfo.ContextLength,
			NumHiddenLayers:       0, // Not provided in Ollama API, might need to be inferred
			HiddenSize:            ollamaModelInfo.ModelInfo.EmbeddingLength,
			NumKeyValueHeads:      ollamaModelInfo.ModelInfo.AttentionHeadCountKV,
			NumAttentionHeads:     ollamaModelInfo.ModelInfo.AttentionHeadCount,
			IntermediateSize:      ollamaModelInfo.ModelInfo.FeedForwardLength,
			VocabSize:             ollamaModelInfo.ModelInfo.VocabSize,
		}

		// Parse BPW from quantization level if not provided
		if bpw == 0 {
			bpw, err = ParseBPWOrQuant(ollamaModelInfo.Details.QuantizationLevel)
			if err != nil {
				return 0, fmt.Errorf("error parsing BPW from Ollama quantization level: %v", err)
			}
		}
	} else {
		// Use Hugging Face model information
		config, err = GetModelConfig(modelID)
		if err != nil {
			return 0, err
		}
	}

	bpwValues := GetBPWValues(bpw, kvCacheQuant)

	if context == 0 {
		context = config.MaxPositionEmbeddings
	}

	vram := CalculateVRAMRaw(config, bpwValues, context, 1, true)
	return math.Round(vram*100) / 100, nil
}

// CalculateContext calculates the maximum context for a given memory constraint
func CalculateContext(modelID string, memory, bpw float64, kvCacheQuant KVCacheQuantisation, ollamaModelInfo *OllamaModelInfo) (int, error) {
	logging.DebugLogger.Println("Calculating context...")

	var maxContext int
	if ollamaModelInfo != nil {
		maxContext = ollamaModelInfo.ModelInfo.ContextLength
	} else {
		config, err := GetModelConfig(modelID)
		if err != nil {
			return 0, err
		}
		maxContext = config.MaxPositionEmbeddings
	}

	minContext := 512
	low, high := minContext, maxContext
	for low < high {
		mid := (low + high + 1) / 2
		vram, err := CalculateVRAM(modelID, bpw, mid, kvCacheQuant, ollamaModelInfo)
		if err != nil {
			return 0, err
		}
		if vram > memory {
			high = mid - 1
		} else {
			low = mid
		}
	}

	context := low
	for context <= maxContext {
		vram, err := CalculateVRAM(modelID, bpw, context, kvCacheQuant, ollamaModelInfo)
		if err != nil {
			return 0, err
		}
		if vram >= memory {
			break
		}
		context += 100
	}

	return context - 100, nil
}

// CalculateVRAMRaw calculates the raw VRAM usage
func CalculateVRAMRaw(config ModelConfig, bpwValues BPWValues, context int, numGPUs int, gqa bool) float64 {
	logging.DebugLogger.Println("Calculating VRAM usage...")

	cudaSize := float64(CUDASize * numGPUs)
	paramsSize := config.NumParams * 1e9 * (bpwValues.BPW / 8)

	kvCacheSize := float64(context*2*config.NumHiddenLayers*config.HiddenSize) * (bpwValues.KVCacheBPW / 8)
	if gqa {
		kvCacheSize *= float64(config.NumKeyValueHeads) / float64(config.NumAttentionHeads)
	}

	bytesPerParam := bpwValues.BPW / 8
	lmHeadBytesPerParam := bpwValues.LMHeadBPW / 8

	headDim := float64(config.HiddenSize) / float64(config.NumAttentionHeads)
	attentionInput := bytesPerParam * float64(context*config.HiddenSize)

	q := bytesPerParam * float64(context) * headDim * float64(config.NumAttentionHeads)
	k := bytesPerParam * float64(context) * headDim * float64(config.NumKeyValueHeads)
	v := bytesPerParam * float64(context) * headDim * float64(config.NumKeyValueHeads)

	softmaxOutput := lmHeadBytesPerParam * float64(config.NumAttentionHeads*context)
	softmaxDropoutMask := float64(config.NumAttentionHeads * context)
	dropoutOutput := lmHeadBytesPerParam * float64(config.NumAttentionHeads*context)

	outProjInput := lmHeadBytesPerParam * float64(context*config.NumAttentionHeads) * headDim
	attentionDropout := float64(context * config.HiddenSize)

	attentionBlock := attentionInput + q + k + softmaxOutput + v + outProjInput + softmaxDropoutMask + dropoutOutput + attentionDropout

	mlpInput := bytesPerParam * float64(context*config.HiddenSize)
	activationInput := bytesPerParam * float64(context*config.IntermediateSize)
	downProjInput := bytesPerParam * float64(context*config.IntermediateSize)
	dropoutMask := float64(context * config.HiddenSize)
	mlpBlock := mlpInput + activationInput + downProjInput + dropoutMask

	layerNorms := bytesPerParam * float64(context*config.HiddenSize*2)
	activationsSize := attentionBlock + mlpBlock + layerNorms

	outputSize := lmHeadBytesPerParam * float64(context*config.VocabSize)

	vramBits := cudaSize + paramsSize + activationsSize + outputSize + kvCacheSize

	return bitsToGB(vramBits)
}
