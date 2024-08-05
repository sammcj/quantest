// File: quantest/calculator.go

package quantest

import (
	"fmt"
	"math"
	"slices"
	"sort"

	"github.com/sammcj/gollama/logging"
)

// QuantRecommendations holds the recommended quantizations for different context sizes
type QuantRecommendations struct {
  UserContext int
  Recommendations map[int]string
}

// CalculateBPW calculates the best BPW for a given memory and context constraint
func CalculateBPW(config ModelConfig, memory float64, context int, kvCacheQuant KVCacheQuantisation, quantType string) (interface{}, QuantRecommendations, error) {
  // fmt.Printf("DEBUG: CalculateBPW called with config: %+v, memory: %.2f, context: %d, kvCacheQuant: %v, quantType: %s\n", config, memory, context, kvCacheQuant, quantType)

  contextSizes := []int{2048, 8192, 16384, 32768, 49152, 65536}
  if !slices.Contains(contextSizes, context) {
      contextSizes = append(contextSizes, context)
      sort.Ints(contextSizes)
  }
  bestQuants := make(map[int]string)

  // Find best quantisation for a given context size
  findBestQuant := func(ctxSize int) string {
      var bestQuant string
      maxBPW := 0.0


      for quantName, bpw := range GGUFMapping {
        // fmt.Printf("DEBUG: Trying quant %s with BPW %.2f\n", quantName, bpw)
        vram, err := CalculateVRAM(config, bpw, ctxSize, kvCacheQuant)
        if err != nil {
            // fmt.Printf("DEBUG: Error calculating VRAM for %s: %v\n", quantName, err)
            continue
        }
        // fmt.Printf("DEBUG: Calculated VRAM for %s: %.2f GB\n", quantName, vram)

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

  recommendations := QuantRecommendations{
      UserContext: context,
      Recommendations: bestQuants,
  }

  if bestQuant, ok := bestQuants[context]; ok && bestQuant != "" {
      return bestQuant, recommendations, nil
  }

  return nil, recommendations, fmt.Errorf("no suitable BPW found for the given memory constraint and context size")
}

// CalculateVRAM calculates the VRAM usage for a given model and configuration
//
// Parameters:
//  - modelName: A string representing the model name.
//  - bpw: A float64 representing the bits per weight.
//  - contextSize: An integer representing the context size.
//  - kvCacheQuant: The KV cache quantization level.
//  - ollamaModelInfo: A pointer to an OllamaModelInfo struct.
//
// Returns:
//  - float64: A float64 representing the VRAM usage in GB.
//  - error: An error if the calculation fails.
//
// Example:
//  vram, _ := CalculateVRAM("llama3.1", 24.0, 8192, KVCacheFP16, nil)
func CalculateVRAM(config ModelConfig, bpw float64, context int, kvCacheQuant KVCacheQuantisation) (float64, error) {
  // fmt.Printf("DEBUG: CalculateVRAM called with config: %+v, bpw: %.2f, context: %d, kvCacheQuant: %v\n", config, bpw, context, kvCacheQuant)


	bpwValues := GetBPWValues(bpw, kvCacheQuant)

	if context == 0 {
		context = config.MaxPositionEmbeddings
	}

  vram := CalculateVRAMRaw(config, bpwValues, context, 1, true)
  // fmt.Printf("DEBUG: Calculated raw VRAM: %.2f GB\n", vram)

  return math.Round(vram*100) / 100, nil
}


// CalculateContext calculates the maximum context for a given memory constraint
//
// Parameters:
//  - modelID: A string representing the model ID.
//  - memory: A float64 representing the available VRAM in GB.
//  - bpw: A float64 representing the bits per weight.
//  - kvCacheQuant: The KV cache quantization level.
//  - ollamaModelInfo: A pointer to an OllamaModelInfo struct.
//
// Returns:
//  - int: An integer representing the maximum context size.
//  - error: An error if the calculation fails.
//
// Example:
//  context, err := CalculateContext("llama3.1", 24.0, 8.0, KVCacheFP16, nil)
//  if err != nil {
//      log.Fatal(err)
//  }
func CalculateContext(config ModelConfig, memory, bpw float64, kvCacheQuant KVCacheQuantisation) (int, error) {
  	logging.DebugLogger.Println("Calculating context...")

	var maxContext int

		config, err := GetHFModelConfig(config.ModelName)
		if err != nil {
			return 0, err
		}
		maxContext = config.MaxPositionEmbeddings


	minContext := 512
	low, high := minContext, maxContext
	for low < high {
		mid := (low + high + 1) / 2
		vram, err := CalculateVRAM(config, bpw, mid, kvCacheQuant)
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
		vram, err := CalculateVRAM(config, bpw, context, kvCacheQuant)
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

// CalculateVRAMRaw calculates the raw VRAM usage for a given model configuration
//
// Parameters:
//  - config: A ModelConfig struct containing the model configuration.
//  - bpwValues: A BPWValues struct containing the bits per weight values.
//  - context: An integer representing the context size.
//  - numGPUs: An integer representing the number of GPUs.
//  - gqa: A boolean indicating whether the model is GQA.
//
// Returns:
//  - float64: A float64 representing the VRAM usage in GB.
//
// Example:
//  vram := CalculateVRAMRaw(config, bpwValues, 8192, 1, true)
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
