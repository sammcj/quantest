package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/sammcj/quantest"
)

// This is a basic CLI client for the quantest package.
// It takes a model name and optional flags to estimate the VRAM required for a model
// and generate a quantisation table for the model.
// It also provides recommendations for the maximum quantisation levels for different context sizes.
// Usage:
// quantest --model <model_name> --vram <vram_in_gb> --context <context_size> --quant <quant_level> --kvQuant <kv_quant_level>
// Example:
// quantest --model gpt2 --vram 16 --context 8192 --quant int8 --kvQuant fp16
func main() {
	var modelName string
	flag.StringVar(&modelName, "model", "", "Huggingface/ModelID or Ollama:modelName")
	vram := flag.Float64("vram", quantest.DefaultVRAM, "Available vRAM in GB")
	contextSize := flag.Int("context", quantest.DefaultContextSize, "Optional context size")
	quantLevel := flag.String("quant", quantest.DefaultQuantLevel, "Optional quantisation level")
	kvQuant := flag.String("kvQuant", "fp16", "Optional KV Cache quantisation level")
	versionFlag := flag.Bool("v", false, "Print the version and exit")

	flag.Parse()

	if *versionFlag {
		fmt.Println(quantest.Version)
		os.Exit(0)
	}

	// If no flags are provided and there's an argument, assume it's the model name
	if flag.NFlag() == 0 && len(flag.Args()) > 0 {
		modelName = flag.Args()[0]
	}

	if modelName == "" {
		fmt.Println("Error: Model name is required. Use --model or provide it as the first argument.")
		os.Exit(1)
	}

	// Estimate the VRAM
	estimation, err := quantest.EstimateVRAMForModel(modelName, *vram, *contextSize, *quantLevel, *kvQuant)
	if err != nil {
		fmt.Printf("Error estimating VRAM: %v\n", err)
		os.Exit(1)
	}

	// Generate and print the quant estimation table
	table, err := quantest.GenerateQuantTableForModel(modelName, *vram)
	if err != nil {
		fmt.Printf("Error generating quant table: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(quantest.PrintFormattedTable(table))

	// Print the recommendations
	fmt.Println("\nMaximum quants for context sizes:\n---")
	contextSizes := []int{2048, 8192, 16384, 32768, 49152, 65536}
	recommendations := quantest.GetRecommendations(estimation)
	for _, ctxSize := range contextSizes {
		if quant, ok := recommendations[ctxSize]; ok && quant != "" {
			if ctxSize == estimation.ContextSize {
				fmt.Printf("Context %d (User Specified): %s \n", ctxSize, quant)
			} else {
				fmt.Printf("Context %d: %s\n", ctxSize, quant)
			}
		} else {
			if ctxSize == estimation.ContextSize {
				fmt.Printf("Context %d: No suitable quantisation found\n", ctxSize)
			} else {
				fmt.Printf("Context %d: No suitable quantisation found\n", ctxSize)
			}
		}
	}

	// Print the estimation results
	fmt.Println(quantest.PrintEstimationResults(estimation))
}
