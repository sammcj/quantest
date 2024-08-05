package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/sammcj/quantest"
)

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
	// If this is where GetHFModelConfig or EstimateVRAMForModel is called:
	estimation, err := quantest.EstimateVRAMForModel(modelName, *vram, *contextSize, *quantLevel, *kvQuant)
	if err != nil {
		handleError(err, modelName)
		os.Exit(1)
	}

	// Generate and print the quant estimation table
	table, err := quantest.GenerateQuantTable(estimation.ModelConfig, *vram)
	if err != nil {
		// fmt.Printf("DEBUG: Error generating quant table: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(quantest.PrintFormattedTable(table))

	// Print the recommendations
	fmt.Println("\nMaximum quants for context sizes:\n---")
	contextSizes := []int{2048, 8192, 16384, 32768, 49152, 65536}
	for _, ctxSize := range contextSizes {
		if ctxSize == estimation.ContextSize {
			fmt.Printf("Context %d (User Specified): %s \n", ctxSize, estimation.MaximumQuant)
		} else if quant, ok := estimation.Recommendations[ctxSize]; ok {
			fmt.Printf("Context %d: %s\n", ctxSize, quant)
		} else {
			fmt.Printf("Context %d: Calculation not available\n", ctxSize)
		}
	}

	// Print the estimation results
	fmt.Printf("\nEstimation Results:\n")
	fmt.Printf("Model: %s\n", estimation.ModelName)
	fmt.Printf("Estimated vRAM Required For A Context Size Of %d: %.2f GB\n", estimation.ContextSize, estimation.EstimatedVRAM)
	fmt.Printf("Fits Available vRAM: %v\n", estimation.FitsAvailable)
	fmt.Printf("Max Context Size: %d\n", estimation.MaxContextSize)
	fmt.Printf("Maximum Quantisation: %s\n", estimation.MaximumQuant)
}

func handleError(err error, modelName string) {
	fmt.Printf("Error processing model '%s':\n", modelName)
	fmt.Printf("%v\n", err)

	if strings.Contains(err.Error(), "Ollama API returned non-OK status") {
		fmt.Println("\nPossible issues:")
		fmt.Println("1. Ollama is not running. Try starting it with 'ollama serve'")
		fmt.Println("2. The model is not available in your Ollama installation. Try 'ollama list' to see available models")
		fmt.Println("3. There's a mismatch between the model name and what Ollama expects. Try using just the base model name without quantization info")
		fmt.Println("\nFor more detailed logs, run the command with the --debug flag")
	}
}
