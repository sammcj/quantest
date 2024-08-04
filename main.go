package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/sammcj/gollama/logging"
)

var Version string

func main() {
	modelName := flag.String("model", "", "Hugginface/ModelID or Ollama:modelName")
	vram := flag.Float64("vram", DefaultVRAM, "Available VRAM in GB")
  contextSize := flag.Int("context", DefaultContextSize, "Optional context size")
  quantLevel := flag.String("quant", DefaultQuantLevel, "Optional quantisation level")
  kvQuant := flag.String("kvQuant", "fp16", "Optional KV Cache quantisation level")
	versionFlag := flag.Bool("v", false, "Print the version and exit")

	flag.Parse()

	if *versionFlag {
		fmt.Println(Version)
		os.Exit(0)
	}


	estimation, err := EstimateVRAM(
		modelName,
		*contextSize,
		*kvQuant,
		*vram,
		*quantLevel,
	)
	if err != nil {
    logging.ErrorLogger.Println("Error estimating VRAM:", err)
		fmt.Printf("Error estimating VRAM: %v\n", err)
		os.Exit(1)
	}

	// Print the estimation results
	fmt.Printf("Model: %s\n", estimation.ModelName)
	fmt.Printf("Context Size: %d\n", estimation.ContextSize)
	fmt.Printf("Estimated VRAM: %.2f GB\n", estimation.EstimatedVRAM)
	fmt.Printf("Fits Available VRAM: %v\n", estimation.FitsAvailable)
	fmt.Printf("Max Context Size: %d\n", estimation.MaxContextSize)
	fmt.Printf("Recommended Quantisation: %s\n", estimation.RecommendedQuant)

	// Generate and print the quant estimation table
	table, err := GenerateQuantTable(*modelName, vram, nil)
	if err != nil {
		fmt.Printf("Error generating quant table: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(PrintFormattedTable(table))
}
