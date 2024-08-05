package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/sammcj/gollama/logging"
)

var Version string

func main() {
  var modelName string
  flag.StringVar(&modelName, "model", "", "Huggingface/ModelID or Ollama:modelName")
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

  // If no flags are provided and there's an argument, assume it's the model name
  if flag.NFlag() == 0 && len(flag.Args()) > 0 {
      modelName = flag.Args()[0]
  }

  if modelName == "" {
      fmt.Println("Error: Model name is required. Use --model or provide it as the first argument.")
      os.Exit(1)
  }


  if strings.Contains(modelName, ":") {
    ollamaHost := os.Getenv("OLLAMA_HOST")
    if ollamaHost == "" {
        fmt.Println("Warning: OLLAMA_HOST environment variable is not set. Using default http://localhost:11434")
        os.Setenv("OLLAMA_HOST", "http://localhost:11434")
    }
  }

  // Get the available VRAM
	estimation, err := EstimateVRAM(
		&modelName,
		*contextSize,
		KVCacheQuantisation(*kvQuant),
		*vram,
		*quantLevel,
	)
	if err != nil {
    logging.ErrorLogger.Println("Error estimating VRAM:", err)
		fmt.Printf("Error estimating VRAM: %v\n", err)
		os.Exit(1)
	}

  // Generate and print the quant estimation table
  table, err := GenerateQuantTable(modelName, *vram, estimation.ollamaModelInfo)
  if err != nil {
      fmt.Printf("Error generating quant table: %v\n", err)
      os.Exit(1)
  }
  fmt.Println(PrintFormattedTable(table))

	// Print the estimation results
	fmt.Printf("Model: %s\n", estimation.ModelName)
	fmt.Printf("Context Size: %d\n", estimation.ContextSize)
	fmt.Printf("Estimated VRAM: %.2f GB\n", estimation.EstimatedVRAM)
	fmt.Printf("Fits Available VRAM: %v\n", estimation.FitsAvailable)
	fmt.Printf("Max Context Size: %d\n", estimation.MaxContextSize)
	fmt.Printf("Recommended Quantisation: %s\n", estimation.RecommendedQuant)


}
