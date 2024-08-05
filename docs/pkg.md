func EstimateVRAM(
	modelName *string,
	contextSize int,
	kvCacheQuant KVCacheQuantisation,
	availableVRAM float64,
	quantLevel string,
) (*VRAMEstimation, error)
    EstimateVRAM calculates VRAM usage for a given model configuration.

    Parameters:
      - modelName: A pointer to a string representing the model name
        (Huggingface/ModelID or Ollama:modelName).
      - contextSize: An integer representing the context size.
      - kvCacheQuant: The KV cache quantization level.
      - availableVRAM: A float64 representing the available VRAM in GB.
      - quantLevel: A string representing the quantization level.

    Returns:
      - *VRAMEstimation: A pointer to a VRAMEstimation struct containing the
        estimation results.
      - error: An error if the estimation fails.

    Example:

        estimation, err := quantest.EstimateVRAM(
        	&modelName,
        	8192,
        	quantest.KVCacheFP16,
        	24.0,
        	"Q4_K_M",
        )
        if err != nil {
        	log.Fatal(err)
        }
        fmt.Printf("Max Context Size: %d\n", estimation.MaxContextSize)



func GenerateQuantTable(modelID string, fitsVRAM float64, ollamaModelInfo *OllamaModelInfo) (QuantResultTable, error)
    GenerateQuantTable generates a quantisation table for a given model.

    Parameters:
      - modelID: A string representing the model ID.
      - fitsVRAM: A float64 representing the available VRAM in GB.
      - ollamaModelInfo: A pointer to an OllamaModelInfo struct.

    Returns:
      - QuantResultTable: A QuantResultTable struct containing the quantisation
        results.
      - error: An error if the quantisation fails.

    Example:

        table, err := GenerateQuantTable("llama3.1", 24.0, nil)
        if err != nil {
        	log.Fatal(err)
        }



type OllamaModelInfo struct {
	Details struct {
		ParentModel       string   `json:"parent_model"`
		Format            string   `json:"format"`
		Family            string   `json:"family"`
		Families          []string `json:"families"`
		ParameterSize     string   `json:"parameter_size"`
		QuantizationLevel string   `json:"quantization_level"`
	} `json:"details"`
	ModelInfo struct {
		Architecture         string `json:"general.architecture"`
		ParameterCount       int64  `json:"general.parameter_count"`
		ContextLength        int    `json:"llama.context_length"`
		AttentionHeadCount   int    `json:"llama.attention.head_count"`
		AttentionHeadCountKV int    `json:"llama.attention.head_count_kv"`
		EmbeddingLength      int    `json:"llama.embedding_length"`
		FeedForwardLength    int    `json:"llama.feed_forward_length"`
		RopeDimensionCount   int    `json:"llama.rope.dimension_count"`
		VocabSize            int    `json:"llama.vocab_size"`
	} `json:"model_info"`
}
    OllamaModelInfo represents the model information returned by Ollama.

func FetchOllamaModelInfo(modelName string) (*OllamaModelInfo, error)


func GetModelConfig(modelID string) (ModelConfig, error)
    GetModelConfig retrieves and parses the model configuration from Huggingface

    Parameters:
      - modelID: A string representing the model ID.

    Returns:
      - ModelConfig: A ModelConfig struct containing the model configuration.
      - error: An error if the request fails.

    Example:

        config, err := GetModelConfig("meta/llama3.1")
        if err != nil {
        	log.Fatal(err)
        }



