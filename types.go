// File: quantest/types.go

package quantest

import "sync"

// ModelConfig represents the configuration of a model.
type ModelConfig struct {
  ModelName             string  `json:"-"`
  NumParams             float64 `json:"-"`
  MaxPositionEmbeddings int     `json:"max_position_embeddings"`
  NumHiddenLayers       int     `json:"num_hidden_layers"`
  HiddenSize            int     `json:"hidden_size"`
  NumKeyValueHeads      int     `json:"num_key_value_heads"`
  NumAttentionHeads     int     `json:"num_attention_heads"`
  IntermediateSize      int     `json:"intermediate_size"`
  VocabSize             int     `json:"vocab_size"`
  IsOllama              bool    `json:"-"`
}

// BPWValues represents the bits per weight values for a given quantisation.
type BPWValues struct {
	BPW        float64
	LMHeadBPW  float64
	KVCacheBPW float64
}

// ContextVRAM represents the VRAM usage for a given context quantisation.
type ContextVRAM struct {
	VRAM     float64
	VRAMQ8_0 float64
	VRAMQ4_0 float64
}

type QuantResult struct {
	QuantType string
	BPW       float64
	Contexts  map[int]ContextVRAM
}

// QuantResultTable represents the results of a quantisation analysis.
type QuantResultTable struct {
	ModelID  string
	Results  []QuantResult
	FitsVRAM float64
}

// VRAMEstimation represents the results of a VRAM estimation.
type VRAMEstimation struct {
  ModelName       string
  ModelConfig     ModelConfig
  ContextSize     int
  KVCacheQuant    KVCacheQuantisation
  AvailableVRAM   float64
  QuantLevel      string
  EstimatedVRAM   float64
  FitsAvailable   bool
  MaxContextSize  int
  MaximumQuant    string
  Recommendations map[int]string
  ollamaModelInfo *OllamaModelInfo
}

// OllamaModelInfo represents the model information returned by Ollama.
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

// KVCacheQuantisation represents the KV cache quantisation options.
type KVCacheQuantisation string

// Quantisation represents the KV cache quantisation options.
const (
	KVCacheFP16 KVCacheQuantisation = "fp16"
	KVCacheQ8_0 KVCacheQuantisation = "q8_0"
	KVCacheQ4_0 KVCacheQuantisation = "q4_0"
)

// Default values for VRAM, context size and quantisation level if not provided.
const (
	DefaultVRAM        = 24.0
	DefaultContextSize = 8192
	DefaultQuantLevel  = "Q4_K_M"
)

// colourMap contains the colours for the quantisation types.
var colourMap = []string{
	"#ff0000", // red
	"#00ff00", // green
}

// GGUFMapping maps GGUF quantisation types to their corresponding bits per weight
var GGUFMapping = map[string]float64{
	"Q8_0":    8.5,
	"Q6_K":    6.59,
	"Q5_K_L":  5.75,
	"Q5_K_M":  5.69,
	"Q5_K_S":  5.54,
	"Q5_0":    5.54,
	"Q4_K_L":  4.9,
	"Q4_K_M":  4.85,
	"Q4_K_S":  4.58,
	"Q4_0":    4.55,
	"IQ4_NL":  4.5,
	"Q3_K_L":  4.27,
	"IQ4_XS":  4.25,
	"Q3_K_M":  3.91,
	"IQ3_M":   3.7,
	"IQ3_S":   3.5,
	"Q3_K_S":  3.5,
	"Q2_K":    3.35,
	"IQ3_XS":  3.3,
	"IQ3_XXS": 3.06,
	"IQ2_M":   2.7,
	"IQ2_S":   2.5,
	"IQ2_XS":  2.31,
	"IQ2_XXS": 2.06,
	"IQ1_S":   1.56,
}

// EXL2Options contains the EXL2 quantisation options
var EXL2Options []float64

var (
	modelConfigCache = make(map[string]ModelConfig)
	cacheMutex       sync.RWMutex
)
