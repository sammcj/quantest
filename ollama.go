package quantest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/sammcj/gollama/logging"
)

func GetOllamaModelConfig(modelID string) (ModelConfig, error) {
	ollamaInfo, err := FetchOllamaModelInfo(modelID)
	if err != nil {
		return ModelConfig{}, fmt.Errorf("error fetching Ollama model info: %w", err)
	}

	return ModelConfig{
		ModelName:             modelID,
		NumParams:             float64(ollamaInfo.ModelInfo.ParameterCount) / 1e9,
		MaxPositionEmbeddings: ollamaInfo.ModelInfo.ContextLength,
		NumHiddenLayers:       estimateHiddenLayers(ollamaInfo),
		HiddenSize:            ollamaInfo.ModelInfo.EmbeddingLength,
		NumKeyValueHeads:      ollamaInfo.ModelInfo.AttentionHeadCountKV,
		NumAttentionHeads:     ollamaInfo.ModelInfo.AttentionHeadCount,
		IntermediateSize:      ollamaInfo.ModelInfo.FeedForwardLength,
		VocabSize:             ollamaInfo.ModelInfo.VocabSize,
		IsOllama:              true,
	}, nil
}

var (
	ollamaModelInfoCache = make(map[string]*OllamaModelInfo)
	ollamaCacheMutex     sync.RWMutex
)

// TODO: Function to estimate the number of hidden layers based on OllamaModelInfo
func estimateHiddenLayers(*OllamaModelInfo) int {
	// Implement logic here to estimate the number of hidden layers
	return 0
}

// OllamaModelInfo gets model information from Ollama.
//
// Parameters:
//   - modelName: A string representing the model name.
//
// Returns:
//   - *OllamaModelInfo: A pointer to an OllamaModelInfo struct containing the model information.
//   - error: An error if the request fails.
//
// Example:
//
//	modelInfo, err := FetchOllamaModelInfo("llama3.1:8b")
//	if err != nil {
//		log.Fatal(err)
//	}
//	fmt.Printf("Model Info: %+v\n", modelInfo)
func FetchOllamaModelInfo(modelName string) (*OllamaModelInfo, error) {
	ollamaCacheMutex.RLock()
	if cachedInfo, ok := ollamaModelInfoCache[modelName]; ok {
		ollamaCacheMutex.RUnlock()
		return cachedInfo, nil
	}
	ollamaCacheMutex.RUnlock()
	apiURL := os.Getenv("OLLAMA_HOST")
	if apiURL == "" {
		apiURL = "http://localhost:11434" // Default Ollama API URL
	}

	logging.InfoLogger.Println("Using Ollama API URL:", apiURL)
	fmt.Println("Using Ollama API URL:", apiURL)

	url := fmt.Sprintf("%s/api/show", apiURL)
	payload := []byte(fmt.Sprintf(`{"model": "%s"}`, modelName))

	logging.InfoLogger.Println("Sending request to:", url)
	logging.DebugLogger.Println("With payload:", string(payload))

	client := &http.Client{Timeout: 10 * time.Second}

	resp, err := client.Post(url, "application/json", bytes.NewBuffer(payload))
	if err != nil {
		return nil, fmt.Errorf("error making request to Ollama API: %v", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama API returned non-OK status: %d, body: %s", resp.StatusCode, string(body))
	}

	var response struct {
		Details   OllamaModelInfo        `json:"details"`
		ModelInfo map[string]interface{} `json:"model_info"`
		Config    map[string]interface{} `json:"config"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("error decoding Ollama API response: %v", err)
	}

	modelInfo := &OllamaModelInfo{
		Details: response.Details.Details,
	}

	// Parse the ModelInfo fields
	if paramCount, ok := response.ModelInfo["general.parameter_count"].(float64); ok {
		modelInfo.ModelInfo.ParameterCount = int64(paramCount)
	}
	if contextLength, ok := response.ModelInfo["llama.context_length"].(float64); ok {
		modelInfo.ModelInfo.ContextLength = int(contextLength)
	}
	if headCount, ok := response.ModelInfo["llama.attention.head_count"].(float64); ok {
		modelInfo.ModelInfo.AttentionHeadCount = int(headCount)
	}
	if headCountKV, ok := response.ModelInfo["llama.attention.head_count_kv"].(float64); ok {
		modelInfo.ModelInfo.AttentionHeadCountKV = int(headCountKV)
	}
	if embeddingLength, ok := response.ModelInfo["llama.embedding_length"].(float64); ok {
		modelInfo.ModelInfo.EmbeddingLength = int(embeddingLength)
	}
	if feedForwardLength, ok := response.ModelInfo["llama.feed_forward_length"].(float64); ok {
		modelInfo.ModelInfo.FeedForwardLength = int(feedForwardLength)
	}
	if ropeDimensionCount, ok := response.ModelInfo["llama.rope.dimension_count"].(float64); ok {
		modelInfo.ModelInfo.RopeDimensionCount = int(ropeDimensionCount)
	}
	if vocabSize, ok := response.ModelInfo["llama.vocab_size"].(float64); ok {
		modelInfo.ModelInfo.VocabSize = int(vocabSize)
	}

	logging.DebugLogger.Println("Response status:", resp.Status)
	logging.DebugLogger.Println("Response body:", string(body))

	ollamaCacheMutex.Lock()
	ollamaModelInfoCache[modelName] = modelInfo
	ollamaCacheMutex.Unlock()

	return modelInfo, nil
}
