package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/sammcj/gollama/logging"
)

func FetchOllamaModelInfo(modelName string) (*OllamaModelInfo, error) {
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
	logging.DebugLogger.Println("Response body:", modelInfo)

	return modelInfo, nil
}
