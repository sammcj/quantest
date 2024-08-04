package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/sammcj/gollama/logging"
)

func FetchOllamaModelInfo(modelName string) (*OllamaModelInfo, error) {
  apiURL := os.Getenv("OLLAMA_HOST")

  logging.DebugLogger.Println("Fetching Ollama model info for", modelName)

  fmt.Println("Fetching Ollama model info for", modelName)

	url := fmt.Sprintf("%s/api/show", apiURL)
	payload := []byte(fmt.Sprintf(`{"name": "%s"}`, modelName))

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(payload))
	if err != nil {
		return nil, fmt.Errorf("error making request to Ollama API: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama API returned non-OK status: %d", resp.StatusCode)
	}

	var modelInfo OllamaModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&modelInfo); err != nil {
		return nil, fmt.Errorf("error decoding Ollama API response: %v", err)
	}

	return &modelInfo, nil
}
