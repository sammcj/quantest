package quantest

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"time"

	"github.com/sammcj/gollama/logging"
)

// DownloadFile downloads a file from a URL and saves it to the specified path
func DownloadFile(url, filePath string, headers map[string]string) error {
	if _, err := os.Stat(filePath); err == nil {
		logging.InfoLogger.Println("File already exists, skipping download")
		return nil
	}

	client := &http.Client{
		Timeout: 30 * time.Second,
	}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	for key, value := range headers {
		req.Header.Set(key, value)
		logging.DebugLogger.Printf("Setting header: %s: %s", key, value)
	}

	logging.DebugLogger.Printf("Sending request to: %s", req.URL.String())
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	logging.DebugLogger.Printf("Response status: %s", resp.Status)
	logging.DebugLogger.Printf("Response headers: %v", resp.Header)

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("bad status: %s, URL: %s, body: %s", resp.Status, url, string(body))
	}

	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	out, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

// GetHFModelConfig retrieves and parses the model configuration from Huggingface
//
// Parameters:
//   - modelID: A string representing the model ID.
//
// Returns:
//   - ModelConfig: A ModelConfig struct containing the model configuration.
//   - error: An error if the request fails.
//
// Example:
//
//	config, err := GetHFModelConfig("meta/llama3.1")
//	if err != nil {
//		log.Fatal(err)
//	}
func GetHFModelConfig(modelID string) (ModelConfig, error) {
	if modelID == "" {
		return ModelConfig{}, fmt.Errorf("empty model ID provided")
	}

	accessToken := os.Getenv("HUGGINGFACE_TOKEN")

	cacheMutex.RLock()
	if config, ok := modelConfigCache[modelID]; ok {
		cacheMutex.RUnlock()
		return config, nil
	}
	cacheMutex.RUnlock()

	baseDir := filepath.Join(os.Getenv("HOME"), ".cache", "huggingface", "hub", modelID)
	configPath := filepath.Join(baseDir, "config.json")
	indexPath := filepath.Join(baseDir, "model.safetensors.index.json")

	configFile, err := os.ReadFile(configPath)
	if err != nil {
		return ModelConfig{}, err
	}

	var config ModelConfig
	if err := json.Unmarshal(configFile, &config); err != nil {
		return ModelConfig{}, err
	}

	// Ensure the modelID is properly URL-encoded
	encodedModelID := url.PathEscape(modelID)
	configURL := fmt.Sprintf("https://huggingface.co/%s/raw/main/config.json", encodedModelID)
	indexURL := fmt.Sprintf("https://huggingface.co/%s/raw/main/model.safetensors.index.json", encodedModelID)

	logging.DebugLogger.Printf("Config URL: %s", configURL)
	logging.DebugLogger.Printf("Index URL: %s", indexURL)

	headers := make(map[string]string)
	if accessToken != "" {
		headers["Authorization"] = "Bearer " + accessToken
	}

	if err := DownloadFile(configURL, configPath, headers); err != nil {
		return ModelConfig{}, fmt.Errorf("failed to download config.json: %w", err)
	}

	indexFile, err := os.ReadFile(indexPath)
	if err != nil {
		return ModelConfig{}, err
	}

	var index struct {
		Metadata struct {
			TotalSize float64 `json:"total_size"`
		} `json:"metadata"`
	}
	if err := json.Unmarshal(indexFile, &index); err != nil {
		return ModelConfig{}, err
	}

	config.NumParams = index.Metadata.TotalSize / 2 / 1e9

	// Set the fields that are not in the JSON
	config.ModelName = modelID
	config.NumParams = index.Metadata.TotalSize / 2 / 1e9
	config.IsOllama = false

	cacheMutex.Lock()
	modelConfigCache[modelID] = config
	cacheMutex.Unlock()

	return config, nil
}
