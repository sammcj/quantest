#### Dynamically Generated Interactive Menu ####

# Error Handling
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

# Name of this Makefile
MAKEFILE_NAME := $(lastword $(MAKEFILE_LIST))

# Special targets that should not be listed
EXCLUDE_LIST := menu all .PHONY

# Function to extract targets from the Makefile
define extract_targets
	$(shell awk -F: '/^[a-zA-Z0-9_-]+:/ {print $$1}' $(MAKEFILE_NAME) | grep -v -E '^($(EXCLUDE_LIST))$$')
endef

TARGETS := $(call extract_targets)

# Set default target to build
.DEFAULT_GOAL := build

.PHONY: $(TARGETS) menu all clean test

menu: ## Makefile Interactive Menu
	@# Check if fzf is installed
	@if command -v fzf >/dev/null 2>&1; then \
		echo "Using fzf for selection..."; \
		echo "$(TARGETS)" | tr ' ' '\n' | fzf > .selected_target; \
		target_choice=$$(cat .selected_target); \
	else \
		echo "fzf not found, using numbered menu:"; \
		echo "$(TARGETS)" | tr ' ' '\n' > .targets; \
		awk '{print NR " - " $$0}' .targets; \
		read -p "Enter choice: " choice; \
		target_choice=$$(awk 'NR == '$$choice' {print}' .targets); \
	fi; \
	if [ -n "$$target_choice" ]; then \
		$(MAKE) $$target_choice; \
	else \
		echo "Invalid choice"; \
	fi

# Default target
all: build

help: ## This help function
	@egrep '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

docs: ## Generate documentation
	@echo "Generating documentation..."
	@rm -f ./docs/cli.md && go run ./cmd/quantest/main.go --help 2> ./docs/cli.md
	@go doc EstimateVRAMForModel > ./docs/pkg.md
	@echo "Documentation generated"

clean: ## Clean
	rm -rf ./dist quantest*.zip quantest *.log

debug-server: ## Debug server
	dlv debug --headless --api-version=2 --listen=127.0.0.1:43000 .

debug-client: ## Debug client
	dlv connect 127.0.0.1:43000

# Targets (example targets listed below)
lint: ## Run lint
	gofmt -w -s .

test: ## Run test
	go test -v ./...

build: ## Run build
	@$(eval QUANTEST_VERSION := $(shell if [ -z "$(QUANTEST_VERSION)" ]; then echo "$(shell git describe --tags --abbrev=0 | sed 's/^v//')"; else echo "$(QUANTEST_VERSION)"; fi))
	@echo "Bumping version to: $(QUANTEST_VERSION)"
	@export QUANTEST_VERSION=$(QUANTEST_VERSION)
	@if [ "$(shell uname)" == "Darwin" ]; then \
		sed -i '' -e "s/Version = \".*\"/Version = \"$(QUANTEST_VERSION)\"/g" cmd/quantest/main.go ; \
	else \
		sed -i -e "s/Version = \".*\"/Version = \"$(QUANTEST_VERSION)\"/g" cmd/quantest/main.go ; \
	fi

	@go build -v -ldflags="-w -s -X 'main.Version=$(QUANTEST_VERSION)'" -o ./quantest cmd/quantest/main.go
	@echo "Build completed, run ./quantest"

ci: ## build for linux and macOS
	$(eval QUANTEST_VERSION := $(shell if [ -z "$(QUANTEST_VERSION)" ]; then echo "$(shell git describe --tags --abbrev=0 | sed 's/^v//')"; else echo "$(QUANTEST_VERSION)"; fi))
	@if [ "$(shell uname)" == "Darwin" ]; then \
		sed -i '' -e "s/Version = \".*\"/Version = \"$(QUANTEST_VERSION)\"/g" cmd/quantest/main.go ; \
	else \
		sed -i -e "s/Version = \".*\"/Version = \"$(QUANTEST_VERSION)\"/g" cmd/quantest/main.go ; \
	fi
	@echo "Building with version: $(QUANTEST_VERSION)"

	@mkdir -p ./dist/macos ./dist/linux_amd64 ./dist/linux_arm64
	GOOS=darwin GOARCH=arm64 go build -v -ldflags="-w -s -X 'main.Version=$(QUANTEST_VERSION)'" -o ./dist/macos/quantest cmd/quantest/main.go
	GOOS=linux GOARCH=amd64 go build -v -ldflags="-w -s -X 'main.Version=$(QUANTEST_VERSION)'" -o ./dist/linux_amd64/quantest cmd/quantest/main.go
	GOOS=linux GOARCH=arm64 go build -v -ldflags="-w -s -X 'main.Version=$(QUANTEST_VERSION)'" -o ./dist/linux_arm64/quantest cmd/quantest/main.go

	@zip -r quantest-macos.zip ./dist/macos/quantest
	@zip -r quantest-linux-amd64.zip ./dist/linux_amd64/quantest
	@zip -r quantest-linux-arm64.zip ./dist/linux_arm64/quantest

	@echo "Build completed"
	@echo "macOS: ./dist/macos/quantest"
	@echo "Linux (amd64): ./dist/linux_amd64/quantest"
	@echo "Linux (arm64): ./dist/linux_arm64/quantest"

install: ## Install latest
	go install github.com/sammcj/quantest@latest

run: ## Run
	@go run $(shell find *.go -not -name '*_test.go')
