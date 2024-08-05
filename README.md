# Quantest

LLM (v)RAM estimator tool and golang package for GGUF models from Ollama and Huggingface across various quantisation and context sizes.

At present quantest is in the process of being used as a library in my [Gollama](https://github.com/sammcj/gollama) and [Ingest](https://github.com/sammcj/ingest) projects.

## Usage

### CLI / Standalone

To use the package as a standalone cli tool, install the package with:

```bash
go install github.com/sammcj/quantest/cmd/quantest@latest
```

Then run the tool with:

```bash
quantest --model llama3.1:8b-instruct-q6_K --vram 12 --context 4096
Using Ollama API URL: http://localhost:11434
📊 VRAM Estimation for Model: llama3.1:8b-instruct-q6_K

| QUANT   | CTX  | BPW | 2K  | 8K              | 16K             | 32K             | 49K             | 64K |
| ------- | ---- | --- | --- | --------------- | --------------- | --------------- | --------------- |
| IQ1_S   | 1.56 | 2.2 | 2.8 | 3.7(3.7,3.7)    | 5.5(5.5,5.5)    | 7.3(7.3,7.3)    | 9.1(9.1,9.1)    |
| IQ2_XXS | 2.06 | 2.6 | 3.3 | 4.3(4.3,4.3)    | 6.1(6.1,6.1)    | 7.9(7.9,7.9)    | 9.8(9.8,9.8)    |
| IQ2_XS  | 2.31 | 2.9 | 3.6 | 4.5(4.5,4.5)    | 6.4(6.4,6.4)    | 8.2(8.2,8.2)    | 10.1(10.1,10.1) |
| IQ2_S   | 2.50 | 3.1 | 3.8 | 4.7(4.7,4.7)    | 6.6(6.6,6.6)    | 8.5(8.5,8.5)    | 10.4(10.4,10.4) |
| IQ2_M   | 2.70 | 3.2 | 4.0 | 4.9(4.9,4.9)    | 6.8(6.8,6.8)    | 8.7(8.7,8.7)    | 10.6(10.6,10.6) |
| IQ3_XXS | 3.06 | 3.6 | 4.3 | 5.3(5.3,5.3)    | 7.2(7.2,7.2)    | 9.2(9.2,9.2)    | 11.1(11.1,11.1) |
| IQ3_XS  | 3.30 | 3.8 | 4.5 | 5.5(5.5,5.5)    | 7.5(7.5,7.5)    | 9.5(9.5,9.5)    | 11.4(11.4,11.4) |
| Q2_K    | 3.35 | 3.9 | 4.6 | 5.6(5.6,5.6)    | 7.6(7.6,7.6)    | 9.5(9.5,9.5)    | 11.5(11.5,11.5) |
| Q3_K_S  | 3.50 | 4.0 | 4.8 | 5.7(5.7,5.7)    | 7.7(7.7,7.7)    | 9.7(9.7,9.7)    | 11.7(11.7,11.7) |
| IQ3_S   | 3.50 | 4.0 | 4.8 | 5.7(5.7,5.7)    | 7.7(7.7,7.7)    | 9.7(9.7,9.7)    | 11.7(11.7,11.7) |
| IQ3_M   | 3.70 | 4.2 | 5.0 | 6.0(6.0,6.0)    | 8.0(8.0,8.0)    | 9.9(9.9,9.9)    | 12.0(12.0,12.0) |
| Q3_K_M  | 3.91 | 4.4 | 5.2 | 6.2(6.2,6.2)    | 8.2(8.2,8.2)    | 10.2(10.2,10.2) | 12.2(12.2,12.2) |
| IQ4_XS  | 4.25 | 4.7 | 5.5 | 6.5(6.5,6.5)    | 8.6(8.6,8.6)    | 10.6(10.6,10.6) | 12.7(12.7,12.7) |
| Q3_K_L  | 4.27 | 4.7 | 5.5 | 6.5(6.5,6.5)    | 8.6(8.6,8.6)    | 10.7(10.7,10.7) | 12.7(12.7,12.7) |
| IQ4_NL  | 4.50 | 5.0 | 5.7 | 6.8(6.8,6.8)    | 8.9(8.9,8.9)    | 10.9(10.9,10.9) | 13.0(13.0,13.0) |
| Q4_0    | 4.55 | 5.0 | 5.8 | 6.8(6.8,6.8)    | 8.9(8.9,8.9)    | 11.0(11.0,11.0) | 13.1(13.1,13.1) |
| Q4_K_S  | 4.58 | 5.0 | 5.8 | 6.9(6.9,6.9)    | 8.9(8.9,8.9)    | 11.0(11.0,11.0) | 13.1(13.1,13.1) |
| Q4_K_M  | 4.85 | 5.3 | 6.1 | 7.1(7.1,7.1)    | 9.2(9.2,9.2)    | 11.4(11.4,11.4) | 13.5(13.5,13.5) |
| Q4_K_L  | 4.90 | 5.3 | 6.1 | 7.2(7.2,7.2)    | 9.3(9.3,9.3)    | 11.4(11.4,11.4) | 13.6(13.6,13.6) |
| Q5_0    | 5.54 | 5.9 | 6.8 | 7.8(7.8,7.8)    | 10.0(10.0,10.0) | 12.2(12.2,12.2) | 14.4(14.4,14.4) |
| Q5_K_S  | 5.54 | 5.9 | 6.8 | 7.8(7.8,7.8)    | 10.0(10.0,10.0) | 12.2(12.2,12.2) | 14.4(14.4,14.4) |
| Q5_K_M  | 5.69 | 6.1 | 6.9 | 8.0(8.0,8.0)    | 10.2(10.2,10.2) | 12.4(12.4,12.4) | 14.6(14.6,14.6) |
| Q5_K_L  | 5.75 | 6.1 | 7.0 | 8.1(8.1,8.1)    | 10.3(10.3,10.3) | 12.5(12.5,12.5) | 14.7(14.7,14.7) |
| Q6_K    | 6.59 | 7.0 | 8.0 | 9.4(9.4,9.4)    | 12.2(12.2,12.2) | 15.0(15.0,15.0) | 17.8(17.8,17.8) |
| Q8_0    | 8.50 | 8.8 | 9.9 | 11.4(11.4,11.4) | 14.4(14.4,14.4) | 17.4(17.4,17.4) | 20.3(20.3,20.3) |

Maximum quants for context sizes:
---
Context 2048: Q8_0
Context 8192: Q8_0
Context 16384: Q8_0
Context 32768: Q5_K_L
Context 49152: Q4_K_L
Context 65536: IQ3_M

Estimation Results:
---
Model: llama3.1:8b-instruct-q6_K
Estimated vRAM Required For A Context Size Of 4096: 5.55 GB
Model Fits In Available vRAM (12.00 GB): true
Max Context Size For vRAM At Supplied Quant (BPW: Q4_K_M): 54004
Maximum Quantisation For Provided Context Size Of 4096: Q8_0
```

### Package

To use this golang package, you can import it into your project with the following:

```go
import "github.com/sammcj/quantest"
```

Run a `go mod tidy` and use the package functions as required, e.g:

```go
func main() {
    estimation, err := quantest.EstimateVRAMForModel("llama3.1:8b", 24.0, 8192, "Q4_K_M", "fp16")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Estimated VRAM: %.2f GB\n", estimation.EstimatedVRAM)

    table, err := quantest.GenerateQuantTableForModel("llama3.1:8b", 24.0)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(quantest.PrintFormattedTable(table))
}

  // Print the estimation results
  fmt.Printf("\nEstimation Results:\n")
  fmt.Printf("Model: %s\n", estimation.ModelName)
  fmt.Printf("Estimated vRAM Required For A Context Size Of %d: %.2f GB\n", estimation.ContextSize, estimation.EstimatedVRAM)
  fmt.Printf("Fits Available vRAM: %v\n", estimation.FitsAvailable)
  fmt.Printf("Max Context Size: %d\n", estimation.MaxContextSize)
  fmt.Printf("Maximum Quantisation: %s\n", estimation.MaximumQuant)
```

#### Package Functions

See [docs/pkg.md](docs/pkg.md) for detailed information.
