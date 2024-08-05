package quantest // import "."

func EstimateVRAMForModel(modelName string, vram float64, contextSize int, quantLevel, kvQuant string) (*VRAMEstimation, error)
    EstimateVRAMForModel is the main function that other packages can call



package quantest // import "."

func GenerateQuantTableForModel(modelName string, vram float64) (QuantResultTable, error)
    GenerateQuantTableForModel generates a quant table for a model



package quantest // import "."

func GetRecommendations(estimation *VRAMEstimation) map[int]string
    GetRecommendations returns the quant recommendations for different context
    sizes



package quantest // import "."

func GetMaxContextSize(estimation *VRAMEstimation) int
    GetMaxContextSize returns the maximum context size for the given VRAM and
    quantization



package quantest // import "."

func GetMaximumQuant(estimation *VRAMEstimation) string
    GetMaximumQuant returns the maximum quantization level for the given context
    size



package quantest // import "."

func PrintEstimationResults(estimation *VRAMEstimation) string
    PrintEstimationResults formats and returns the estimation results as a
    string

