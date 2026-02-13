# TorchSharpNetworkReference

A C# reference implementation of a neural network for MNIST digit classification using TorchSharp. This project provides a fully inspectable training pipeline with weight/gradient capture, JSON serialization, and an interactive WPF visualizer with configurable training parameters.

## Features

- **Neural Network Models**: Fixed architecture (784→128→64→10) and configurable MLP
- **Centralized Configuration**: All training parameters in one immutable `TrainingConfiguration` class
- **Training Pipeline**: Manual SGD implementation with cross-entropy loss and progress reporting
- **Full Inspection**: Captures weights, biases, activations, and gradients at each training step
- **JSON Serialization**: Persist training iterations for analysis and regression testing
- **WPF Visualizer**: Interactive GUI with configurable architecture and hyperparameters
- **Comprehensive Tests**: xUnit tests verifying convergence (>95% accuracy) and data integrity

## Project Structure

```
├── src/
│   ├── TorchSharpNetworkReference/           # Core library
│   │   ├── Models/                           # Neural network architectures
│   │   ├── Training/                         # Training logic and configuration
│   │   ├── Inspection/                       # Data capture during training
│   │   └── Serialization/                    # JSON export/import
│   └── TorchSharpNetworkReference.Visualizer/# WPF GUI application
├── tests/TorchSharpNetworkReference.Tests/   # xUnit test suite
└── mnist_reference_3iterations.json          # Reference training data
```

## Quick Start

### Prerequisites

- .NET 9 SDK
- Windows (for WPF visualizer)

### Build

```bash
dotnet build
```

### Run Tests

```bash
dotnet test
```

### Run Visualizer

```bash
dotnet run --project src/TorchSharpNetworkReference.Visualizer
```

## Usage

### Training with Configuration

The recommended way to train uses the `TrainingConfiguration` class which centralizes all parameters:

```csharp
using TorchSharpNetworkReference.Training;

var config = new TrainingConfiguration
{
    HiddenLayerSizes = new[] { 128, 64 },  // Architecture
    Epochs = 10,
    BatchSize = 64,
    LearningRate = 0.05
};

// Validate before training
var validation = TrainingConfiguration.Validate(config);
if (!validation.IsValid)
{
    Console.WriteLine($"Errors: {validation.ErrorMessage}");
    return;
}

// Train with progress reporting
var progress = new Progress<string>(msg => Console.WriteLine(msg));
var (model, result) = MnistTrainer.Train(config, progress);

Console.WriteLine($"Accuracy: {result.FinalTestAccuracy:P2}");
Console.WriteLine($"Architecture: {config.GetArchitectureDescription()}");
```

### Parsing Architecture Strings

```csharp
// Supports formats like "128,64", "256, 128, 64", "512,256,128,64"
var architecture = TrainingConfiguration.ParseArchitecture("256, 128, 64");
// Returns: int[] { 256, 128, 64 }

var config = new TrainingConfiguration
{
    HiddenLayerSizes = architecture ?? new[] { 128, 64 }  // Fallback to default
};
```

### Legacy Training API

For backward compatibility, you can still train models directly:

```csharp
using TorchSharpNetworkReference.Models;
using TorchSharpNetworkReference.Training;

var model = new MnistModel();
var result = MnistTrainer.Train(model, epochs: 10, learningRate: 0.05);
Console.WriteLine($"Accuracy: {result.FinalTestAccuracy:P2}");
```

### Capturing Training Data

```csharp
using TorchSharpNetworkReference.Inspection;
using TorchSharpNetworkReference.Serialization;

var data = InspectionRunner.CaptureIterations(
    iterationCount: 3, 
    batchSize: 64, 
    seed: 42
);

IterationDataSerializer.SerializeToFile(data, "training.json");
```

### Loading Saved Data

```csharp
var data = IterationDataSerializer.DeserializeFromFile("training.json");

foreach (var iteration in data.Iterations)
{
    Console.WriteLine($"Iteration {iteration.IterationIndex}: Loss = {iteration.Loss}");
    
    // Access layer weights before/after update
    var fc1Before = iteration.LayersBefore["fc1"].Weights;
    var fc1Gradients = iteration.LayersAfterBackward["fc1"].WeightGradients;
    var fc1After = iteration.LayersAfterUpdate["fc1"].Weights;
}
```

## Architecture

### MnistModel

Fixed 3-layer MLP:
- Input: 784 (28×28 pixels flattened)
- Hidden 1: 128 units with ReLU
- Hidden 2: 64 units with ReLU
- Output: 10 units (digit classes 0-9)

### ConfigurableModel

Dynamic architecture accepting any array of hidden layer sizes:

```csharp
var model = new ConfigurableModel(new[] { 256, 128, 64 });
// Creates: 784 → 256 (ReLU) → 128 (ReLU) → 64 (ReLU) → 10
```

### TrainingConfiguration

Centralized, immutable configuration for training:

```csharp
public record TrainingConfiguration
{
    public int[] HiddenLayerSizes { get; init; } = { 128, 64 };
    public int Epochs { get; init; } = 10;
    public int BatchSize { get; init; } = 64;
    public double LearningRate { get; init; } = 0.05;
    public string DataPath { get; init; } = "./data";
}
```

**Validation Ranges:**

| Parameter | Minimum | Maximum | Notes |
|-----------|---------|---------|-------|
| Hidden Layers | 1 | 5 layers | Architecture depth |
| Neurons per Layer | 16 | 1024 | Layer width |
| Epochs | 1 | 500 | Training iterations |
| Batch Size | 1 | 512 | Samples per batch |
| Learning Rate | 0.0001 | 1.0 | SGD step size |

### Smart Model Selection

When using `TrainingConfiguration`:
- **Default architecture [128, 64]** → Uses `MnistModel` (full inspection support)
- **Custom architecture** → Uses `ConfigurableModel` (flexible but basic)

### Inspection Data Structure

Each training iteration captures:

| Phase | Data Captured |
|-------|--------------|
| Before Forward | Initial weights and biases |
| After Backward | Activations (pre/post ReLU), gradients |
| After Update | Updated weights and biases (post-SGD) |

## WPF Visualizer Features

The visualizer includes a configuration panel for interactive experimentation:

- **Architecture**: Enter hidden layer sizes (e.g., "128,64" or "256, 128, 64")
- **Epochs**: Set training duration (1-500)
- **Batch Size**: Adjust batch size (1-512)
- **Learning Rate**: Set SGD learning rate (0.0001-1.0)
- **Train Button**: Start training with current configuration
- **Reset Button**: Restore default values
- **Progress Display**: Real-time training status and epoch progress

Invalid inputs show error messages in red; successful training displays the architecture used.

## Testing

The test suite verifies:

- **Convergence**: Model achieves >95% accuracy after 10 epochs
- **Weight Updates**: SGD correctly updates weights (w' = w - lr * grad)
- **Configuration Validation**: All parameter ranges enforced correctly
- **Architecture Parsing**: Various input formats handled properly
- **Serialization**: Round-trip JSON preserves all data
- **Inspection**: All intermediate values are captured correctly

## Test Suite

The project includes comprehensive xUnit tests organized into several categories:

### Test Organization

Tests are tagged with categories for selective execution:

```bash
# Run only fast tests (validation, parsing, basic functionality)
dotnet test --filter "Category=FastTests"

# Run only slow tests (convergence, training)
dotnet test --filter "Category=SlowTests"

# Run all tests
dotnet test
```

### TrainingTests

Core training functionality tests:

- `MnistModel_TrainsAndConverges_Above95Percent` - Verifies default model reaches 95%+
- `MnistModel_TrainsAndConverges_Above90Percent_In2Epochs` - Quick convergence test

### TopologyTests

Extensive topology exploration and minimum model size determination:

**Convergence Tests (Single Layer):**
- Tests architectures from 16 to 1024 neurons
- Verifies accuracy ranges from 75% to 94%

**Convergence Tests (Multi-Layer):**
- Two-layer: [16,16] to [1024,512]
- Three-layer: [16,16,16] to [512,256,128]
- Deep networks: 4-5 layer architectures

**Hyperparameter Variations:**
- Batch sizes: 1, 16, 128, 256, 512
- Learning rates: 0.001, 0.01, 0.05, 0.1, 0.5

**Minimum Model Size Tests:**
Empirical tests to find the smallest model achieving 90%+ accuracy:

```csharp
// Single-layer candidates
[70], [80], [90], [96], [100] neurons

// Two-layer candidates (more parameter-efficient)
[48,24], [56,28], [64,32]

// Extended training for small models
[64] with 20-25 epochs
[70] with 20 epochs
```

**Validation Tests:**
- Zero layers, excessive layers (6+)
- Invalid neuron counts (<16, >1024)
- Negative/zero values
- Mixed valid/invalid configurations

**Edge Cases:**
- All layers at minimum [16,16,16,16,16]
- All layers at maximum [1024,1024,1024,1024,1024]
- Increasing sizes [32,64,128]
- Decreasing pyramid [256,128,64,32,16]

**Model Selection Tests:**
- Verifies default [128,64] uses MnistModel
- Verifies custom architectures use ConfigurableModel

### InspectionTests

Training data capture and inspection:

- `CaptureIterations_Returns2Iterations_WithAllValues` - Verifies complete data capture
- `CaptureIterations_WeightsChangeAfterUpdate` - Confirms SGD updates weights
- `CaptureIterations_SerializesToFile` - Tests JSON persistence
- `CaptureIterations_ManualSgdUpdate_IsCorrect` - Validates update formula
- `CaptureIterations_RealTraining_Serializes3Iterations` - Creates reference data

### SerializationTests

JSON serialization validation:

- `RoundTrip_PreservesAllData` - Complete data integrity
- `RoundTrip_FileBasedSerialize` - File I/O testing
- `RoundTrip_NullGradients_HandledCorrectly` - Edge case handling

### Running Specific Test Categories

```bash
# Run topology convergence tests only
dotnet test --filter "SingleLayerTopology_Converges"

# Run minimum model size determination
dotnet test --filter "SmallSingleLayer_Models_ConvergenceTest"

# Run parameter efficiency comparison
dotnet test --filter "MinimumModelComparison_SingleVersusTwoLayer"

# Run validation tests only
dotnet test --filter "InvalidTopology_FailsValidation"
```

## Dependencies

- [TorchSharp](https://github.com/dotnet/TorchSharp) 0.105.2 - PyTorch bindings for .NET
- TorchVision 0.105.2 - Vision datasets and transforms
- xUnit - Testing framework

## License

MIT
