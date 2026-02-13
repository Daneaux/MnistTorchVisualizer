using Xunit;
using Xunit.Abstractions;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharpNetworkReference.Models;
using TorchSharpNetworkReference.Training;
using static TorchSharp.torch;

namespace TorchSharpNetworkReference.Tests;

/// <summary>
/// Comprehensive tests for various network topologies.
/// These tests verify that different architectures train correctly
/// and that invalid configurations are properly rejected.
/// </summary>
public class TopologyTests
{
    #region Convergence Tests - Single Layer

    [Theory]
    [Trait("Category", "SlowTests")]
    [InlineData(new[] { 16 }, 10, 64, 0.05, 0.75)]      // Minimal single layer
    [InlineData(new[] { 64 }, 10, 64, 0.05, 0.85)]      // Small single layer
    [InlineData(new[] { 128 }, 10, 64, 0.05, 0.90)]     // Medium single layer
    [InlineData(new[] { 256 }, 10, 64, 0.05, 0.92)]     // Large single layer
    [InlineData(new[] { 512 }, 10, 64, 0.05, 0.93)]     // Very large single layer
    [InlineData(new[] { 1024 }, 10, 64, 0.05, 0.94)]    // Maximum single layer
    public void SingleLayerTopology_Converges(
        int[] hiddenLayers, int epochs, int batchSize, double lr, double minAccuracy)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr
        };

        var (model, result) = MnistTrainer.Train(config);

        Assert.True(result.FinalTestAccuracy >= minAccuracy,
            $"Architecture [{string.Join(", ", hiddenLayers)}] expected >= {minAccuracy:P0} accuracy, " +
            $"got {result.FinalTestAccuracy:P2} after {epochs} epochs");
    }

    #endregion

    #region Convergence Tests - Two Layers

    [Theory]
    [Trait("Category", "SlowTests")]
    [InlineData(new[] { 16, 16 }, 10, 64, 0.05, 0.75)]      // Minimal 2-layer
    [InlineData(new[] { 64, 32 }, 10, 64, 0.05, 0.88)]      // Small 2-layer
    [InlineData(new[] { 128, 64 }, 10, 64, 0.05, 0.95)]     // Default (reference)
    [InlineData(new[] { 256, 128 }, 10, 64, 0.05, 0.95)]    // Large 2-layer
    [InlineData(new[] { 512, 256 }, 10, 64, 0.05, 0.96)]    // Very large 2-layer
    [InlineData(new[] { 1024, 512 }, 10, 64, 0.05, 0.96)]   // Maximum 2-layer
    public void TwoLayerTopology_Converges(
        int[] hiddenLayers, int epochs, int batchSize, double lr, double minAccuracy)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr
        };

        var (model, result) = MnistTrainer.Train(config);

        Assert.True(result.FinalTestAccuracy >= minAccuracy,
            $"Architecture [{string.Join(", ", hiddenLayers)}] expected >= {minAccuracy:P0} accuracy, " +
            $"got {result.FinalTestAccuracy:P2} after {epochs} epochs");
    }

    #endregion

    #region Convergence Tests - Three Layers

    [Theory]
    [Trait("Category", "SlowTests")]
    [InlineData(new[] { 16, 16, 16 }, 15, 64, 0.05, 0.75)]     // Minimal 3-layer
    [InlineData(new[] { 64, 32, 16 }, 15, 64, 0.05, 0.88)]     // Small 3-layer
    [InlineData(new[] { 128, 64, 32 }, 15, 64, 0.05, 0.94)]    // Medium 3-layer
    [InlineData(new[] { 256, 128, 64 }, 15, 64, 0.05, 0.95)]   // Large 3-layer
    [InlineData(new[] { 512, 256, 128 }, 15, 64, 0.05, 0.96)]  // Very large 3-layer
    public void ThreeLayerTopology_Converges(
        int[] hiddenLayers, int epochs, int batchSize, double lr, double minAccuracy)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr
        };

        var (model, result) = MnistTrainer.Train(config);

        Assert.True(result.FinalTestAccuracy >= minAccuracy,
            $"Architecture [{string.Join(", ", hiddenLayers)}] expected >= {minAccuracy:P0} accuracy, " +
            $"got {result.FinalTestAccuracy:P2} after {epochs} epochs");
    }

    #endregion

    #region Convergence Tests - Deep Networks (4-5 Layers)

    [Theory]
    [Trait("Category", "SlowTests")]
    [InlineData(new[] { 32, 32, 32, 32 }, 20, 64, 0.05, 0.85)]        // 4-layer uniform
    [InlineData(new[] { 128, 64, 32, 16 }, 20, 64, 0.05, 0.94)]       // 4-layer tapered
    [InlineData(new[] { 256, 128, 64, 32 }, 20, 64, 0.05, 0.95)]      // 4-layer large
    [InlineData(new[] { 32, 32, 32, 32, 32 }, 25, 64, 0.05, 0.85)]     // 5-layer uniform (max depth)
    [InlineData(new[] { 64, 48, 32, 24, 16 }, 25, 64, 0.05, 0.90)]    // 5-layer tapered
    public void DeepTopology_Converges(
        int[] hiddenLayers, int epochs, int batchSize, double lr, double minAccuracy)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr
        };

        var (model, result) = MnistTrainer.Train(config);

        Assert.True(result.FinalTestAccuracy >= minAccuracy,
            $"Architecture [{string.Join(", ", hiddenLayers)}] expected >= {minAccuracy:P0} accuracy, " +
            $"got {result.FinalTestAccuracy:P2} after {epochs} epochs");
    }

    #endregion

    #region Hyperparameter Variations - Batch Sizes

    [Theory]
    [Trait("Category", "SlowTests")]
    [InlineData(new[] { 128, 64 }, 15, 1, 0.05, 0.90)]      // Batch size 1 (pure SGD)
    [InlineData(new[] { 128, 64 }, 10, 16, 0.05, 0.93)]     // Batch size 16
    [InlineData(new[] { 128, 64 }, 10, 128, 0.05, 0.95)]    // Batch size 128
    [InlineData(new[] { 128, 64 }, 10, 256, 0.05, 0.95)]    // Batch size 256
    [InlineData(new[] { 128, 64 }, 10, 512, 0.05, 0.95)]    // Batch size 512 (max)
    public void Topology_WithVariousBatchSizes(
        int[] hiddenLayers, int epochs, int batchSize, double lr, double minAccuracy)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr
        };

        var (model, result) = MnistTrainer.Train(config);

        Assert.True(result.FinalTestAccuracy >= minAccuracy,
            $"Batch size {batchSize} expected >= {minAccuracy:P0} accuracy, " +
            $"got {result.FinalTestAccuracy:P2}");
    }

    #endregion

    #region Hyperparameter Variations - Learning Rates

    [Theory]
    [Trait("Category", "SlowTests")]
    [InlineData(new[] { 128, 64 }, 15, 64, 0.001, 0.86)]   // Very small LR
    [InlineData(new[] { 128, 64 }, 12, 64, 0.01, 0.94)]    // Small LR
    [InlineData(new[] { 128, 64 }, 10, 64, 0.05, 0.95)]    // Default LR
    [InlineData(new[] { 128, 64 }, 10, 64, 0.1, 0.95)]     // Large LR
    [InlineData(new[] { 128, 64 }, 8, 64, 0.5, 0.92)]      // Very large LR
    public void Topology_WithVariousLearningRates(
        int[] hiddenLayers, int epochs, int batchSize, double lr, double minAccuracy)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr
        };

        var (model, result) = MnistTrainer.Train(config);

        Assert.True(result.FinalTestAccuracy >= minAccuracy,
            $"Learning rate {lr} expected >= {minAccuracy:P0} accuracy, " +
            $"got {result.FinalTestAccuracy:P2}");
    }

    #endregion

    #region Validation Tests - Invalid Topologies

    [Theory]
    [Trait("Category", "FastTests")]
    [InlineData(new int[0], "at least 1")]                                  // Zero layers
    [InlineData(new[] { 16, 16, 16, 16, 16, 16 }, "more than 5")]         // 6 layers (too many)
    [InlineData(new[] { 1 }, "at least 2")]                                // Below minimum neurons (1 is too small)
    [InlineData(new[] { 1025 }, "more than 1024")]                         // Above maximum neurons
    [InlineData(new[] { -1 }, "at least 2")]                               // Negative neurons
    [InlineData(new[] { 0 }, "at least 2")]                                // Zero neurons
    [InlineData(new[] { 16, 1, 16 }, "at least 2")]                        // Mixed valid/invalid
    [InlineData(new[] { 1, 1025, 0 }, "at least 2")]                       // All invalid
    public void InvalidTopology_FailsValidation(int[] hiddenLayers, string expectedError)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers
        };

        var validation = TrainingConfiguration.Validate(config);

        Assert.False(validation.IsValid);
        Assert.Contains(expectedError, validation.ErrorMessage);
    }

    #endregion

    #region Edge Cases

    [Fact]
    [Trait("Category", "SlowTests")]
    public void AllLayersAtMinimum_TrainsSuccessfully()
    {
        // All layers at lower bound [2, 2, 2, 2, 2]
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = new[] { 2, 2, 2, 2, 2 },
            Epochs = 20,
            BatchSize = 64,
            LearningRate = 0.05
        };

        var validation = TrainingConfiguration.Validate(config);
        Assert.True(validation.IsValid, $"Should be valid but got: {validation.ErrorMessage}");

        var (model, result) = MnistTrainer.Train(config);

        // Very small network - just verify it trains without error
        // Accuracy will likely be poor but should be above random chance (10%)
        Assert.True(result.FinalTestAccuracy >= 0.15,
            $"All-minimum [2,2,2,2,2] architecture should achieve at least 15% accuracy (above random), got {result.FinalTestAccuracy:P2}");
    }

    [Fact]
    [Trait("Category", "SlowTests")]
    public void AllLayersAtMaximum_TrainsSuccessfully()
    {
        // All layers at upper bound [1024, 1024, 1024, 1024, 1024]
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = new[] { 1024, 1024, 1024, 1024, 1024 },
            Epochs = 10,
            BatchSize = 64,
            LearningRate = 0.05
        };

        var validation = TrainingConfiguration.Validate(config);
        Assert.True(validation.IsValid, $"Should be valid but got: {validation.ErrorMessage}");

        var (model, result) = MnistTrainer.Train(config);

        Assert.True(result.FinalTestAccuracy >= 0.95,
            $"All-maximum architecture should achieve at least 95% accuracy, got {result.FinalTestAccuracy:P2}");
    }

    [Fact]
    [Trait("Category", "SlowTests")]
    public void IncreasingLayerSizes_TrainsSuccessfully()
    {
        // Expanding architecture [32, 64, 128] - unusual but valid
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = new[] { 32, 64, 128 },
            Epochs = 15,
            BatchSize = 64,
            LearningRate = 0.05
        };

        var (model, result) = MnistTrainer.Train(config);

        Assert.True(result.FinalTestAccuracy >= 0.92,
            $"Expanding architecture should achieve at least 92% accuracy, got {result.FinalTestAccuracy:P2}");
    }

    [Fact]
    [Trait("Category", "SlowTests")]
    public void DecreasingLayerSizes_TrainsSuccessfully()
    {
        // Tapered pyramid [256, 128, 64, 32, 16] - typical pattern
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = new[] { 256, 128, 64, 32, 16 },
            Epochs = 20,
            BatchSize = 64,
            LearningRate = 0.05
        };

        var (model, result) = MnistTrainer.Train(config);

        Assert.True(result.FinalTestAccuracy >= 0.94,
            $"Tapered pyramid architecture should achieve at least 94% accuracy, got {result.FinalTestAccuracy:P2}");
    }

    [Fact]
    [Trait("Category", "FastTests")]
    public void ZeroEpochs_FailsValidation()
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = new[] { 128, 64 },
            Epochs = 0
        };

        var validation = TrainingConfiguration.Validate(config);

        Assert.False(validation.IsValid);
        Assert.Contains("at least 1", validation.ErrorMessage);
    }

    [Fact]
    [Trait("Category", "FastTests")]
    public void ExcessiveEpochs_FailsValidation()
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = new[] { 128, 64 },
            Epochs = 501
        };

        var validation = TrainingConfiguration.Validate(config);

        Assert.False(validation.IsValid);
        Assert.Contains("exceed 500", validation.ErrorMessage);
    }

    #endregion

    #region Model Selection Tests

    [Fact]
    [Trait("Category", "FastTests")]
    public void DefaultArchitecture_UsesMnistModel()
    {
        // Default [128, 64] should create MnistModel
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = new[] { 128, 64 },
            Epochs = 1  // Just need to verify model type, not full convergence
        };

        Assert.True(config.IsDefaultArchitecture);

        // Quick training to verify model instantiation
        var (model, _) = MnistTrainer.Train(config);

        Assert.IsType<MnistModel>(model);
    }

    [Theory]
    [Trait("Category", "FastTests")]
    [InlineData(new[] { 64 })]                    // Single layer
    [InlineData(new[] { 256, 128 })]              // Different 2-layer
    [InlineData(new[] { 128, 32 })]               // Different 2-layer
    [InlineData(new[] { 16, 16, 16 })]            // 3-layer
    [InlineData(new[] { 512, 256, 128, 64 })]     // 4-layer
    public void NonDefaultArchitecture_UsesConfigurableModel(int[] hiddenLayers)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = 1  // Just need to verify model type
        };

        Assert.False(config.IsDefaultArchitecture);

        // Quick training to verify model instantiation
        var (model, _) = MnistTrainer.Train(config);

        Assert.IsType<ConfigurableModel>(model);
    }

    #endregion

    #region Architecture Description Tests

    [Theory]
    [Trait("Category", "FastTests")]
    [InlineData(new[] { 128, 64 }, "784 → 128 (ReLU) → 64 (ReLU) → 10")]
    [InlineData(new[] { 256 }, "784 → 256 (ReLU) → 10")]
    [InlineData(new[] { 32, 32, 32 }, "784 → 32 (ReLU) → 32 (ReLU) → 32 (ReLU) → 10")]
    public void ArchitectureDescription_IsCorrect(int[] hiddenLayers, string expected)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers
        };

        var description = config.GetArchitectureDescription();

        Assert.Equal(expected, description);
    }

    #endregion

    #region Architecture Parsing Tests

    [Theory]
    [Trait("Category", "FastTests")]
    [InlineData("128,64", new[] { 128, 64 })]
    [InlineData("256, 128, 64", new[] { 256, 128, 64 })]
    [InlineData("  32 , 16 , 8  ", new[] { 32, 16, 8 })]
    [InlineData("512", new[] { 512 })]
    [InlineData("16,16,16,16,16", new[] { 16, 16, 16, 16, 16 })]
    public void ParseArchitecture_ValidInput_ReturnsCorrectArray(string input, int[] expected)
    {
        var result = TrainingConfiguration.ParseArchitecture(input);

        Assert.NotNull(result);
        Assert.Equal(expected, result);
    }

    [Theory]
    [Trait("Category", "FastTests")]
    [InlineData("")]           // Empty string
    [InlineData(" ")]          // Whitespace only
    [InlineData("abc")]         // Non-numeric
    [InlineData("128,abc,64")]  // Mixed valid/invalid
    [InlineData(",,")]         // Empty parts
    public void ParseArchitecture_InvalidInput_ReturnsNull(string input)
    {
        var result = TrainingConfiguration.ParseArchitecture(input);

        Assert.Null(result);
    }

    #endregion

    #region Minimum Model Size Tests - Finding the 90% Threshold

    /// <summary>
    /// These tests empirically determine the smallest model that achieves 90%+ accuracy on MNIST.
    /// We test single-layer architectures with varying neuron counts to find the threshold.
    /// </summary>

    [Theory]
    [Trait("Category", "SlowTests")]
    [InlineData(new[] { 70 }, 15, 64, 0.05)]     // 70 neurons - likely to hit 90%
    [InlineData(new[] { 80 }, 15, 64, 0.05)]     // 80 neurons - likely to hit 90%
    [InlineData(new[] { 90 }, 15, 64, 0.05)]     // 90 neurons - likely to hit 90%
    [InlineData(new[] { 96 }, 15, 64, 0.05)]     // 96 neurons - likely to hit 90%
    [InlineData(new[] { 100 }, 15, 64, 0.05)]    // 100 neurons - benchmark
    public void SmallSingleLayer_Models_ConvergenceTest(
        int[] hiddenLayers, int epochs, int batchSize, double lr)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr
        };

        var (model, result) = MnistTrainer.Train(config);

        // Just report the actual accuracy - we'll see which ones hit 90%
        // These tests will help us identify the threshold
        _testOutputHelper.WriteLine(
            $"Architecture [{string.Join(", ", hiddenLayers)}] achieved " +
            $"{result.FinalTestAccuracy:P2} accuracy after {epochs} epochs");

        // For now, just verify it trains without error and achieves reasonable accuracy (>80%)
        Assert.True(result.FinalTestAccuracy >= 0.80,
            $"Architecture [{string.Join(", ", hiddenLayers)}] should achieve at least 80% accuracy, " +
            $"got {result.FinalTestAccuracy:P2}");
    }

    [Theory]
    [Trait("Category", "SlowTests")]
    [InlineData(new[] { 64 }, 20, 64, 0.05)]     // 64 neurons with more epochs
    [InlineData(new[] { 64 }, 25, 64, 0.05)]     // 64 neurons with even more epochs
    [InlineData(new[] { 70 }, 20, 64, 0.05)]     // 70 neurons with more epochs
    public void SmallSingleLayer_ExtendedTraining_ConvergenceTest(
        int[] hiddenLayers, int epochs, int batchSize, double lr)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr
        };

        var (model, result) = MnistTrainer.Train(config);

        _testOutputHelper.WriteLine(
            $"Architecture [{string.Join(", ", hiddenLayers)}] with {epochs} epochs achieved " +
            $"{result.FinalTestAccuracy:P2} accuracy");

        // Extended training should get closer to 90%
        Assert.True(result.FinalTestAccuracy >= 0.85,
            $"Architecture [{string.Join(", ", hiddenLayers)}] with extended training should " +
            $"achieve at least 85% accuracy, got {result.FinalTestAccuracy:P2}");
    }

    [Theory]
    [Trait("Category", "SlowTests")]
    [InlineData(new[] { 48, 24 }, 15, 64, 0.05)]    // Small 2-layer [48,24]
    [InlineData(new[] { 56, 28 }, 15, 64, 0.05)]    // Small 2-layer [56,28]
    [InlineData(new[] { 64, 32 }, 15, 64, 0.05)]    // Small 2-layer [64,32]
    public void SmallTwoLayer_Models_ConvergenceTest(
        int[] hiddenLayers, int epochs, int batchSize, double lr)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr
        };

        var (model, result) = MnistTrainer.Train(config);

        int totalParams = CalculateTotalParameters(hiddenLayers);
        _testOutputHelper.WriteLine(
            $"Architecture [{string.Join(", ", hiddenLayers)}] with {totalParams} parameters achieved " +
            $"{result.FinalTestAccuracy:P2} accuracy");

        // Verify reasonable convergence
        Assert.True(result.FinalTestAccuracy >= 0.82,
            $"Architecture [{string.Join(", ", hiddenLayers)}] should achieve at least 82% accuracy, " +
            $"got {result.FinalTestAccuracy:P2}");
    }

    [Fact]
    [Trait("Category", "SlowTests")]
    public void MinimumModelComparison_SingleVersusTwoLayer()
    {
        // Compare parameter efficiency: single [80] vs two-layer [56,28]
        // [80]: ~63,450 params, [56,28]: ~46,000 params

        var configs = new[]
        {
            new { Name = "Single [80]", Layers = new[] { 80 }, Params = 63450 },
            new { Name = "Two-layer [56,28]", Layers = new[] { 56, 28 }, Params = 46000 },
            new { Name = "Two-layer [48,24]", Layers = new[] { 48, 24 }, Params = 39200 }
        };

        foreach (var cfg in configs)
        {
            var config = new TrainingConfiguration
            {
                HiddenLayerSizes = cfg.Layers,
                Epochs = 20,
                BatchSize = 64,
                LearningRate = 0.05
            };

            var (model, result) = MnistTrainer.Train(config);

            _testOutputHelper.WriteLine(
                $"{cfg.Name}: {cfg.Params} parameters → {result.FinalTestAccuracy:P2} accuracy");
        }

        // This test documents the comparison but doesn't assert specific values
        // Run it to see which architecture is most parameter-efficient for 90%+
        Assert.True(true, "Comparison test completed - see output for results");
    }

    [Theory]
    [Trait("Category", "SlowTests")]
    [InlineData(new[] { 80 }, 15, 1, 0.05)]     // Single layer, batch size 1
    [InlineData(new[] { 80 }, 15, 16, 0.05)]    // Single layer, batch size 16
    [InlineData(new[] { 80 }, 15, 128, 0.05)]   // Single layer, batch size 128
    public void SmallModel_WithVariousBatchSizes(
        int[] hiddenLayers, int epochs, int batchSize, double lr)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr
        };

        var (model, result) = MnistTrainer.Train(config);

        _testOutputHelper.WriteLine(
            $"Architecture [{string.Join(", ", hiddenLayers)}] with batch size {batchSize} achieved " +
            $"{result.FinalTestAccuracy:P2} accuracy");

        // Small models should still achieve reasonable accuracy with different batch sizes
        Assert.True(result.FinalTestAccuracy >= 0.80,
            $"Should achieve at least 80% accuracy with batch size {batchSize}");
    }

    /// <summary>
    /// Helper method to calculate total parameter count for a given architecture.
    /// </summary>
    private static int CalculateTotalParameters(int[] hiddenLayers)
    {
        const int inputSize = 784;
        const int outputSize = 10;

        int totalParams = 0;
        int prevSize = inputSize;

        foreach (var layerSize in hiddenLayers)
        {
            // Weights + biases for each layer
            totalParams += (prevSize * layerSize) + layerSize;
            prevSize = layerSize;
        }

        // Output layer
        totalParams += (prevSize * outputSize) + outputSize;

        return totalParams;
    }

    #region Extremely Small Model Tests (2-8 neurons)

    [Theory]
    [Trait("Category", "SlowTests")]
    [InlineData(new[] { 2 }, 20, 64, 0.05)]      // 2 neurons - minimal possible
    [InlineData(new[] { 4 }, 20, 64, 0.05)]      // 4 neurons
    [InlineData(new[] { 8 }, 20, 64, 0.05)]      // 8 neurons
    public void ExtremeMinimalSingleLayer_TrainsAndConverges(
        int[] hiddenLayers, int epochs, int batchSize, double lr)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr
        };

        var validation = TrainingConfiguration.Validate(config);
        Assert.True(validation.IsValid, $"Config should be valid: {validation.ErrorMessage}");

        var (model, result) = MnistTrainer.Train(config);

        int totalParams = CalculateTotalParameters(hiddenLayers);
        _testOutputHelper.WriteLine(
            $"Extreme minimal [{string.Join(", ", hiddenLayers)}] with {totalParams} parameters achieved " +
            $"{result.FinalTestAccuracy:P2} accuracy after {epochs} epochs");

        // For 2 neurons, expect at least 20% (twice random chance)
        // For 4 neurons, expect at least 30%
        // For 8 neurons, expect at least 40%
        double expectedMinAccuracy = hiddenLayers[0] switch
        {
            2 => 0.20,
            4 => 0.30,
            8 => 0.40,
            _ => 0.15
        };

        Assert.True(result.FinalTestAccuracy >= expectedMinAccuracy,
            $"Architecture [{string.Join(", ", hiddenLayers)}] with {totalParams} parameters " +
            $"should achieve at least {expectedMinAccuracy:P0} accuracy, got {result.FinalTestAccuracy:P2}");
    }

    [Theory]
    [Trait("Category", "SlowTests")]
    [InlineData(new[] { 2, 2 }, 20, 64, 0.05)]     // [2,2] - minimal 2-layer
    [InlineData(new[] { 4, 2 }, 20, 64, 0.05)]     // [4,2] - small 2-layer
    [InlineData(new[] { 4, 4 }, 20, 64, 0.05)]     // [4,4] - small 2-layer
    public void ExtremeMinimalTwoLayer_TrainsAndConverges(
        int[] hiddenLayers, int epochs, int batchSize, double lr)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr
        };

        var (model, result) = MnistTrainer.Train(config);

        int totalParams = CalculateTotalParameters(hiddenLayers);
        _testOutputHelper.WriteLine(
            $"Extreme minimal 2-layer [{string.Join(", ", hiddenLayers)}] with {totalParams} parameters achieved " +
            $"{result.FinalTestAccuracy:P2} accuracy");

        // Very small networks - just verify they train and beat random chance
        Assert.True(result.FinalTestAccuracy >= 0.20,
            $"Architecture [{string.Join(", ", hiddenLayers)}] should achieve at least 20% accuracy, " +
            $"got {result.FinalTestAccuracy:P2}");
    }

    [Fact]
    [Trait("Category", "SlowTests")]
    public void SingleNeuronPerLayer_FailsValidation()
    {
        // Single neuron layers should fail validation (minimum is 2)
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = new[] { 1, 1, 1 }
        };

        var validation = TrainingConfiguration.Validate(config);

        Assert.False(validation.IsValid);
        Assert.Contains("at least 2", validation.ErrorMessage);
    }

    [Theory]
    [Trait("Category", "SlowTests")]
    [InlineData(new[] { 2 }, 50, 64, 0.05)]      // 2 neurons with extended training
    [InlineData(new[] { 4 }, 50, 64, 0.05)]      // 4 neurons with extended training
    public void ExtremeMinimalExtendedTraining_ConvergenceTest(
        int[] hiddenLayers, int epochs, int batchSize, double lr)
    {
        var config = new TrainingConfiguration
        {
            HiddenLayerSizes = hiddenLayers,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = lr
        };

        var (model, result) = MnistTrainer.Train(config);

        int totalParams = CalculateTotalParameters(hiddenLayers);
        _testOutputHelper.WriteLine(
            $"Extended training: [{string.Join(", ", hiddenLayers)}] with {totalParams} parameters, " +
            $"{epochs} epochs achieved {result.FinalTestAccuracy:P2} accuracy");

        // Extended training should help minimal networks
        double expectedMinAccuracy = hiddenLayers[0] switch
        {
            2 => 0.25,
            4 => 0.40,
            _ => 0.20
        };

        Assert.True(result.FinalTestAccuracy >= expectedMinAccuracy,
            $"With extended training, [{string.Join(", ", hiddenLayers)}] should achieve at least {expectedMinAccuracy:P0}, " +
            $"got {result.FinalTestAccuracy:P2}");
    }

    #endregion

    private readonly ITestOutputHelper _testOutputHelper;

    public TopologyTests(ITestOutputHelper testOutputHelper)
    {
        _testOutputHelper = testOutputHelper;
    }

    #endregion
}
