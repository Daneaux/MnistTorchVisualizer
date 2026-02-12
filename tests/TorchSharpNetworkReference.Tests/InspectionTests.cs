using Xunit;
using TorchSharpNetworkReference.Inspection;
using TorchSharpNetworkReference.Serialization;

namespace TorchSharpNetworkReference.Tests;

public class InspectionTests
{
    [Fact]
    public void CaptureIterations_Returns2Iterations_WithAllValues()
    {
        var data = InspectionRunner.CaptureIterations(
            iterationCount: 2, batchSize: 4, seed: 42);

        Assert.Equal(2, data.Iterations.Count);

        foreach (var iteration in data.Iterations)
        {
            Assert.NotEmpty(iteration.Input);
            Assert.NotEmpty(iteration.Targets);
            Assert.Equal(4, iteration.Targets.Length);
            Assert.NotEmpty(iteration.Logits);
            Assert.False(float.IsNaN(iteration.Loss));

            Assert.Equal(3, iteration.LayersBefore.Count);
            Assert.Equal(3, iteration.LayersAfterBackward.Count);
            Assert.Equal(3, iteration.LayersAfterUpdate.Count);

            foreach (var layerName in new[] { "fc1", "fc2", "fc3" })
            {
                var before = iteration.LayersBefore[layerName];
                Assert.NotEmpty(before.Weights);
                Assert.NotEmpty(before.Biases);

                var afterBackward = iteration.LayersAfterBackward[layerName];
                Assert.NotNull(afterBackward.WeightGradients);
                Assert.NotNull(afterBackward.BiasGradients);
                Assert.NotEmpty(afterBackward.WeightGradients!);
                Assert.NotEmpty(afterBackward.BiasGradients!);

                var afterUpdate = iteration.LayersAfterUpdate[layerName];
                Assert.NotEmpty(afterUpdate.Weights);
            }

            // Hidden layers have both pre and post activation
            var fc1 = iteration.LayersAfterBackward["fc1"];
            Assert.NotNull(fc1.PreActivation);
            Assert.NotNull(fc1.PostActivation);

            var fc2 = iteration.LayersAfterBackward["fc2"];
            Assert.NotNull(fc2.PreActivation);
            Assert.NotNull(fc2.PostActivation);

            // Output layer has pre-activation (logits) but no post-activation
            var fc3 = iteration.LayersAfterBackward["fc3"];
            Assert.NotNull(fc3.PreActivation);
            Assert.Null(fc3.PostActivation);
        }
    }

    [Fact]
    public void CaptureIterations_WeightsChangeAfterUpdate()
    {
        var data = InspectionRunner.CaptureIterations(
            iterationCount: 1, batchSize: 4, seed: 42);

        var iteration = data.Iterations[0];

        foreach (var layerName in new[] { "fc1", "fc2", "fc3" })
        {
            var before = iteration.LayersBefore[layerName].Weights;
            var after = iteration.LayersAfterUpdate[layerName].Weights;

            bool anyDifferent = false;
            for (int i = 0; i < before.Length; i++)
            {
                if (Math.Abs(before[i] - after[i]) > 1e-10f)
                {
                    anyDifferent = true;
                    break;
                }
            }
            Assert.True(anyDifferent,
                $"Weights for {layerName} did not change after SGD step");
        }
    }

    [Fact]
    public void CaptureIterations_SerializesToFile()
    {
        var data = InspectionRunner.CaptureIterations(
            iterationCount: 2, batchSize: 4, seed: 42);

        var filePath = Path.Combine(Path.GetTempPath(), "mnist_inspection.json");
        try
        {
            IterationDataSerializer.SerializeToFile(data, filePath);
            Assert.True(File.Exists(filePath));
            Assert.True(new FileInfo(filePath).Length > 0);

            var deserialized = IterationDataSerializer.DeserializeFromFile(filePath);
            Assert.NotNull(deserialized);
            Assert.Equal(2, deserialized!.Iterations.Count);
            Assert.Equal(data.Iterations[0].Loss, deserialized.Iterations[0].Loss);
        }
        finally
        {
            if (File.Exists(filePath)) File.Delete(filePath);
        }
    }

    [Fact]
    public void CaptureIterations_ManualSgdUpdate_IsCorrect()
    {
        // Verify that weights_after = weights_before - lr * gradients
        // This confirms the manual SGD update is working correctly
        // and that captured data is internally consistent
        double learningRate = 0.01;
        var data = InspectionRunner.CaptureIterations(
            iterationCount: 1, batchSize: 4, seed: 42, learningRate: learningRate);

        var iteration = data.Iterations[0];

        foreach (var layerName in new[] { "fc1", "fc2", "fc3" })
        {
            var before = iteration.LayersBefore[layerName].Weights;
            var gradients = iteration.LayersAfterBackward[layerName].WeightGradients!;
            var after = iteration.LayersAfterUpdate[layerName].Weights;

            for (int i = 0; i < before.Length; i++)
            {
                float expected = before[i] - (float)(learningRate * gradients[i]);
                Assert.True(Math.Abs(expected - after[i]) < 1e-5f,
                    $"Layer {layerName} weight[{i}]: expected {expected}, got {after[i]}");
            }
        }
    }

    [Fact]
    public void CaptureIterations_RealTraining_Serializes3Iterations()
    {
        // Real MNIST training with same hyperparameters as training tests
        // Captures 3 iterations with all layer data for use as reference
        var data = InspectionRunner.CaptureIterations(
            iterationCount: 3, batchSize: 64, seed: 42, learningRate: 0.05);

        // Verify 3 iterations captured
        Assert.Equal(3, data.Iterations.Count);
        Assert.Equal(64, data.BatchSize);
        Assert.Equal(0.05, data.LearningRate);

        for (int i = 0; i < 3; i++)
        {
            var iteration = data.Iterations[i];
            Assert.Equal(i, iteration.IterationIndex);
            Assert.Equal(64, iteration.Targets.Length);
            Assert.NotEmpty(iteration.Input);
            Assert.NotEmpty(iteration.Logits);
            Assert.False(float.IsNaN(iteration.Loss));

            foreach (var layerName in new[] { "fc1", "fc2", "fc3" })
            {
                // Before forward: weights and biases captured
                var before = iteration.LayersBefore[layerName];
                Assert.NotEmpty(before.Weights);
                Assert.NotEmpty(before.Biases);

                // After backward: activations and gradients captured
                var afterBackward = iteration.LayersAfterBackward[layerName];
                Assert.NotNull(afterBackward.PreActivation);
                Assert.NotNull(afterBackward.WeightGradients);
                Assert.NotEmpty(afterBackward.WeightGradients!);
                Assert.NotNull(afterBackward.BiasGradients);
                Assert.NotEmpty(afterBackward.BiasGradients!);

                // After update: updated weights captured
                var afterUpdate = iteration.LayersAfterUpdate[layerName];
                Assert.NotEmpty(afterUpdate.Weights);
            }
        }

        // Verify loss is a reasonable value (not NaN/Inf)
        foreach (var iter in data.Iterations)
        {
            Assert.False(float.IsNaN(iter.Loss), "Loss should not be NaN");
            Assert.False(float.IsInfinity(iter.Loss), "Loss should not be Infinity");
            Assert.True(iter.Loss > 0, "Loss should be positive for cross-entropy");
        }

        // Serialize to project root as persistent reference file
        var solutionDir = Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
        var filePath = Path.Combine(solutionDir, "mnist_reference_3iterations.json");

        IterationDataSerializer.SerializeToFile(data, filePath);
        Assert.True(File.Exists(filePath));
        Assert.True(new FileInfo(filePath).Length > 0);

        // Round-trip verification
        var deserialized = IterationDataSerializer.DeserializeFromFile(filePath);
        Assert.NotNull(deserialized);
        Assert.Equal(3, deserialized!.Iterations.Count);
        Assert.Equal(data.Iterations[0].Loss, deserialized.Iterations[0].Loss);
        Assert.Equal(data.Iterations[1].Loss, deserialized.Iterations[1].Loss);
        Assert.Equal(data.Iterations[2].Loss, deserialized.Iterations[2].Loss);

        // Verify layer data survived serialization
        for (int i = 0; i < 3; i++)
        {
            foreach (var layerName in new[] { "fc1", "fc2", "fc3" })
            {
                Assert.Equal(
                    data.Iterations[i].LayersBefore[layerName].Weights,
                    deserialized.Iterations[i].LayersBefore[layerName].Weights);
                Assert.Equal(
                    data.Iterations[i].LayersAfterBackward[layerName].WeightGradients,
                    deserialized.Iterations[i].LayersAfterBackward[layerName].WeightGradients);
            }
        }
    }
}
