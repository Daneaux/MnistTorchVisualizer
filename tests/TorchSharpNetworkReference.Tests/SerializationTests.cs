using Xunit;
using TorchSharpNetworkReference.Inspection;
using TorchSharpNetworkReference.Serialization;

namespace TorchSharpNetworkReference.Tests;

public class SerializationTests
{
    [Fact]
    public void RoundTrip_PreservesAllData()
    {
        var original = CreateSampleData();

        var json = IterationDataSerializer.Serialize(original);
        var deserialized = IterationDataSerializer.Deserialize(json);

        Assert.NotNull(deserialized);
        Assert.Equal(original.ModelArchitecture, deserialized!.ModelArchitecture);
        Assert.Equal(original.LearningRate, deserialized.LearningRate);
        Assert.Equal(original.BatchSize, deserialized.BatchSize);
        Assert.Equal(original.RandomSeed, deserialized.RandomSeed);
        Assert.Equal(original.Iterations.Count, deserialized.Iterations.Count);

        var origIter = original.Iterations[0];
        var deserIter = deserialized.Iterations[0];

        Assert.Equal(origIter.IterationIndex, deserIter.IterationIndex);
        Assert.Equal(origIter.Loss, deserIter.Loss);
        Assert.Equal(origIter.Input, deserIter.Input);
        Assert.Equal(origIter.InputShape, deserIter.InputShape);
        Assert.Equal(origIter.Targets, deserIter.Targets);
        Assert.Equal(origIter.Logits, deserIter.Logits);
        Assert.Equal(origIter.LogitsShape, deserIter.LogitsShape);

        foreach (var layerName in origIter.LayersBefore.Keys)
        {
            var origLayer = origIter.LayersBefore[layerName];
            var deserLayer = deserIter.LayersBefore[layerName];
            Assert.Equal(origLayer.LayerName, deserLayer.LayerName);
            Assert.Equal(origLayer.Weights, deserLayer.Weights);
            Assert.Equal(origLayer.WeightShape, deserLayer.WeightShape);
            Assert.Equal(origLayer.Biases, deserLayer.Biases);
        }

        // Verify gradients round-trip
        var origAfter = origIter.LayersAfterBackward["fc1"];
        var deserAfter = deserIter.LayersAfterBackward["fc1"];
        Assert.Equal(origAfter.WeightGradients, deserAfter.WeightGradients);
        Assert.Equal(origAfter.BiasGradients, deserAfter.BiasGradients);
    }

    [Fact]
    public void RoundTrip_FileBasedSerialize()
    {
        var original = CreateSampleData();
        var filePath = Path.Combine(Path.GetTempPath(), "test_roundtrip.json");
        try
        {
            IterationDataSerializer.SerializeToFile(original, filePath);
            Assert.True(File.Exists(filePath));

            var deserialized = IterationDataSerializer.DeserializeFromFile(filePath);
            Assert.NotNull(deserialized);
            Assert.Equal(original.Iterations.Count, deserialized!.Iterations.Count);
            Assert.Equal(original.Iterations[0].Loss, deserialized.Iterations[0].Loss);
        }
        finally
        {
            if (File.Exists(filePath)) File.Delete(filePath);
        }
    }

    [Fact]
    public void RoundTrip_NullGradients_HandledCorrectly()
    {
        var data = CreateSampleData();
        data.Iterations[0].LayersBefore["fc1"].WeightGradients = null;
        data.Iterations[0].LayersBefore["fc1"].BiasGradients = null;

        var json = IterationDataSerializer.Serialize(data);
        var deserialized = IterationDataSerializer.Deserialize(json);

        Assert.Null(deserialized!.Iterations[0].LayersBefore["fc1"].WeightGradients);
        Assert.Null(deserialized.Iterations[0].LayersBefore["fc1"].BiasGradients);
    }

    private static ForwardPassData CreateSampleData()
    {
        return new ForwardPassData
        {
            LearningRate = 0.01,
            BatchSize = 4,
            RandomSeed = 42,
            Iterations = new List<IterationData>
            {
                new()
                {
                    IterationIndex = 0,
                    Input = new float[] { 0.1f, 0.2f, 0.3f },
                    InputShape = new long[] { 1, 3 },
                    Targets = new long[] { 5 },
                    Loss = 2.3026f,
                    Logits = new float[] { 0.1f, -0.2f, 0.05f, 0f, 0f, 0f, 0f, 0f, 0f, 0f },
                    LogitsShape = new long[] { 1, 10 },
                    LayersBefore = new Dictionary<string, LayerSnapshot>
                    {
                        ["fc1"] = new()
                        {
                            LayerName = "fc1",
                            Weights = new float[] { 0.01f, 0.02f },
                            WeightShape = new long[] { 1, 2 },
                            Biases = new float[] { 0.0f }
                        }
                    },
                    LayersAfterBackward = new Dictionary<string, LayerSnapshot>
                    {
                        ["fc1"] = new()
                        {
                            LayerName = "fc1",
                            Weights = new float[] { 0.01f, 0.02f },
                            WeightShape = new long[] { 1, 2 },
                            Biases = new float[] { 0.0f },
                            WeightGradients = new float[] { -0.001f, 0.002f },
                            BiasGradients = new float[] { 0.001f }
                        }
                    },
                    LayersAfterUpdate = new Dictionary<string, LayerSnapshot>
                    {
                        ["fc1"] = new()
                        {
                            LayerName = "fc1",
                            Weights = new float[] { 0.011f, 0.018f },
                            WeightShape = new long[] { 1, 2 },
                            Biases = new float[] { -0.001f }
                        }
                    }
                }
            }
        };
    }
}
