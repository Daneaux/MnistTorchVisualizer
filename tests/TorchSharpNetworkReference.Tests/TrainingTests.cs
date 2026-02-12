using Xunit;
using TorchSharpNetworkReference.Models;
using TorchSharpNetworkReference.Training;

namespace TorchSharpNetworkReference.Tests;

public class TrainingTests
{
    [Fact]
    public void MnistModel_TrainsAndConverges_Above95Percent()
    {
        var model = new MnistModel();
        var result = MnistTrainer.Train(model, epochs: 10, batchSize: 64, learningRate: 0.05);

        Assert.True(result.FinalTestAccuracy >= 0.95,
            $"Expected >= 95% accuracy, got {result.FinalTestAccuracy:P2}");
    }

    [Fact]
    public void MnistModel_TrainsAndConverges_Above90Percent_In2Epochs()
    {
        var model = new MnistModel();
        var result = MnistTrainer.Train(model, epochs: 3, batchSize: 64, learningRate: 0.05);

        Assert.True(result.FinalTestAccuracy >= 0.90,
            $"Expected >= 90% accuracy in 3 epochs, got {result.FinalTestAccuracy:P2}");
    }
}
