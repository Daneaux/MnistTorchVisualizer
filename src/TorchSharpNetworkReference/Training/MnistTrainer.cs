using TorchSharp;
using TorchSharpNetworkReference.Models;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision;

namespace TorchSharpNetworkReference.Training;

public static class MnistTrainer
{
    public record TrainingResult(int Epochs, double FinalTestAccuracy, double FinalTestLoss);

    public static TrainingResult Train(
        MnistModel model,
        int epochs = 5,
        int batchSize = 64,
        double learningRate = 0.01,
        string dataPath = "./data")
    {
        var lossFn = CrossEntropyLoss();

        using var trainData = datasets.MNIST(dataPath, true, download: true);
        using var testData = datasets.MNIST(dataPath, false, download: true);

        using var trainLoader = DataLoader(trainData, batchSize, shuffle: true);
        using var testLoader = DataLoader(testData, batchSize, shuffle: false);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            model.train();

            foreach (var batch in trainLoader)
            {
                using var disposeScope = NewDisposeScope();

                var data = batch["data"];
                var target = batch["label"];

                var output = model.call(data);
                var loss = lossFn.call(output, target);

                model.zero_grad();
                loss.backward();
                model.ManualSgdStep(learningRate);
            }
        }

        var (accuracy, avgLoss) = Evaluate(model, testData, batchSize);
        return new TrainingResult(epochs, accuracy, avgLoss);
    }

    /// <summary>
    /// Creates a new model, trains it, and returns it ready for inference.
    /// </summary>
    public static (MnistModel model, TrainingResult result) TrainAndReturn(
        int epochs = 10,
        int batchSize = 64,
        double learningRate = 0.05,
        string dataPath = "./data")
    {
        var model = new MnistModel();
        var result = Train(model, epochs, batchSize, learningRate, dataPath);
        return (model, result);
    }

    public static (double accuracy, double avgLoss) Evaluate(
        MnistModel model,
        string dataPath = "./data",
        int batchSize = 128)
    {
        using var testData = datasets.MNIST(dataPath, false, download: true);
        return Evaluate(model, testData, batchSize);
    }

    private static (double accuracy, double avgLoss) Evaluate(
        MnistModel model,
        Dataset testData,
        int batchSize)
    {
        var lossFn = CrossEntropyLoss();
        using var testLoader = DataLoader(testData, batchSize, shuffle: false);

        model.eval();
        long correct = 0;
        long total = 0;
        double totalLoss = 0;
        int batchCount = 0;

        using (no_grad())
        {
            foreach (var batch in testLoader)
            {
                using var disposeScope = NewDisposeScope();

                var data = batch["data"];
                var target = batch["label"];

                var output = model.call(data);
                var loss = lossFn.call(output, target);

                totalLoss += loss.item<float>();
                batchCount++;

                var predicted = output.argmax(1);
                correct += predicted.eq(target).sum().item<long>();
                total += target.shape[0];
            }
        }

        return ((double)correct / total, totalLoss / batchCount);
    }
}
