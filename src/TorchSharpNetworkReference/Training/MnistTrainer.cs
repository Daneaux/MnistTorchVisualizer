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

    /// <summary>
    /// Trains a model using the provided configuration.
    /// Creates MnistModel for default architecture, ConfigurableModel otherwise.
    /// </summary>
    public static (Module<Tensor, Tensor> model, TrainingResult result) Train(
        TrainingConfiguration config,
        IProgress<string>? progress = null)
    {
        var validation = TrainingConfiguration.Validate(config);
        if (!validation.IsValid)
            throw new ArgumentException($"Invalid configuration: {validation.ErrorMessage}");

        progress?.Report($"Creating model: {config.GetArchitectureDescription()}");

        // Use MnistModel for default architecture, ConfigurableModel otherwise
        Module<Tensor, Tensor> model;
        Action<double> sgdStep;

        if (config.IsDefaultArchitecture)
        {
            var mnistModel = new MnistModel();
            model = mnistModel;
            sgdStep = lr => mnistModel.ManualSgdStep(lr);
        }
        else
        {
            var configurableModel = new ConfigurableModel(config.HiddenLayerSizes);
            model = configurableModel;
            sgdStep = lr => configurableModel.ManualSgdStep(lr);
        }

        progress?.Report($"Training: {config.Epochs} epochs, batch={config.BatchSize}, lr={config.LearningRate}");
        var result = TrainInternal(model, sgdStep, config, progress);

        return (model, result);
    }

    public static TrainingResult Train(
        MnistModel model,
        int epochs = 5,
        int batchSize = 64,
        double learningRate = 0.01,
        string dataPath = "./data")
    {
        return TrainInternal(model, lr => model.ManualSgdStep(lr), epochs, batchSize, learningRate, dataPath);
    }

    public static TrainingResult Train(
        ConfigurableModel model,
        int epochs = 5,
        int batchSize = 64,
        double learningRate = 0.01,
        string dataPath = "./data")
    {
        return TrainInternal(model, lr => model.ManualSgdStep(lr), epochs, batchSize, learningRate, dataPath);
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
        return EvaluateInternal(model, testData, batchSize);
    }

    private static TrainingResult TrainInternal(
        Module<Tensor, Tensor> model,
        Action<double> sgdStep,
        TrainingConfiguration config,
        IProgress<string>? progress)
    {
        var lossFn = CrossEntropyLoss();

        using var trainData = datasets.MNIST(config.DataPath, true, download: true);
        using var testData = datasets.MNIST(config.DataPath, false, download: true);

        using var trainLoader = DataLoader(trainData, config.BatchSize, shuffle: true);

        for (int epoch = 0; epoch < config.Epochs; epoch++)
        {
            progress?.Report($"Epoch {epoch + 1}/{config.Epochs}...");
            model.train();

            int batchCount = 0;
            foreach (var batch in trainLoader)
            {
                using var disposeScope = NewDisposeScope();

                var data = batch["data"];
                var target = batch["label"];

                var output = model.call(data);
                var loss = lossFn.call(output, target);

                model.zero_grad();
                loss.backward();
                sgdStep(config.LearningRate);
                batchCount++;
            }
            progress?.Report($"Epoch {epoch + 1}/{config.Epochs} complete ({batchCount} batches)");
        }

        progress?.Report("Evaluating model...");
        var (accuracy, avgLoss) = EvaluateInternal(model, testData, config.BatchSize);
        progress?.Report($"Training complete. Accuracy: {accuracy:P2}, Loss: {avgLoss:F4}");

        return new TrainingResult(config.Epochs, accuracy, avgLoss);
    }

    private static TrainingResult TrainInternal(
        Module<Tensor, Tensor> model,
        Action<double> sgdStep,
        int epochs,
        int batchSize,
        double learningRate,
        string dataPath)
    {
        var config = new TrainingConfiguration
        {
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = learningRate,
            DataPath = dataPath
        };
        return TrainInternal(model, sgdStep, config, null);
    }

    private static (double accuracy, double avgLoss) EvaluateInternal(
        Module<Tensor, Tensor> model,
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
