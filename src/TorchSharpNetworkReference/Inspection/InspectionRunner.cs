using TorchSharp;
using TorchSharpNetworkReference.Models;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision;

namespace TorchSharpNetworkReference.Inspection;

public static class InspectionRunner
{
    private static readonly string[] LayerNames = { "fc1", "fc2", "fc3" };

    public static ForwardPassData CaptureIterations(
        int iterationCount = 2,
        int batchSize = 4,
        double learningRate = 0.01,
        long seed = 42,
        string dataPath = "./data")
    {
        manual_seed(seed);

        var model = new MnistModel();
        var lossFn = CrossEntropyLoss();

        using var trainData = datasets.MNIST(dataPath, train: true, download: true);
        using var trainLoader = DataLoader(trainData, batchSize, shuffle: false);

        var result = new ForwardPassData
        {
            LearningRate = learningRate,
            BatchSize = batchSize,
            RandomSeed = seed,
        };

        int iteration = 0;
        foreach (var batch in trainLoader)
        {
            if (iteration >= iterationCount) break;

            var data = batch["data"];
            var target = batch["label"];

            var iterData = new IterationData
            {
                IterationIndex = iteration,
            };

            // Capture input
            var flatInput = data.view(-1, 784);
            iterData.Input = ToFloatArray(flatInput);
            iterData.InputShape = flatInput.shape;
            iterData.Targets = ToLongArray(target);
            flatInput.Dispose();

            // 1. Capture weights/biases BEFORE forward
            iterData.LayersBefore = CaptureAllLayerWeights(model);

            // 2. Forward with intermediates
            model.train();
            var (preAct1, postAct1, preAct2, postAct2, logits) = model.ForwardWithIntermediates(data);

            iterData.Logits = ToFloatArray(logits);
            iterData.LogitsShape = logits.shape;

            // 3. Compute loss
            var loss = lossFn.call(logits, target);
            iterData.Loss = loss.item<float>();

            // 4. Backward
            model.zero_grad();
            loss.backward();

            // 5. Capture after backward (activations + gradients)
            iterData.LayersAfterBackward = CaptureAllLayerWeights(model);
            AddActivations(iterData.LayersAfterBackward["fc1"], preAct1, postAct1);
            AddActivations(iterData.LayersAfterBackward["fc2"], preAct2, postAct2);
            AddActivations(iterData.LayersAfterBackward["fc3"], logits, null);
            AddGradients(iterData.LayersAfterBackward, model);

            // 6. Manual SGD step
            model.ManualSgdStep(learningRate);

            // 7. Capture after update
            iterData.LayersAfterUpdate = CaptureAllLayerWeights(model);

            result.Iterations.Add(iterData);

            // Dispose intermediates
            preAct1.Dispose();
            postAct1.Dispose();
            preAct2.Dispose();
            postAct2.Dispose();
            logits.Dispose();
            loss.Dispose();

            iteration++;
        }

        return result;
    }

    private static Dictionary<string, LayerSnapshot> CaptureAllLayerWeights(MnistModel model)
    {
        var snapshots = new Dictionary<string, LayerSnapshot>();
        var layers = new (string name, TorchSharp.Modules.Linear layer)[]
        {
            ("fc1", model.fc1),
            ("fc2", model.fc2),
            ("fc3", model.fc3),
        };

        foreach (var (name, layer) in layers)
        {
            var weight = layer.weight!;
            var bias = layer.bias!;

            snapshots[name] = new LayerSnapshot
            {
                LayerName = name,
                Weights = ToFloatArray(weight),
                WeightShape = weight.shape,
                Biases = ToFloatArray(bias),
            };
        }

        return snapshots;
    }

    private static void AddActivations(LayerSnapshot snapshot, Tensor preAct, Tensor? postAct)
    {
        snapshot.PreActivation = ToFloatArray(preAct);
        snapshot.PreActivationShape = preAct.shape;
        if (postAct is not null)
        {
            snapshot.PostActivation = ToFloatArray(postAct);
            snapshot.PostActivationShape = postAct.shape;
        }
    }

    private static void AddGradients(Dictionary<string, LayerSnapshot> snapshots, MnistModel model)
    {
        var layers = new (string name, TorchSharp.Modules.Linear layer)[]
        {
            ("fc1", model.fc1),
            ("fc2", model.fc2),
            ("fc3", model.fc3),
        };

        foreach (var (name, layer) in layers)
        {
            var snapshot = snapshots[name];
            var wGrad = layer.weight!.grad;
            var bGrad = layer.bias!.grad;

            if (wGrad is not null)
            {
                snapshot.WeightGradients = ToFloatArray(wGrad);
                snapshot.WeightGradientShape = wGrad.shape;
            }
            if (bGrad is not null)
            {
                snapshot.BiasGradients = ToFloatArray(bGrad);
            }
        }
    }

    private static float[] ToFloatArray(Tensor t)
    {
        return t.detach().cpu().data<float>().ToArray();
    }

    private static long[] ToLongArray(Tensor t)
    {
        return t.detach().cpu().data<long>().ToArray();
    }
}
