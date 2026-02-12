namespace TorchSharpNetworkReference.Inspection;

/// <summary>
/// Contains captured data for multiple iterations, plus metadata.
/// </summary>
public class ForwardPassData
{
    public string ModelArchitecture { get; set; } = "MLP: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10";
    public string UpdateRule { get; set; } = "ManualSGD";
    public string LossFunction { get; set; } = "CrossEntropyLoss";
    public double LearningRate { get; set; }
    public int BatchSize { get; set; }
    public long RandomSeed { get; set; }
    public List<IterationData> Iterations { get; set; } = new();
}
