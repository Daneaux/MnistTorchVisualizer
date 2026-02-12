namespace TorchSharpNetworkReference.Inspection;

/// <summary>
/// Captures all values for a single layer at a point in time.
/// </summary>
public class LayerSnapshot
{
    public string LayerName { get; set; } = "";

    /// <summary>Weight matrix flattened to 1D.</summary>
    public float[] Weights { get; set; } = Array.Empty<float>();

    /// <summary>Shape of the weight matrix, e.g. [128, 784].</summary>
    public long[] WeightShape { get; set; } = Array.Empty<long>();

    /// <summary>Bias vector.</summary>
    public float[] Biases { get; set; } = Array.Empty<float>();

    /// <summary>Pre-activation values (output of Linear, before ReLU). Flattened.</summary>
    public float[]? PreActivation { get; set; }

    /// <summary>Pre-activation shape, e.g. [batchSize, 128].</summary>
    public long[]? PreActivationShape { get; set; }

    /// <summary>Post-activation values (after ReLU). Null for output layer (no ReLU).</summary>
    public float[]? PostActivation { get; set; }

    /// <summary>Post-activation shape.</summary>
    public long[]? PostActivationShape { get; set; }

    /// <summary>Weight gradients after backward pass. Null before backward.</summary>
    public float[]? WeightGradients { get; set; }

    /// <summary>Weight gradient shape.</summary>
    public long[]? WeightGradientShape { get; set; }

    /// <summary>Bias gradients after backward pass. Null before backward.</summary>
    public float[]? BiasGradients { get; set; }
}
