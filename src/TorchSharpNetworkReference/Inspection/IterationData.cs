namespace TorchSharpNetworkReference.Inspection;

/// <summary>
/// Captures all data for a single training iteration (one batch: forward + backward + update).
/// </summary>
public class IterationData
{
    /// <summary>0-based iteration index.</summary>
    public int IterationIndex { get; set; }

    /// <summary>Input tensor flattened to [batchSize, 784].</summary>
    public float[] Input { get; set; } = Array.Empty<float>();

    /// <summary>Shape of input tensor.</summary>
    public long[] InputShape { get; set; } = Array.Empty<long>();

    /// <summary>Target labels for this batch.</summary>
    public long[] Targets { get; set; } = Array.Empty<long>();

    /// <summary>Loss value for this iteration.</summary>
    public float Loss { get; set; }

    /// <summary>Final logits output. Shape: [batchSize, 10].</summary>
    public float[] Logits { get; set; } = Array.Empty<float>();

    /// <summary>Shape of logits tensor.</summary>
    public long[] LogitsShape { get; set; } = Array.Empty<long>();

    /// <summary>Layer snapshots BEFORE forward pass (initial weights/biases).</summary>
    public Dictionary<string, LayerSnapshot> LayersBefore { get; set; } = new();

    /// <summary>Layer snapshots AFTER backward (activations + gradients).</summary>
    public Dictionary<string, LayerSnapshot> LayersAfterBackward { get; set; } = new();

    /// <summary>Layer snapshots AFTER manual SGD step (updated weights/biases).</summary>
    public Dictionary<string, LayerSnapshot> LayersAfterUpdate { get; set; } = new();
}
