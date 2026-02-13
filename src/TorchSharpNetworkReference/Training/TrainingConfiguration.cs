namespace TorchSharpNetworkReference.Training;

/// <summary>
/// Configuration for MNIST model training. All parameters in one place.
/// </summary>
public record TrainingConfiguration
{
    /// <summary>Hidden layer sizes (e.g., [128, 64] for 784→128→64→10)</summary>
    public int[] HiddenLayerSizes { get; init; } = { 128, 64 };

    /// <summary>Number of training epochs</summary>
    public int Epochs { get; init; } = 10;

    /// <summary>Batch size for training</summary>
    public int BatchSize { get; init; } = 64;

    /// <summary>Learning rate for SGD optimizer</summary>
    public double LearningRate { get; init; } = 0.05;

    /// <summary>Path to MNIST dataset</summary>
    public string DataPath { get; init; } = "./data";

    /// <summary>
    /// Returns true if this config uses the default architecture (128, 64).
    /// </summary>
    public bool IsDefaultArchitecture => 
        HiddenLayerSizes.Length == 2 && 
        HiddenLayerSizes[0] == 128 && 
        HiddenLayerSizes[1] == 64;

    /// <summary>
    /// Validates the configuration and returns validation errors if any.
    /// </summary>
    public static ValidationResult Validate(TrainingConfiguration config)
    {
        var errors = new List<string>();

        // Validate HiddenLayerSizes
        if (config.HiddenLayerSizes == null || config.HiddenLayerSizes.Length == 0)
        {
            errors.Add("Architecture must have at least 1 hidden layer");
        }
        else if (config.HiddenLayerSizes.Length > 5)
        {
            errors.Add("Architecture cannot have more than 5 hidden layers");
        }
        else
        {
            for (int i = 0; i < config.HiddenLayerSizes.Length; i++)
            {
                var size = config.HiddenLayerSizes[i];
                if (size < 2)
                    errors.Add($"Layer {i + 1}: Must have at least 2 neurons");
                else if (size > 1024)
                    errors.Add($"Layer {i + 1}: Cannot have more than 1024 neurons");
            }
        }

        // Validate Epochs
        if (config.Epochs < 1)
            errors.Add("Epochs must be at least 1");
        else if (config.Epochs > 500)
            errors.Add("Epochs cannot exceed 500");

        // Validate BatchSize
        if (config.BatchSize < 1)
            errors.Add("Batch size must be at least 1");
        else if (config.BatchSize > 512)
            errors.Add("Batch size cannot exceed 512");

        // Validate LearningRate
        if (config.LearningRate < 0.0001)
            errors.Add("Learning rate must be at least 0.0001");
        else if (config.LearningRate > 1.0)
            errors.Add("Learning rate cannot exceed 1.0");

        return new ValidationResult(errors.Count == 0, errors);
    }

    /// <summary>
    /// Parses an architecture string (e.g., "128,64" or "256, 128, 64") into int array.
    /// Returns null if parsing fails.
    /// </summary>
    public static int[]? ParseArchitecture(string architecture)
    {
        if (string.IsNullOrWhiteSpace(architecture))
            return null;

        var parts = architecture.Split(',')
            .Select(s => s.Trim())
            .Where(s => !string.IsNullOrEmpty(s))
            .ToList();

        if (parts.Count == 0)
            return null;

        var result = new int[parts.Count];
        for (int i = 0; i < parts.Count; i++)
        {
            if (!int.TryParse(parts[i], out var value))
                return null;
            result[i] = value;
        }

        return result;
    }

    /// <summary>
    /// Returns a human-readable description of the architecture.
    /// </summary>
    public string GetArchitectureDescription()
    {
        var parts = new List<string> { "784" };
        foreach (var size in HiddenLayerSizes)
            parts.Add($"{size} (ReLU)");
        parts.Add("10");
        return string.Join(" → ", parts);
    }

    /// <summary>
    /// Returns a summary string for display in status bars.
    /// </summary>
    public override string ToString()
    {
        return $"{GetArchitectureDescription()}, {Epochs} epochs, batch={BatchSize}, lr={LearningRate}";
    }
}

/// <summary>
/// Result of configuration validation.
/// </summary>
public record ValidationResult(bool IsValid, IReadOnlyList<string> Errors)
{
    public string ErrorMessage => string.Join("; ", Errors);
}
