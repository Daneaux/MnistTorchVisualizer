using System.Text.Json;
using System.Text.Json.Serialization;
using TorchSharpNetworkReference.Inspection;

namespace TorchSharpNetworkReference.Serialization;

public static class IterationDataSerializer
{
    private static readonly JsonSerializerOptions Options = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    public static string Serialize(ForwardPassData data)
    {
        return JsonSerializer.Serialize(data, Options);
    }

    public static ForwardPassData? Deserialize(string json)
    {
        return JsonSerializer.Deserialize<ForwardPassData>(json, Options);
    }

    public static void SerializeToFile(ForwardPassData data, string filePath)
    {
        var json = Serialize(data);
        File.WriteAllText(filePath, json);
    }

    public static ForwardPassData? DeserializeFromFile(string filePath)
    {
        var json = File.ReadAllText(filePath);
        return Deserialize(json);
    }
}
