using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpNetworkReference.Models;

public class ConfigurableModel : Module<Tensor, Tensor>
{
    private const int InputSize = 784;
    private const int OutputSize = 10;

    private readonly Linear[] _layers;
    public int[] HiddenSizes { get; }

    public ConfigurableModel(int[] hiddenSizes, string name = "ConfigurableMLP") : base(name)
    {
        HiddenSizes = hiddenSizes;

        var allSizes = new List<int> { InputSize };
        allSizes.AddRange(hiddenSizes);
        allSizes.Add(OutputSize);

        _layers = new Linear[allSizes.Count - 1];
        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i] = Linear(allSizes[i], allSizes[i + 1]);
            register_module($"fc{i + 1}", _layers[i]);
        }

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = input.view(-1, InputSize);
        for (int i = 0; i < _layers.Length - 1; i++)
        {
            x = functional.relu(_layers[i].call(x));
        }
        return _layers[^1].call(x);
    }

    public void ManualSgdStep(double learningRate)
    {
        using (no_grad())
        {
            foreach (var param in parameters())
            {
                var grad = param.grad;
                if (grad is not null)
                {
                    param.sub_(learningRate * grad);
                }
            }
        }
    }

    /// <summary>
    /// Returns a human-readable description, e.g. "784 -> 128 (ReLU) -> 64 (ReLU) -> 10"
    /// </summary>
    public string ArchitectureDescription()
    {
        var parts = new List<string> { InputSize.ToString() };
        foreach (var h in HiddenSizes)
            parts.Add($"{h} (ReLU)");
        parts.Add(OutputSize.ToString());
        return string.Join(" -> ", parts);
    }
}
