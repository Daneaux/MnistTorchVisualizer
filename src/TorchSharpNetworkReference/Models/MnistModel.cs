using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharpNetworkReference.Models;

public class MnistModel : Module<Tensor, Tensor>
{
    public readonly Linear fc1;
    public readonly Linear fc2;
    public readonly Linear fc3;

    public MnistModel(string name = "MnistMLP") : base(name)
    {
        fc1 = Linear(784, 128);
        fc2 = Linear(128, 64);
        fc3 = Linear(64, 10);
        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        using var flat = input.view(-1, 784);
        using var h1 = functional.relu(fc1.call(flat));
        using var h2 = functional.relu(fc2.call(h1));
        return fc3.call(h2);
    }

    /// <summary>
    /// Forward pass returning all intermediate values for inspection.
    /// Caller is responsible for disposing returned tensors.
    /// </summary>
    public (Tensor preAct1, Tensor postAct1, Tensor preAct2, Tensor postAct2, Tensor logits) ForwardWithIntermediates(Tensor input)
    {
        var flat = input.view(-1, 784);
        var preAct1 = fc1.call(flat);
        var postAct1 = functional.relu(preAct1);
        var preAct2 = fc2.call(postAct1);
        var postAct2 = functional.relu(preAct2);
        var logits = fc3.call(postAct2);
        flat.Dispose();
        return (preAct1, postAct1, preAct2, postAct2, logits);
    }

    /// <summary>
    /// Manual SGD update: param -= learningRate * param.grad for all parameters.
    /// </summary>
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
}
