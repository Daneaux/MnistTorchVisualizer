using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using TorchSharp;
using TorchSharpNetworkReference.Models;
using TorchSharpNetworkReference.Training;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision;

namespace TorchSharpNetworkReference.Visualizer;

public partial class MainWindow : Window
{
    private MnistModel? _model;
    private List<(float[] pixels, long label)> _testImages = new();
    private int _currentIndex;

    public MainWindow()
    {
        InitializeComponent();
    }

    private async void TrainButton_Click(object sender, RoutedEventArgs e)
    {
        TrainButton.IsEnabled = false;
        PrevButton.IsEnabled = false;
        NextButton.IsEnabled = false;
        StatusText.Text = "Training network (10 epochs, lr=0.05)...";

        MnistTrainer.TrainingResult? result = null;

        await Task.Run(() =>
        {
            (_model, result) = MnistTrainer.TrainAndReturn(
                epochs: 10, batchSize: 64, learningRate: 0.05);
        });

        AccuracyText.Text = $"Accuracy: {result!.FinalTestAccuracy:P2}";
        StatusText.Text = "Loading test images...";

        await Task.Run(LoadTestImages);

        StatusText.Text = $"Ready -- {_testImages.Count} test images loaded.";
        TrainButton.IsEnabled = true;
        PrevButton.IsEnabled = true;
        NextButton.IsEnabled = true;

        _currentIndex = 0;
        ShowCurrentImage();
    }

    private void LoadTestImages()
    {
        _testImages.Clear();

        using var testData = datasets.MNIST("./data", false, download: true);
        using var loader = DataLoader(testData, 256, shuffle: false);

        foreach (var batch in loader)
        {
            var data = batch["data"];   // [batch, 28, 28, 1] or [batch, 1, 28, 28]
            var labels = batch["label"];
            int batchSize = (int)data.shape[0];

            for (int i = 0; i < batchSize; i++)
            {
                using var img = data[i];
                var pixels = img.detach().cpu().data<float>().ToArray();
                var label = labels[i].item<long>();
                _testImages.Add((pixels, label));
            }
        }
    }

    private void ShowCurrentImage()
    {
        if (_model is null || _testImages.Count == 0) return;

        var (pixels, trueLabel) = _testImages[_currentIndex];

        // Display the digit image
        DigitImage.Source = CreateBitmap(pixels);

        // Run inference
        _model.eval();
        float[] probabilities;
        long predicted;

        using (no_grad())
        {
            using var input = tensor(pixels).view(1, 1, 28, 28);
            using var logits = _model.call(input);
            using var softmax = torch.nn.functional.softmax(logits, dim: 1);

            probabilities = softmax.detach().cpu().data<float>().ToArray();
            predicted = logits.argmax(1).item<long>();
        }

        // Update labels
        TrueLabelText.Text = trueLabel.ToString();
        PredictedText.Text = predicted.ToString();
        PredictedText.Foreground = predicted == trueLabel
            ? new SolidColorBrush(Color.FromRgb(34, 139, 34))
            : new SolidColorBrush(Color.FromRgb(200, 40, 40));

        IndexText.Text = $"{_currentIndex + 1} / {_testImages.Count}";

        // Update confidence bars
        var bars = new List<BarItem>();
        for (int d = 0; d < 10; d++)
        {
            float pct = probabilities[d];
            bool isMax = d == predicted;
            bars.Add(new BarItem
            {
                Label = d.ToString(),
                BarWidth = Math.Max(1, pct * 280),
                PctText = $"{pct * 100:F1}%",
                BarColor = isMax
                    ? new SolidColorBrush(Color.FromRgb(50, 130, 240))
                    : new SolidColorBrush(Color.FromRgb(180, 200, 220)),
            });
        }
        BarsPanel.ItemsSource = bars;
    }

    private static WriteableBitmap CreateBitmap(float[] pixels)
    {
        var bmp = new WriteableBitmap(28, 28, 96, 96, PixelFormats.Gray8, null);
        var bytePixels = new byte[28 * 28];

        // pixels may be [1,28,28] or [28,28,1] — just take first 784
        for (int i = 0; i < 784 && i < pixels.Length; i++)
        {
            bytePixels[i] = (byte)(Math.Clamp(pixels[i], 0f, 1f) * 255);
        }

        bmp.WritePixels(new System.Windows.Int32Rect(0, 0, 28, 28), bytePixels, 28, 0);
        return bmp;
    }

    private void PrevButton_Click(object sender, RoutedEventArgs e)
    {
        if (_testImages.Count == 0) return;
        _currentIndex = (_currentIndex - 1 + _testImages.Count) % _testImages.Count;
        ShowCurrentImage();
    }

    private void NextButton_Click(object sender, RoutedEventArgs e)
    {
        if (_testImages.Count == 0) return;
        _currentIndex = (_currentIndex + 1) % _testImages.Count;
        ShowCurrentImage();
    }
}

public class BarItem
{
    public string Label { get; set; } = "";
    public double BarWidth { get; set; }
    public string PctText { get; set; } = "";
    public SolidColorBrush BarColor { get; set; } = Brushes.Gray;
}
