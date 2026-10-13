using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace ModelTraining.Services
{
    /// <summary>
    /// Service for running a multi-class ONNX classification model (Method 2).
    /// Used for categorizing specific geometric or visual defects (e.g., OPEN, SHORT, PASS).
    /// </summary>
    public class OnnxClassifierService : IDisposable
    {
        private const int InputSize = 224;

        // ImageNet normalization constants (matches Python transforms)
        private static readonly float[] Scale =
            { 1f / (255f * 0.229f), 1f / (255f * 0.224f), 1f / (255f * 0.225f) };
        private static readonly float[] Shift =
            { -0.485f / 0.229f, -0.456f / 0.224f, -0.406f / 0.225f };

        private readonly InferenceSession _session;
        private readonly string _inputName;

        // Dynamically populated from recipe.xml
        private readonly List<string> _classNames;

        public string Provider { get; private set; } = "CPU";

        public OnnxClassifierService(string modelPath, List<string> classNames, int dmlDeviceId = 1)
        {
            _classNames = classNames ?? new List<string>();

            try
            {
                // Attempt to load with DirectML for GPU acceleration
                var gpuOptions = new SessionOptions
                {
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    EnableMemoryPattern = false
                };
                gpuOptions.AppendExecutionProvider_DML(dmlDeviceId);
                _session = new InferenceSession(modelPath, gpuOptions);
                Provider = "DirectML";
            }
            catch (Exception ex)
            {
                // Fallback to CPU if DirectML fails
                Debug.WriteLine($"DirectML unavailable for Classifier, using CPU: {ex.Message}");
                var cpuOptions = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    IntraOpNumThreads = 6
                };
                _session = new InferenceSession(modelPath, cpuOptions);
                Provider = "CPU";
            }

            _inputName = _session.InputMetadata.Keys.First();
        }

        /// <summary>
        /// Runs the crop through the classification model and returns the predicted label and its confidence probability.
        /// </summary>
        public string ClassifyDefect(Mat bgrCrop, out float confidence)
        {
            if (_classNames.Count == 0)
            {
                throw new InvalidOperationException("Class names list is empty. Ensure recipe.xml contains the <DefectClasses> definition.");
            }

            var data = new float[3 * InputSize * InputSize];
            FillTensorData(bgrCrop, data, 0);

            var tensor = new DenseTensor<float>(data, new[] { 1, 3, InputSize, InputSize });

            using var results = _session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, tensor)
            });

            // Extract the probabilities for each class (e.g., [0.05, 0.90, 0.05])
            float[] probabilities = results.First().AsTensor<float>().ToArray();

            // Find the index with the highest probability
            int maxIndex = 0;
            float maxProb = 0f;
            for (int i = 0; i < probabilities.Length; i++)
            {
                if (probabilities[i] > maxProb)
                {
                    maxProb = probabilities[i];
                    maxIndex = i;
                }
            }

            confidence = maxProb;

            // Safety bounds check in case the model outputs more nodes than we have classes defined in XML
            if (maxIndex >= _classNames.Count)
            {
                return "UNKNOWN_CLASS";
            }

            return _classNames[maxIndex];
        }

        /// <summary>
        /// Matches Python: BGR -> RGB, Resize(224,224), /255, ImageNet mean/std, NCHW.
        /// Writes one image into 'data' starting at 'offset' (planar R, G, B).
        /// </summary>
        private static void FillTensorData(Mat imgBgr, float[] data, int offset)
        {
            if (imgBgr.Channels() != 3)
                throw new ArgumentException("The model expects a 3-channel color crop, not grayscale.");

            using var resized = new Mat();
            Cv2.Resize(imgBgr, resized, new Size(InputSize, InputSize), 0, 0, InterpolationFlags.Linear);

            using var rgb = new Mat();
            Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);
            rgb.GetArray(out Vec3b[] px);

            int plane = InputSize * InputSize;
            for (int i = 0; i < plane; i++)
            {
                Vec3b c = px[i];                 // after BGR2RGB: Item0 = R, Item1 = G, Item2 = B
                data[offset + i] = c.Item0 * Scale[0] + Shift[0];
                data[offset + plane + i] = c.Item1 * Scale[1] + Shift[1];
                data[offset + 2 * plane + i] = c.Item2 * Scale[2] + Shift[2];
            }
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}