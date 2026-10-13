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
    /// Runs the end-to-end ONNX recipe exported by EdgeLearning.py.
    /// The model extracts a 1280-dimension feature vector from the live crop.
    ///
    /// Requires the NuGet package Microsoft.ML.OnnxRuntime.DirectML
    /// (remove Microsoft.ML.OnnxRuntime; only one of them can be installed).
    /// Falls back to CPU automatically if DirectML / the GPU cannot be used.
    /// </summary>
    public class OnnxInferenceService : IDisposable
    {
        private const int InputSize = 224;

        // Same as transforms.Normalize in EdgeLearning.py, folded into  x * Scale + Shift
        //   (x/255 - mean) / std  ==  x * (1/(255*std)) + (-mean/std)
        private static readonly float[] Scale =
            { 1f / (255f * 0.229f), 1f / (255f * 0.224f), 1f / (255f * 0.225f) };
        private static readonly float[] Shift =
            { -0.485f / 0.229f, -0.456f / 0.224f, -0.406f / 0.225f };

        private const int MaxBatch = 64;

        private readonly InferenceSession _session;
        private readonly string _inputName;
        private readonly int _featureDim;

        /// <summary>"DirectML" or "CPU": which provider the session is really using.</summary>
        public string Provider { get; private set; } = "CPU";

        /// <summary>True when the ONNX was exported with a dynamic batch dimension.</summary>
        public bool SupportsBatch { get; }

        /// <param name="dmlDeviceId">
        /// GPU index for DirectML. This laptop has Intel Iris Xe and NVIDIA T550:
        /// try 0 and 1 and watch Task Manager -> Performance to see which one is busy.
        /// </param>
        public OnnxInferenceService(string modelPath, int dmlDeviceId = 1)
        {
            try
            {
                var gpuOptions = new SessionOptions
                {
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,   // required for DirectML
                    EnableMemoryPattern = false                     // required for DirectML
                };
                gpuOptions.AppendExecutionProvider_DML(dmlDeviceId);
                _session = new InferenceSession(modelPath, gpuOptions);
                Provider = "DirectML";
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"DirectML unavailable, using CPU: {ex.Message}");
                var cpuOptions = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    IntraOpNumThreads = 6        // try 4, 6, 8, 12 and keep the fastest
                };
                _session = new InferenceSession(modelPath, cpuOptions);
                Provider = "CPU";
            }

            _inputName = _session.InputMetadata.Keys.First();   // "input"

            // A dynamic batch dimension shows up as -1 (or 0) in the metadata
            int[] dims = _session.InputMetadata[_inputName].Dimensions;
            SupportsBatch = dims.Length > 0 && dims[0] <= 0;

            // Determine output dimension size (usually 1280 for MobileNetV4)
            var outDims = _session.OutputMetadata.First().Value.Dimensions;
            _featureDim = outDims.Length > 1 ? outDims[1] : 1280;

            Debug.WriteLine($"ONNX provider: {Provider} | dynamic batch: {SupportsBatch}");

            // Warm-up so the first real board is not slow (GPU / kernel initialisation)
            try
            {
                using var dummy = new Mat(InputSize, InputSize, MatType.CV_8UC3, Scalar.All(128));
                ExtractFeatureVector(dummy);
                ExtractFeatureVector(dummy);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Warm-up failed: {ex.Message}");
            }
        }

        /// <summary>
        /// Runs the cropped image through the ONNX backbone and returns the feature array.
        /// </summary>
        public float[] ExtractFeatureVector(Mat bgrCrop)
        {
            var data = new float[3 * InputSize * InputSize];
            FillTensorData(bgrCrop, data, 0);
            var tensor = new DenseTensor<float>(data, new[] { 1, 3, InputSize, InputSize });

            using var results = _session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, tensor)
            });

            // Returns the full array of decimal numbers instead of a single score
            return results.First().AsTensor<float>().ToArray();
        }

        /// <summary>
        /// Mathematically compares the live feature array against the XML's Golden Vector.
        /// </summary>
        public float GetCosineSimilarity(float[] vectorA, float[] vectorB)
        {
            if (vectorA.Length != vectorB.Length) return 0.0f;

            double dotProduct = 0.0;
            double normA = 0.0;
            double normB = 0.0;

            for (int i = 0; i < vectorA.Length; i++)
            {
                dotProduct += vectorA[i] * vectorB[i];
                normA += Math.Pow(vectorA[i], 2);
                normB += Math.Pow(vectorB[i], 2);
            }

            if (normA == 0 || normB == 0) return 0.0f;
            return (float)(dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB)));
        }

        /// <summary>
        /// All crops in as few model calls as possible. Needs an ONNX exported with a dynamic
        /// batch dimension; otherwise it falls back to one call per crop.
        /// </summary>
        public List<float[]> ExtractFeatureVectors(IReadOnlyList<Mat> bgrCrops)
        {
            int total = bgrCrops.Count;
            var vectors = new List<float[]>();
            if (total == 0) return vectors;

            if (!SupportsBatch)
            {
                for (int i = 0; i < total; i++) vectors.Add(ExtractFeatureVector(bgrCrops[i]));
                return vectors;
            }

            int plane = InputSize * InputSize;
            int stride = 3 * plane;

            for (int start = 0; start < total; start += MaxBatch)
            {
                int n = Math.Min(MaxBatch, total - start);
                var data = new float[n * stride];

                for (int k = 0; k < n; k++)
                    FillTensorData(bgrCrops[start + k], data, k * stride);

                var tensor = new DenseTensor<float>(data, new[] { n, 3, InputSize, InputSize });

                using var results = _session.Run(new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(_inputName, tensor)
                });

                float[] batchScores = results.First().AsTensor<float>().ToArray();

                // Split the flat batch array back into individual component vectors
                for (int i = 0; i < n; i++)
                {
                    float[] singleVector = new float[_featureDim];
                    Array.Copy(batchScores, i * _featureDim, singleVector, 0, _featureDim);
                    vectors.Add(singleVector);
                }
            }
            return vectors;
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

        public void Dispose() => _session?.Dispose();
    }
}