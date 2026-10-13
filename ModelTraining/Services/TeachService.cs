using ModelTraining.ToolLibrary;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace ModelTraining.Services
{
    /// <summary>One taught component type (e.g. PAD, RESISTOR).</summary>
    public sealed class TeachComponent : IDisposable
    {
        public string Name { get; }
        public Mat Template { get; }                          // gray template cut from the reference
        public List<Rect> Instances { get; } = new List<Rect>();   // every location on the reference board
        public List<Mat> PassCrops { get; } = new List<Mat>();     // color crops from the PASS images
        public float[]? GoldenVector { get; set; }

        public TeachComponent(string name, Mat template)
        {
            Name = name;
            Template = template;
        }

        public void ClearCrops()
        {
            foreach (Mat c in PassCrops) c.Dispose();
            PassCrops.Clear();
        }

        public void Dispose()
        {
            ClearCrops();
            Template.Dispose();
        }
    }

    /// <summary>
    /// The whole teaching workflow of Main8.py in C#:
    ///   reference image -> fiducial -> ONE search region -> any number of components ->
    ///   extract PASS crops -> Instant Teach (golden vectors) -> save the .aoi package.
    /// Because teaching now runs through the same C# preprocessing as inspection,
    /// there is no Python/C# preprocessing mismatch any more.
    /// </summary>
    public sealed class TeachService : IDisposable
    {
        private readonly List<TeachComponent> _components = new List<TeachComponent>();
        private Mat? _reference;
        private Mat? _referenceGray;
        private Mat? _fiducial;
        private Point _fiducialPos;

        public double MatchThreshold { get; set; } = 0.75;     // template match threshold
        public double NmsIou { get; set; } = 0.4;
        public float Threshold { get; set; } = 0.80f;          // PASS/NG similarity threshold
        public DefectParams Defects { get; } = new DefectParams();

        public Mat? Reference => _reference;
        public Rect? FiducialRect { get; private set; }
        public Rect? SearchRect { get; private set; }
        public IReadOnlyList<TeachComponent> Components => _components;

        public bool HasReference => _reference != null;
        public bool HasFiducial => _fiducial != null;
        public bool HasSearchRegion => SearchRect != null;
        public bool HasComponents => _components.Count > 0 && _components.Any(c => c.Instances.Count > 0);
        public bool HasPassCrops => _components.Count > 0 && _components.All(c => c.PassCrops.Count > 0);

        // ------------------------------------------------------------------ setup
        public bool LoadReference(string path)
        {
            Mat img = Cv2.ImRead(path, ImreadModes.Color);
            if (img.Empty()) { img.Dispose(); return false; }

            Reset();
            _reference = img;
            _referenceGray = new Mat();
            Cv2.CvtColor(img, _referenceGray, ColorConversionCodes.BGR2GRAY);
            return true;
        }

        private void Reset()
        {
            foreach (TeachComponent c in _components) c.Dispose();
            _components.Clear();
            _reference?.Dispose();
            _referenceGray?.Dispose();
            _fiducial?.Dispose();
            _reference = _referenceGray = _fiducial = null;
            FiducialRect = SearchRect = null;
        }

        public void SetFiducial(Rect roi)
        {
            Rect r = Clamp(roi);
            _fiducial?.Dispose();
            using Mat view = new Mat(_referenceGray!, r);
            _fiducial = view.Clone();
            _fiducialPos = new Point(r.X, r.Y);
            FiducialRect = r;
        }

        /// <summary>Declared ONCE. If components already exist they are re-mapped inside the new region.</summary>
        public void SetSearchRegion(Rect roi)
        {
            Rect previous = SearchRect ?? default;
            bool hadPrevious = SearchRect != null;
            SearchRect = Clamp(roi);
            try
            {
                MapInstances();
            }
            catch
            {
                SearchRect = hadPrevious ? previous : (Rect?)null;
                if (hadPrevious) MapInstances();
                throw;
            }
        }

        /// <summary>Teach one more component type inside the search region. Returns name -> instance count.</summary>
        public Dictionary<string, int> AddComponent(string name, Rect roi)
        {
            if (SearchRect == null) throw new InvalidOperationException("Draw the search region first.");

            name = Regex.Replace(name.Trim(), "[^A-Za-z0-9_-]", "_");
            if (name.Length == 0) name = "COMPONENT";

            Rect r = Clamp(roi);
            using Mat view = new Mat(_referenceGray!, r);
            var comp = new TeachComponent(name, view.Clone());

            TeachComponent? previous = _components.FirstOrDefault(c => c.Name == name);
            int index = previous != null ? _components.IndexOf(previous) : -1;
            if (index >= 0) _components[index] = comp; else _components.Add(comp);

            try
            {
                MapInstances();
            }
            catch
            {
                // roll back
                if (index >= 0) _components[index] = previous!; else _components.Remove(comp);
                comp.Dispose();
                MapInstances();
                throw;
            }

            previous?.Dispose();
            return ComponentCounts();
        }

        public void RemoveComponent(string name)
        {
            TeachComponent? c = _components.FirstOrDefault(x => x.Name == name);
            if (c == null) return;
            _components.Remove(c);
            c.Dispose();
            MapInstances();
        }

        public Dictionary<string, int> ComponentCounts() =>
            _components.ToDictionary(c => c.Name, c => c.Instances.Count);

        /// <summary>Finds every taught component inside the single search region, one NMS across all.</summary>
        private void MapInstances()
        {
            foreach (TeachComponent c in _components) c.Instances.Clear();
            if (SearchRect == null || _components.Count == 0) return;

            Rect sr = SearchRect.Value;
            using Mat searchCrop = new Mat(_referenceGray!, sr);

            var candidates = new List<(Rect box, float score, int comp)>();
            for (int ci = 0; ci < _components.Count; ci++)
            {
                Mat tpl = _components[ci].Template;
                if (tpl.Width > searchCrop.Width || tpl.Height > searchCrop.Height)
                    throw new InvalidOperationException($"Component '{_components[ci].Name}' is bigger than the search region.");

                using Mat res = new Mat();
                Cv2.MatchTemplate(searchCrop, tpl, res, TemplateMatchModes.CCoeffNormed);
                res.GetArray(out float[] data);
                int cols = res.Cols;

                for (int i = 0; i < data.Length; i++)
                {
                    if (data[i] < MatchThreshold) continue;
                    candidates.Add((new Rect(i % cols + sr.X, i / cols + sr.Y, tpl.Width, tpl.Height), data[i], ci));
                    if (candidates.Count > 300000)
                        throw new InvalidOperationException(
                            "Far too many matches. Use a more distinctive component or a smaller search region.");
                }
            }

            // Greedy non-maximum suppression across all components (best score first)
            candidates.Sort((a, b) => b.score.CompareTo(a.score));
            var kept = new List<(Rect box, float score, int comp)>();
            foreach (var cand in candidates)
            {
                bool suppressed = false;
                foreach (var k in kept)
                {
                    if (IoU(cand.box, k.box) > NmsIou) { suppressed = true; break; }
                }
                if (!suppressed) kept.Add(cand);
            }

            foreach (var k in kept) _components[k.comp].Instances.Add(k.box);
        }

        private static double IoU(Rect a, Rect b)
        {
            int x1 = Math.Max(a.X, b.X), y1 = Math.Max(a.Y, b.Y);
            int x2 = Math.Min(a.Right, b.Right), y2 = Math.Min(a.Bottom, b.Bottom);
            int iw = x2 - x1, ih = y2 - y1;
            if (iw <= 0 || ih <= 0) return 0;
            double inter = (double)iw * ih;
            double union = (double)a.Width * a.Height + (double)b.Width * b.Height - inter;
            return inter / union;
        }

        private Rect Clamp(Rect r)
        {
            if (_reference == null) throw new InvalidOperationException("Load the reference image first.");
            int x0 = Math.Max(0, r.X), y0 = Math.Max(0, r.Y);
            int x1 = Math.Min(_reference.Width, r.Right), y1 = Math.Min(_reference.Height, r.Bottom);
            if (x1 - x0 < 4 || y1 - y0 < 4)
                throw new InvalidOperationException("The selected box is too small or outside the image.");
            return new Rect(x0, y0, x1 - x0, y1 - y0);
        }

        // ------------------------------------------------------------------ alignment
        private (int dx, int dy) Align(Mat gray)
        {
            if (_fiducial == null) return (0, 0);
            using Mat res = new Mat();
            Cv2.MatchTemplate(gray, _fiducial, res, TemplateMatchModes.CCoeffNormed);
            Cv2.MinMaxLoc(res, out _, out _, out _, out Point loc);
            return (loc.X - _fiducialPos.X, loc.Y - _fiducialPos.Y);
        }

        // ------------------------------------------------------------------ PASS data
        /// <summary>Cuts every component instance out of the given PASS board images. Returns the crop count.</summary>
        public int ExtractPassCrops(IEnumerable<string> files)
        {
            foreach (TeachComponent c in _components) c.ClearCrops();

            int total = 0;
            foreach (string file in files)
            {
                using Mat img = Cv2.ImRead(file, ImreadModes.Color);
                if (img.Empty()) continue;

                using Mat gray = new Mat();
                Cv2.CvtColor(img, gray, ColorConversionCodes.BGR2GRAY);
                var (dx, dy) = Align(gray);

                foreach (TeachComponent comp in _components)
                {
                    foreach (Rect inst in comp.Instances)
                    {
                        var box = new Rect(inst.X + dx, inst.Y + dy, inst.Width, inst.Height);
                        if (box.X < 0 || box.Y < 0 || box.Right > img.Width || box.Bottom > img.Height) continue;

                        using Mat view = new Mat(img, box);
                        comp.PassCrops.Add(view.Clone());
                        total++;
                    }
                }
            }
            return total;
        }
        
        public (bool ok, string message, int images) InstantTeach(OnnxInferenceService backbone)
        {
            if (_components.Count == 0) return (false, "No component taught.", 0);

            int total = 0;
            foreach (TeachComponent comp in _components)
            {
                if (comp.PassCrops.Count == 0)
                    return (false, $"No PASS crops for '{comp.Name}' (run Extract PASS first).", 0);

                List<float[]> vectors = backbone.ExtractFeatureVectors(comp.PassCrops);
                var mean = new float[vectors[0].Length];
                foreach (float[] v in vectors)
                {
                    float[] n = Normalize(v);
                    for (int i = 0; i < mean.Length; i++) mean[i] += n[i];
                }
                for (int i = 0; i < mean.Length; i++) mean[i] /= vectors.Count;

                comp.GoldenVector = Normalize(mean);
                total += vectors.Count;
            }
            return (true, $"{_components.Count} component(s), {total} images", total);
        }

        private static float[] Normalize(float[] v)
        {
            double sum = 0;
            foreach (float x in v) sum += (double)x * x;
            float inv = sum > 0 ? (float)(1.0 / Math.Sqrt(sum)) : 1f;
            var o = new float[v.Length];
            for (int i = 0; i < v.Length; i++) o[i] = v[i] * inv;
            return o;
        }

        
        public void SaveAoi(string destination, string backboneOnnxPath, string backboneName)
        {
            if (_reference == null || _fiducial == null || SearchRect == null)
                throw new InvalidOperationException("The reference, fiducial and search region must be set.");
            if (_components.Any(c => c.GoldenVector == null))
                throw new InvalidOperationException("Run Instant Teach first.");
            if (!File.Exists(backboneOnnxPath))
                throw new FileNotFoundException("Backbone ONNX not found.", backboneOnnxPath);

            string work = Path.Combine(Path.GetTempPath(), "ModelTraining", "export_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(work);
            try
            {
                Cv2.ImWrite(Path.Combine(work, "reference.png"), _reference);
                Cv2.ImWrite(Path.Combine(work, "fiducial.png"), _fiducial);
                foreach (TeachComponent c in _components)
                    Cv2.ImWrite(Path.Combine(work, $"tpl_{c.Name}.png"), c.Template);

                string modelFile = Path.GetFileName(backboneOnnxPath);
                File.Copy(backboneOnnxPath, Path.Combine(work, modelFile));
                string dataFile = backboneOnnxPath + ".data";            // external weights, same base name
                if (File.Exists(dataFile))
                    File.Copy(dataFile, Path.Combine(work, modelFile + ".data"));

                WriteRecipeXml(Path.Combine(work, "recipe.xml"), modelFile, backboneName);

                if (File.Exists(destination)) File.Delete(destination);
                ZipFile.CreateFromDirectory(work, destination, CompressionLevel.NoCompression, false);
            }
            finally
            {
                try { Directory.Delete(work, true); } catch { /* temp folder, best effort */ }
            }
        }

        private void WriteRecipeXml(string path, string modelFile, string backboneName)
        {
            CultureInfo inv = CultureInfo.InvariantCulture;
            int dim = _components[0].GoldenVector!.Length;
            Rect s = SearchRect!.Value;
            DefectParams d = Defects;

            var components = new XElement("Components");
            foreach (TeachComponent c in _components)
            {
                components.Add(new XElement("Component",
                    new XAttribute("name", c.Name),
                    new XAttribute("width", c.Template.Width),
                    new XAttribute("height", c.Template.Height),
                    new XAttribute("template", $"tpl_{c.Name}.png"),
                    new XElement("GoldenVector",
                        new XAttribute("count", c.GoldenVector!.Length),
                        string.Join(" ", c.GoldenVector.Select(v => v.ToString("G9", inv)))),
                    new XElement("Instances",
                        c.Instances.Select(i => new XElement("Instance",
                            new XAttribute("x", i.X), new XAttribute("y", i.Y),
                            new XAttribute("width", i.Width), new XAttribute("height", i.Height))))));
            }

            var root = new XElement("AoiRecipe", new XAttribute("version", "2"),
                new XElement("Model",
                    new XAttribute("backbone", backboneName),
                    new XAttribute("file", modelFile),
                    new XAttribute("inputSize", 224),
                    new XAttribute("embeddingDim", dim)),
                new XElement("Reference", new XAttribute("image", "reference.png")),
                new XElement("Fiducial",
                    new XAttribute("x", _fiducialPos.X), new XAttribute("y", _fiducialPos.Y),
                    new XAttribute("template", "fiducial.png")),
                new XElement("SearchRegion",
                    new XAttribute("x", s.X), new XAttribute("y", s.Y),
                    new XAttribute("width", s.Width), new XAttribute("height", s.Height)),
                new XElement("Inspection",
                    new XAttribute("threshold", Threshold.ToString("0.00", inv)),
                    new XAttribute("matchThreshold", MatchThreshold.ToString("0.00", inv)),
                    new XAttribute("nmsIou", NmsIou.ToString("0.00", inv))),
                new XElement("DefectParams",
                    new XAttribute("contextMargin", d.ContextMargin),
                    new XAttribute("alignSearch", d.AlignSearch),
                    new XAttribute("minBlobArea", d.MinBlobArea),
                    new XAttribute("holeMinArea", d.HoleMinArea),
                    new XAttribute("holeFillRatio", d.HoleFillRatio.ToString(inv)),
                    new XAttribute("biteMaxFraction", d.BiteMaxFraction.ToString(inv))),
                components);

            new XDocument(new XDeclaration("1.0", "utf-8", null), root).Save(path);
        }

        public void Dispose() => Reset();
    }
}
