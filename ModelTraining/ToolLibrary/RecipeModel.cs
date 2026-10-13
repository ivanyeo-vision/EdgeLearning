using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Xml.Linq;
using ModelTraining.Services;
using OpenCvSharp;

namespace ModelTraining.ToolLibrary
{
    /// <summary>One taught component type: its golden vector and every place it sits on the board.</summary>
    public sealed class ComponentRecipe
    {
        public string Name { get; set; } = "";
        public int Width { get; set; }
        public int Height { get; set; }
        /// <summary>both = AI + geometric short/open, ai = AI only, geometry = short/open only (no golden vector).</summary>
        public string Mode { get; set; } = "both";
        public float[] GoldenVector { get; set; } = Array.Empty<float>();
        public List<OpenCvSharp.Rect> Instances { get; } = new List<OpenCvSharp.Rect>();
    }

    /// <summary>
    /// The contents of recipe.xml (written by TeachService.SaveAoi or by Main8.py).
    /// File names inside are relative to ExtractionDirectory.
    /// </summary>
    public sealed class RecipeModel
    {
        public string Backbone { get; private set; } = "";
        public string ModelFile { get; private set; } = "";
        public string ReferenceFile { get; private set; } = "";
        public string FiducialFile { get; private set; } = "";
        public int FiducialX { get; private set; }
        public int FiducialY { get; private set; }
        public OpenCvSharp.Rect SearchRegion { get; private set; }
        public float Threshold { get; private set; } = 0.80f;
        public DefectParams Defects { get; } = new DefectParams();
        public List<ComponentRecipe> Components { get; } = new List<ComponentRecipe>();

        /// <summary>Folder the .aoi was extracted to (the ONNX and the images are in here).</summary>
        public string ExtractionDirectory { get; internal set; } = "";

        /// <summary>Class names of the optional multi-class defect model (defect_classifier.onnx).</summary>
        public List<string> DefectClasses { get; } = new List<string>();

        /// <summary>
        /// Opens a .aoi package (a ZIP). It is extracted ONCE per package version into %TEMP%
        /// and reused afterwards, so the files are never deleted while an ONNX session has them open.
        /// </summary>
        public static RecipeModel LoadAoi(string aoiFilePath)
        {
            var info = new FileInfo(aoiFilePath);
            if (!info.Exists)
                throw new FileNotFoundException($"AOI package not found at: {aoiFilePath}");

            string key = $"{info.FullName.ToLowerInvariant()}|{info.Length}|{info.LastWriteTimeUtc.Ticks}";
            string hash = BitConverter.ToString(SHA256.HashData(Encoding.UTF8.GetBytes(key)))
                                      .Replace("-", "").Substring(0, 16);

            string root = Path.Combine(Path.GetTempPath(), "ModelTraining", "recipes");
            string folder = Path.Combine(root, hash);
            string marker = Path.Combine(folder, ".complete");

            if (!File.Exists(marker))
            {
                Directory.CreateDirectory(root);
                if (Directory.Exists(folder)) Directory.Delete(folder, true);

                string tmp = folder + ".tmp";
                if (Directory.Exists(tmp)) Directory.Delete(tmp, true);

                ZipFile.ExtractToDirectory(aoiFilePath, tmp);

                if (!File.Exists(Path.Combine(tmp, "recipe.xml")))
                {
                    Directory.Delete(tmp, true);
                    throw new FileNotFoundException("recipe.xml was not found inside the .aoi package.");
                }

                File.WriteAllText(Path.Combine(tmp, ".complete"), "ok");
                Directory.Move(tmp, folder);
                PurgeOldCache(root, folder);
            }

            RecipeModel model = Load(Path.Combine(folder, "recipe.xml"));
            model.ExtractionDirectory = folder;
            return model;
        }

        // Each package version needs ~120 MB: remove cache folders that have not been used for a week
        private static void PurgeOldCache(string root, string keepFolder)
        {
            try
            {
                foreach (string dir in Directory.GetDirectories(root))
                {
                    if (string.Equals(dir, keepFolder, StringComparison.OrdinalIgnoreCase)) continue;
                    if (Directory.GetLastWriteTimeUtc(dir) < DateTime.UtcNow.AddDays(-7))
                        Directory.Delete(dir, true);
                }
            }
            catch { /* best effort: a folder may still be in use */ }
        }

        /// <summary>Parses a raw recipe.xml.</summary>
        public static RecipeModel Load(string xmlPath)
        {
            XDocument doc = XDocument.Load(xmlPath);
            XElement root = doc.Root ?? throw new InvalidDataException("recipe.xml is empty.");
            var r = new RecipeModel();

            XElement model = Need(root, "Model");
            r.Backbone = (string?)model.Attribute("backbone") ?? "";
            r.ModelFile = Text(model, "file");

            r.ReferenceFile = Text(Need(root, "Reference"), "image");

            XElement fid = Need(root, "Fiducial");
            r.FiducialX = Int(fid, "x");
            r.FiducialY = Int(fid, "y");
            r.FiducialFile = Text(fid, "template");

            XElement sr = Need(root, "SearchRegion");
            r.SearchRegion = new OpenCvSharp.Rect(Int(sr, "x"), Int(sr, "y"), Int(sr, "width"), Int(sr, "height"));

            XElement insp = Need(root, "Inspection");
            r.Threshold = (float)Dbl(insp, "threshold");

            XElement? dp = root.Element("DefectParams");
            if (dp != null)
            {
                DefectParams d = r.Defects;
                d.ContextMargin = (int?)dp.Attribute("contextMargin") ?? d.ContextMargin;
                d.AlignSearch = (int?)dp.Attribute("alignSearch") ?? d.AlignSearch;
                d.MinBlobArea = (int?)dp.Attribute("minBlobArea") ?? d.MinBlobArea;
                d.HoleMinArea = (int?)dp.Attribute("holeMinArea") ?? d.HoleMinArea;
                d.HoleFillRatio = (double?)dp.Attribute("holeFillRatio") ?? d.HoleFillRatio;
                d.BiteMaxFraction = (double?)dp.Attribute("biteMaxFraction") ?? d.BiteMaxFraction;
                d.HybridMinArea = (int?)dp.Attribute("hybridMinArea") ?? d.HybridMinArea;
            }

            XElement? classes = root.Element("DefectClasses");
            if (classes != null)
                foreach (XElement dc in classes.Elements("Class"))
                    r.DefectClasses.Add(Text(dc, "name"));

            foreach (XElement ce in Need(root, "Components").Elements("Component"))
            {
                var c = new ComponentRecipe
                {
                    Name = Text(ce, "name"),
                    Width = Int(ce, "width"),
                    Height = Int(ce, "height"),
                    Mode = ((string?)ce.Attribute("mode") ?? "both").Trim().ToLowerInvariant()
                };
                if (c.Mode != "both" && c.Mode != "ai" && c.Mode != "geometry")
                    throw new InvalidDataException($"recipe.xml: component '{c.Name}' has an unknown mode '{c.Mode}'.");

                // Geometry-only components (trace areas) have no golden vector
                XElement? gv = ce.Element("GoldenVector");
                if (gv != null && !string.IsNullOrWhiteSpace(gv.Value))
                {
                    c.GoldenVector = gv.Value
                        .Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
                        .Select(s => float.Parse(s, CultureInfo.InvariantCulture))
                        .ToArray();
                }
                if (c.Mode != "geometry" && c.GoldenVector.Length == 0)
                    throw new InvalidDataException($"recipe.xml: component '{c.Name}' (mode {c.Mode}) has no GoldenVector.");

                foreach (XElement i in Need(ce, "Instances").Elements("Instance"))
                    c.Instances.Add(new OpenCvSharp.Rect(Int(i, "x"), Int(i, "y"), Int(i, "width"), Int(i, "height")));

                r.Components.Add(c);
            }

            if (r.Components.Count == 0)
                throw new InvalidDataException("recipe.xml contains no components.");

            return r;
        }

        // ---- small XML helpers with readable error messages ----
        private static XElement Need(XElement parent, string name) =>
            parent.Element(name) ?? throw new InvalidDataException($"recipe.xml: missing <{name}> in <{parent.Name}>.");

        private static string Text(XElement e, string attr) =>
            (string?)e.Attribute(attr) ?? throw new InvalidDataException($"recipe.xml: <{e.Name}> is missing '{attr}'.");

        private static int Int(XElement e, string attr) =>
            (int?)e.Attribute(attr) ?? throw new InvalidDataException($"recipe.xml: <{e.Name}> is missing '{attr}'.");

        private static double Dbl(XElement e, string attr) =>
            (double?)e.Attribute(attr) ?? throw new InvalidDataException($"recipe.xml: <{e.Name}> is missing '{attr}'.");
    }
}