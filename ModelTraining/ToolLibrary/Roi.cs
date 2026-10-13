using System;
using System.Collections.Generic;
using System.IO;
using OpenCvSharp;

namespace ModelTraining.ToolLibrary
{
    /// <summary>One place on the board where a taught component is expected.</summary>
    public class MappedComponent
    {
        public string Name { get; set; } = "";
        /// <summary>both | ai | geometry (see ComponentRecipe.Mode).</summary>
        public string Mode { get; set; } = "both";
        public float[] GoldenVector { get; set; } = Array.Empty<float>();
        /// <summary>Box on the live image after fiducial alignment.</summary>
        public OpenCvSharp.Rect Region { get; set; }
        /// <summary>Box on the REFERENCE board (used to cut the golden crop for defect classification).</summary>
        public OpenCvSharp.Rect RefBox { get; set; }
    }

    /// <summary>Fiducial alignment + fixed component locations, all read from the recipe.</summary>
    public class RoiTool : IDisposable
    {
        private readonly RecipeModel _activeRecipe;
        private readonly Mat _fiducial;
        private readonly OpenCvSharp.Point _fidPos;

        public float Threshold { get; }
        public int LastDx { get; private set; }
        public int LastDy { get; private set; }

        /// <summary>Fiducial match score of the last image (1.0 = perfect). Low = the board was not found.</summary>
        public double AlignScore { get; private set; }

        public RoiTool(RecipeModel recipe)
        {
            _activeRecipe = recipe;
            _fidPos = new OpenCvSharp.Point(recipe.FiducialX, recipe.FiducialY);
            Threshold = recipe.Threshold;

            string fiducialPath = Path.Combine(recipe.ExtractionDirectory, recipe.FiducialFile);
            _fiducial = Cv2.ImRead(fiducialPath, ImreadModes.Grayscale);
            if (_fiducial.Empty()) throw new FileNotFoundException("Could not read fiducial.", fiducialPath);
        }

        public List<MappedComponent> GetMappedComponents(Mat fullImage)
        {
            using var gray = new Mat();
            if (fullImage.Channels() == 1) fullImage.CopyTo(gray);
            else Cv2.CvtColor(fullImage, gray, ColorConversionCodes.BGR2GRAY);

            using var res = new Mat();
            Cv2.MatchTemplate(gray, _fiducial, res, TemplateMatchModes.CCoeffNormed);
            Cv2.MinMaxLoc(res, out _, out double maxVal, out _, out OpenCvSharp.Point maxLoc);

            LastDx = maxLoc.X - _fidPos.X;
            LastDy = maxLoc.Y - _fidPos.Y;
            AlignScore = maxVal;

            var results = new List<MappedComponent>();
            foreach (var comp in _activeRecipe.Components)
            {
                foreach (var p in comp.Instances)
                {
                    var r = new OpenCvSharp.Rect(p.X + LastDx, p.Y + LastDy, p.Width, p.Height);
                    bool inside = r.X >= 0 && r.Y >= 0 && r.Right <= fullImage.Width && r.Bottom <= fullImage.Height;
                    if (!inside) continue;

                    results.Add(new MappedComponent
                    {
                        Name = comp.Name,
                        Mode = comp.Mode,
                        GoldenVector = comp.GoldenVector,
                        Region = r,
                        RefBox = p
                    });
                }
            }
            return results;
        }

        public void Dispose()
        {
            if (_fiducial != null && !_fiducial.IsDisposed) _fiducial.Dispose();
        }
    }
}