using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using ModelTraining.Commands;
using ModelTraining.Services;
using ModelTraining.ToolLibrary;
using OpenCvSharp;

namespace ModelTraining.ViewModels
{
    public class MainViewModel : INotifyPropertyChanged
    {
        private const float ClassifyBelow = 0.97f;      // pins scoring below this get a defect label ready for the slider
        private const double MinAlignScore = 0.5;       // below this the fiducial was not found

        // --- Backing Fields ---
        private OnnxInferenceService? _inferenceService;     // backbone: crop -> embedding
        private OnnxClassifierService? _classifierService;   // optional trained multi-class defect model
        private RoiTool? _roiTool;
        private RecipeModel? _recipe;
        private Mat? _referenceImage;

        private BitmapSource? _currentDisplayImage;
        private string _inspectionStatus = "IDLE";
        private string _statusColor = "#FF555555";
        private float _similarityScore = 0.0f;
        private long _processTimeMs = 0;
        private long _replyTimeMs = 0;
        private int _queueCount = 0;

        private double _threshold = 0.80;
        private int _failCount = 0;
        private int _pinCount = 0;
        private string _defectSummary = "";
        private double _alignScore = 1.0;
        private bool _alignFailed = false;

        // Scores of the most recent board. The slider re-labels these without re-running the model.
        private List<InspectionRegion> _lastRegions = new List<InspectionRegion>();

        // --- UI Properties ---
        public ObservableCollection<InspectionRegion> PinRegions { get; } = new ObservableCollection<InspectionRegion>();

        public BitmapSource? CurrentDisplayImage
        {
            get => _currentDisplayImage;
            set { _currentDisplayImage = value; OnPropertyChanged(); }
        }
        public string InspectionStatus
        {
            get => _inspectionStatus;
            set { _inspectionStatus = value; OnPropertyChanged(); }
        }
        public string StatusColor
        {
            get => _statusColor;
            set { _statusColor = value; OnPropertyChanged(); }
        }
        public float SimilarityScore
        {
            get => _similarityScore;
            set { _similarityScore = value; OnPropertyChanged(); }
        }
        public long ProcessTimeMs
        {
            get => _processTimeMs;
            set { _processTimeMs = value; OnPropertyChanged(); }
        }
        public long ReplyTimeMs
        {
            get => _replyTimeMs;
            set { _replyTimeMs = value; OnPropertyChanged(); }
        }
        public int QueueCount
        {
            get => _queueCount;
            set { _queueCount = value; OnPropertyChanged(); }
        }
        public int FailCount
        {
            get => _failCount;
            set { _failCount = value; OnPropertyChanged(); }
        }
        public int PinCount
        {
            get => _pinCount;
            set { _pinCount = value; OnPropertyChanged(); }
        }
        /// <summary>e.g. "Short x1, Mouse bite x2"</summary>
        public string DefectSummary
        {
            get => _defectSummary;
            set { _defectSummary = value; OnPropertyChanged(); }
        }
        /// <summary>Fiducial match score of the last board (1.0 = perfect).</summary>
        public double AlignScore
        {
            get => _alignScore;
            set { _alignScore = value; OnPropertyChanged(); }
        }

        /// <summary>AI PASS/NG threshold bound to the slider. Changing it re-labels the last board instantly.
        /// (The geometric short/open verdicts do not depend on it.)</summary>
        public double Threshold
        {
            get => _threshold;
            set
            {
                double v = Math.Round(Math.Max(0.0, Math.Min(1.0, value)), 2);
                if (Math.Abs(v - _threshold) < 1e-9) return;
                _threshold = v;
                OnPropertyChanged();
                ReapplyThreshold();
            }
        }

        // --- Commands ---
        public ICommand StartInspectionCommand { get; }
        public ICommand LoadSingleImageCommand { get; }
        public ICommand LoadBatchCommand { get; }
        public ICommand ExportCsvCommand { get; }
        public ICommand LoadRecipeCommand { get; }

        public MainViewModel()
        {
            StartInspectionCommand = new RelayCommand(ExecuteStartInspection);
            LoadSingleImageCommand = new RelayCommand(ExecuteLoadSingleImage);
            LoadBatchCommand = new RelayCommand(ExecuteLoadBatch);
            ExportCsvCommand = new RelayCommand(ExecuteExportCsv);
            LoadRecipeCommand = new RelayCommand(ExecuteLoadRecipe);

            if (DesignerProperties.GetIsInDesignMode(new DependencyObject()))
            {
                return;
            }

            try
            {
                string? aoi = AssetLocator.Find("board_recipe.aoi");
                if (aoi != null) LoadRecipe(aoi);
                else InspectionStatus = "NO RECIPE";      // teach in Python, copy board_recipe.aoi to Assets or use Load Recipe
            }
            catch (Exception ex)
            {
                InspectionStatus = "LOAD ERROR";
                StatusColor = "#FFDC3545";
                MessageBox.Show(ex.Message, "Failed to load model / recipe",
                                MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        // --- Recipe loading ---
        private void LoadRecipe(string aoiPath)
        {
            RecipeModel recipe = RecipeModel.LoadAoi(aoiPath);

            // The similarity model is the BACKBONE named in recipe.xml (not the defect classifier)
            string backbonePath = Path.Combine(recipe.ExtractionDirectory, recipe.ModelFile);
            var inference = new OnnxInferenceService(backbonePath);

            RoiTool? roi = null;
            Mat? reference = null;
            OnnxClassifierService? classifier = null;
            try
            {
                roi = new RoiTool(recipe);

                reference = Cv2.ImRead(Path.Combine(recipe.ExtractionDirectory, recipe.ReferenceFile), ImreadModes.Color);
                if (reference.Empty())
                    throw new FileNotFoundException("Could not read the reference image from the recipe.");

                // Optional "Method 2": a trained multi-class defect model shipped inside the package
                string classifierPath = Path.Combine(recipe.ExtractionDirectory, "defect_classifier.onnx");
                if (File.Exists(classifierPath) && recipe.DefectClasses.Count > 0)
                    classifier = new OnnxClassifierService(classifierPath, recipe.DefectClasses);
            }
            catch
            {
                inference.Dispose();
                roi?.Dispose();
                reference?.Dispose();
                classifier?.Dispose();
                throw;
            }

            _inferenceService?.Dispose();
            _classifierService?.Dispose();
            _roiTool?.Dispose();
            _referenceImage?.Dispose();

            _inferenceService = inference;
            _classifierService = classifier;
            _roiTool = roi;
            _referenceImage = reference;
            _recipe = recipe;

            // Reset the display and start the slider at the threshold taught with the recipe
            _lastRegions = new List<InspectionRegion>();
            PinRegions.Clear();
            PinCount = 0;
            FailCount = 0;
            DefectSummary = "";
            InspectionStatus = "IDLE";
            StatusColor = "#FF555555";

            _threshold = Math.Round(recipe.Threshold, 2);
            OnPropertyChanged(nameof(Threshold));

            Debug.WriteLine($"Recipe loaded: {Path.GetFileName(aoiPath)} | backbone {recipe.Backbone} | " +
                            $"{recipe.Components.Count} component type(s) | provider {inference.Provider} | " +
                            $"classifier {(classifier != null ? "yes" : "no (geometric)")}");
        }

        private void ExecuteLoadRecipe(object obj)
        {
            var dialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "AOI recipe package|*.aoi",
                Title = "Select a recipe package"
            };
            if (dialog.ShowDialog() != true) return;

            try
            {
                LoadRecipe(dialog.FileName);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Failed to load recipe package",
                                MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        // --- Core inspection (same for single and batch) ---
        // HYBRID: the AI similarity model (MobileNetV4) cannot see thin traces, so
        //   1. AI similarity                          (modes both / ai)        -> PASS / NG
        //   2. geometric short / open check on EVERY pin (modes both / geometry) -> fails the pin even if the AI passed
        //   3. AI-NG pins also get the full defect name (Missing hole / Mouse bite / ...)
        private List<InspectionRegion> InspectBoard(Mat mat)
        {
            List<MappedComponent> mapped = _roiTool!.GetMappedComponents(mat);
            var crops = new List<Mat>(mapped.Count);
            try
            {
                foreach (MappedComponent m in mapped)
                    crops.Add(new Mat(mat, m.Region));                       // color BGR crops (views, no copy)

                // 1. AI: one batched model call for every pin that uses the AI
                var aiIndex = new List<int>();
                var aiCrops = new List<Mat>();
                for (int i = 0; i < mapped.Count; i++)
                {
                    if (mapped[i].Mode == "geometry") continue;
                    aiIndex.Add(i);
                    aiCrops.Add(crops[i]);
                }

                float[] scores = Enumerable.Repeat(1f, mapped.Count).ToArray();   // geometry-only pins have no AI score
                if (aiCrops.Count > 0)
                {
                    List<float[]> vectors = _inferenceService!.ExtractFeatureVectors(aiCrops);
                    for (int k = 0; k < aiIndex.Count; k++)
                        scores[aiIndex[k]] = _inferenceService.GetCosineSimilarity(vectors[k], mapped[aiIndex[k]].GoldenVector);
                }

                // 2 + 3. geometry and defect names
                var regions = new List<InspectionRegion>(mapped.Count);
                for (int i = 0; i < mapped.Count; i++)
                {
                    MappedComponent m = mapped[i];
                    bool usesAi = m.Mode != "geometry";

                    string? geo = m.Mode != "ai" ? DetectShortOpen(mat, m) : null;
                    string? defect = usesAi && scores[i] < ClassifyBelow ? ClassifyDefect(mat, crops[i], m) : null;

                    regions.Add(new InspectionRegion
                    {
                        X = m.Region.X,
                        Y = m.Region.Y,
                        Width = m.Region.Width,
                        Height = m.Region.Height,
                        Score = scores[i],
                        UsesAi = usesAi,
                        Component = m.Name,
                        GeoLabel = geo,
                        DefectLabel = defect
                    });
                }
                return regions;
            }
            finally
            {
                foreach (Mat c in crops) c.Dispose();
            }
        }

        // Hybrid stage: geometric short / open check, independent of the AI score and of the slider
        private string? DetectShortOpen(Mat live, MappedComponent m)
        {
            try
            {
                return DefectClassifier.DetectShortOpen(_referenceImage!, live, m.RefBox,
                                                        _roiTool!.LastDx, _roiTool.LastDy, _recipe!.Defects, out _);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Short/open check failed: {ex.Message}");
                return null;
            }
        }

        // Stage 2: which defect is it?  Trained model if the package has one, geometric analysis otherwise.
        private string? ClassifyDefect(Mat live, Mat crop, MappedComponent m)
        {
            try
            {
                if (_classifierService != null)
                    return _classifierService.ClassifyDefect(crop, out _);

                return DefectClassifier.ClassifyPin(_referenceImage!, live, m.RefBox,
                                                    _roiTool!.LastDx, _roiTool.LastDy, _recipe!.Defects);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Defect classification failed: {ex.Message}");
                return null;
            }
        }

        private static string ColorFor(string label) => label switch
        {
            "PASS" => "LimeGreen",
            "Missing hole" => "Orange",
            "Short" => "Red",
            "Open circuit" => "Magenta",
            "Mouse bite" or "Chipping" => "Gold",
            "Extra copper" => "DeepSkyBlue",
            "Missing" => "Coral",
            _ => "Red" // Any custom ONNX defect class from recipe.xml
        };

        // Applies the current Threshold to a scored board and updates all UI state (UI thread only).
        private void Publish(List<InspectionRegion> regions)
        {
            int fails = 0;
            float lowest = 1.0f;

            foreach (var r in regions)
            {
                if (!string.IsNullOrEmpty(r.GeoLabel))
                {
                    // geometric short / open: fails the pin whatever the AI score and the slider say
                    r.Label = r.GeoLabel!;
                    r.Caption = r.UsesAi ? $"{r.GeoLabel} (geo) {r.Score:F2}" : $"{r.GeoLabel} (geo)";
                    fails++;
                }
                else if (r.UsesAi && r.Score < Threshold)
                {
                    string defect = string.IsNullOrEmpty(r.DefectLabel) || r.DefectLabel == "PASS" ? "NG" : r.DefectLabel!;
                    r.Label = defect;
                    r.Caption = $"{defect} {r.Score:F2}";
                    fails++;
                }
                else
                {
                    r.Label = "PASS";
                    r.Caption = "";
                }

                r.BoxColor = ColorFor(r.Label);
                if (r.UsesAi && r.Score < lowest) lowest = r.Score;
            }

            PinRegions.Clear();
            foreach (var r in regions) PinRegions.Add(r);

            PinCount = regions.Count;
            FailCount = fails;
            SimilarityScore = lowest;
            DefectSummary = string.Join(", ", regions.Where(r => r.Label != "PASS")
                                                     .GroupBy(r => r.Label)
                                                     .Select(g => $"{g.Key} x{g.Count()}"));

            if (_alignFailed)
            {
                InspectionStatus = "ALIGN FAIL";             // fiducial not found: results are meaningless
                StatusColor = "#FFFD7E14";
                return;
            }

            // No pins found means alignment/recipe is wrong: never report PASS
            bool boardPass = regions.Count > 0 && fails == 0;
            InspectionStatus = boardPass ? "PASS" : "NG";
            StatusColor = boardPass ? "#FF28A745" : "#FFDC3545";
        }

        private void ApplyResult(BitmapSource image, List<InspectionRegion> regions, long elapsedMs, double alignScore)
        {
            CurrentDisplayImage = image;
            ProcessTimeMs = elapsedMs;
            AlignScore = alignScore;
            _alignFailed = alignScore < MinAlignScore;
            _lastRegions = regions;
            Publish(regions);
        }

        // Called by the slider
        private void ReapplyThreshold()
        {
            if (_lastRegions.Count == 0) return;
            Publish(_lastRegions);
        }

        // --- Command Execution Methods ---
        private void ExecuteLoadSingleImage(object obj)
        {
            if (_inferenceService == null || _roiTool == null) return;

            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp",
                Title = "Select a Test Image"
            };
            if (openFileDialog.ShowDialog() != true) return;

            using Mat mat = Cv2.ImRead(openFileDialog.FileName, ImreadModes.Color);
            if (mat.Empty()) return;

            try
            {
                var stopwatch = Stopwatch.StartNew();
                var regions = InspectBoard(mat);
                stopwatch.Stop();

                Debug.WriteLine($"Alignment dx={_roiTool.LastDx}, dy={_roiTool.LastDy}, score={_roiTool.AlignScore:F2}, pins={regions.Count}");
                ApplyResult(ImageConverter.MatToBitmapSource(mat), regions, stopwatch.ElapsedMilliseconds, _roiTool.AlignScore);
            }
            catch (Exception ex)
            {
                InspectionStatus = "ERROR";
                StatusColor = "#FFDC3545";
                MessageBox.Show(ex.Message, "Inspection failed", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private async void ExecuteLoadBatch(object obj)
        {
            if (_inferenceService == null || _roiTool == null) return;

            var openFileDialog = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp",
                Title = "Select a Batch of Test Images",
                Multiselect = true
            };
            if (openFileDialog.ShowDialog() != true) return;

            string[] files = openFileDialog.FileNames;
            string? error = null;

            await Task.Run(async () =>
            {
                try
                {
                    foreach (var file in files)
                    {
                        using Mat mat = Cv2.ImRead(file, ImreadModes.Color);
                        if (mat.Empty()) continue;

                        var stopwatch = Stopwatch.StartNew();
                        var regions = InspectBoard(mat);
                        stopwatch.Stop();

                        double align = _roiTool.AlignScore;
                        BitmapSource image = ImageConverter.MatToBitmapSource(mat);   // frozen, safe across threads
                        long elapsed = stopwatch.ElapsedMilliseconds;

                        Application.Current.Dispatcher.Invoke(() => ApplyResult(image, regions, elapsed, align));
                        await Task.Delay(500);
                    }
                }
                catch (Exception ex)
                {
                    error = ex.Message;
                }
            });

            if (error != null)
            {
                InspectionStatus = "ERROR";
                StatusColor = "#FFDC3545";
                MessageBox.Show(error, "Batch inspection failed", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            InspectionStatus = "BATCH DONE";
            StatusColor = "#FF6C757D";
        }

        private void ExecuteStartInspection(object obj)
        {
            InspectionStatus = "INSPECTING";
            StatusColor = "#FF007ACC";
        }

        private void ExecuteExportCsv(object obj)
        {
            if (PinRegions.Count == 0)
            {
                MessageBox.Show("No inspection results to export.", "Export CSV", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var dialog = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "CSV File (*.csv)|*.csv",
                Title = "Export Inspection Results",
                FileName = $"AOI_Report_{DateTime.Now:yyyyMMdd_HHmmss}.csv",
                InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments)
            };

            if (dialog.ShowDialog() != true) return;

            var lines = new List<string>
    {
        // CSV Header: Matches the live table + adds metadata for deep analysis
        "Timestamp,Board_Status,Threshold_Used,Component_Type,Verdict,AI_Score,Uses_AI,Geo_Defect,Box_X,Box_Y,Box_W,Box_H"
    };

            string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");

            // Write a row for every region currently in the table
            foreach (var r in PinRegions)
            {
                // Format the AI score. If it's a geometry-only pin, leave AI score empty for cleaner Excel filtering.
                string scoreStr = r.UsesAi ? r.Score.ToString("F4") : "N/A";
                string geoDefect = string.IsNullOrEmpty(r.GeoLabel) ? "None" : r.GeoLabel;

                // Construct the row
                string row = $"{timestamp}," +
                             $"{InspectionStatus}," +      // Overall board PASS/NG/ALIGN FAIL
                             $"{Threshold:F2}," +          // The slider position at time of export
                             $"{r.Component}," +           // Live Table: Region / Type
                             $"{r.Label}," +               // Live Table: Defect / Verdict
                             $"{scoreStr}," +              // Live Table: AI Score
                             $"{r.UsesAi}," +              // True/False
                             $"{geoDefect}," +             // Did circuit geometry fail it?
                             $"{r.X:F0},{r.Y:F0},{r.Width:F0},{r.Height:F0}"; // Bounding box

                lines.Add(row);
            }

            try
            {
                File.WriteAllLines(dialog.FileName, lines);
                MessageBox.Show($"Successfully exported {PinRegions.Count} regions to:\n{dialog.FileName}",
                                "Export Complete", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Failed to save CSV:\n{ex.Message}", "Export Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

       
        public event PropertyChangedEventHandler? PropertyChanged;
        protected void OnPropertyChanged([CallerMemberName] string name = null!)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }

    public class InspectionRegion
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Width { get; set; }
        public double Height { get; set; }
        public string BoxColor { get; set; } = "LimeGreen";
        public float Score { get; set; }

        
        public string Component { get; set; } = "";
       
        public string Label { get; set; } = "";
       
        public bool UsesAi { get; set; } = true;
        
        public string? GeoLabel { get; set; }
       
        public string? DefectLabel { get; set; }
      
        public string Caption { get; set; } = "";
    }
}