using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using ModelTraining.Commands;
using ModelTraining.Services;

namespace ModelTraining.ViewModels
{
    /// <summary>
    /// Drives the Teach window = the whole Main8.py / MainView8.py workflow:
    ///   1 Load reference  2 Fiducial  3 Search region (once)  4 Add components (repeat)
    ///   5 Extract PASS    6 Instant Teach + save .aoi
    /// </summary>
    public class TeachViewModel : INotifyPropertyChanged
    {
        private enum DrawTarget { None, Fiducial, Search, Component }

        private static readonly string[] Palette =
            { "Orange", "DeepSkyBlue", "Magenta", "Gold", "LimeGreen", "HotPink" };

        private readonly TeachService _teach = new TeachService();
        private DrawTarget _target = DrawTarget.None;

        private BitmapSource? _referenceImage;
        private bool _isDrawMode;
        private bool _isBusy;
        private bool _isTaught;
        private string _hint = "";
        private string _status = "Step 1: load the reference (golden) board image.";
        private string _newComponentName = "PAD";
        private string _backboneName = "mobilenetv4_large";
        private string _backboneFile = "backbone.onnx";

        // --- bound to the UI ---
        public BitmapSource? ReferenceImage { get => _referenceImage; private set { _referenceImage = value; OnPropertyChanged(); } }
        public ObservableCollection<InspectionRegion> Overlay { get; } = new ObservableCollection<InspectionRegion>();

        public bool IsDrawMode { get => _isDrawMode; private set { _isDrawMode = value; OnPropertyChanged(); } }
        public bool IsBusy { get => _isBusy; private set { _isBusy = value; OnPropertyChanged(); RaiseFlags(); } }
        public bool IsTaught { get => _isTaught; private set { _isTaught = value; OnPropertyChanged(); } }
        public string Hint { get => _hint; private set { _hint = value; OnPropertyChanged(); } }
        public string Status { get => _status; private set { _status = value; OnPropertyChanged(); } }

        public string NewComponentName { get => _newComponentName; set { _newComponentName = value; OnPropertyChanged(); } }
        /// <summary>Name stored in recipe.xml (information only).</summary>
        public string BackboneName { get => _backboneName; set { _backboneName = value; OnPropertyChanged(); } }
        /// <summary>Backbone ONNX in the Assets folder, exported once from Python (export_backbone_onnx).</summary>
        public string BackboneFile { get => _backboneFile; set { _backboneFile = value; OnPropertyChanged(); } }

        public bool HasReference => _teach.HasReference;
        public bool HasFiducial => _teach.HasFiducial;
        public bool HasSearch => _teach.HasSearchRegion;
        public bool HasComponents => _teach.HasComponents;
        public bool HasPassCrops => _teach.HasPassCrops;

        public bool CanSetFiducial => HasReference && !IsBusy;
        public bool CanDrawSearch => HasFiducial && !IsBusy;
        public bool CanAddComponent => HasSearch && !IsBusy;
        public bool CanExtract => HasComponents && !IsBusy;
        public bool CanTeach => HasPassCrops && !IsBusy;

        public string ComponentSummary
        {
            get
            {
                var counts = _teach.ComponentCounts();
                return counts.Count == 0 ? "No components taught yet."
                    : "Components: " + string.Join(", ", counts.Select(kv => $"{kv.Key} x{kv.Value}"));
            }
        }

        // --- commands ---
        public ICommand LoadReferenceCommand { get; }
        public ICommand SetFiducialCommand { get; }
        public ICommand DrawSearchCommand { get; }
        public ICommand AddComponentCommand { get; }
        public ICommand ExtractPassCommand { get; }
        public ICommand InstantTeachCommand { get; }

        public TeachViewModel()
        {
            LoadReferenceCommand = new RelayCommand(_ => LoadReference());
            SetFiducialCommand = new RelayCommand(_ => BeginDraw(DrawTarget.Fiducial,
                "Drag a box around a unique, high-contrast feature (the fiducial anchor)."));
            DrawSearchCommand = new RelayCommand(_ => BeginDraw(DrawTarget.Search,
                "Drag a box around the area that contains ALL the components. This is done only once."));
            AddComponentCommand = new RelayCommand(_ => AddComponent());
            ExtractPassCommand = new RelayCommand(_ => ExtractPass());
            InstantTeachCommand = new RelayCommand(_ => InstantTeach());
        }

        // ------------------------------------------------------------------ steps
        private void LoadReference()
        {
            var dlg = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "Images|*.jpg;*.jpeg;*.png;*.bmp",
                Title = "Select the reference (golden) board image"
            };
            if (dlg.ShowDialog() != true) return;

            if (!_teach.LoadReference(dlg.FileName))
            {
                Status = "Could not read that image.";
                return;
            }

            ReferenceImage = ImageConverter.MatToBitmapSource(_teach.Reference!);
            IsTaught = false;
            CancelDraw();
            RefreshOverlay();
            Status = "Reference loaded. Step 2: set the fiducial anchor.";
            RaiseFlags();
        }

        private void AddComponent()
        {
            if (string.IsNullOrWhiteSpace(NewComponentName))
            {
                Status = "Type a component name first (for example PAD).";
                return;
            }
            BeginDraw(DrawTarget.Component, $"Drag a box around ONE '{NewComponentName.Trim()}' (the component to teach).");
        }

        private void BeginDraw(DrawTarget target, string hint)
        {
            _target = target;
            Hint = hint;
            IsDrawMode = true;
        }

        private void CancelDraw()
        {
            _target = DrawTarget.None;
            Hint = "";
            IsDrawMode = false;
        }

        /// <summary>Called by the window when the user finishes drawing a box (image pixel coordinates).</summary>
        public void OnRegionDrawn(Rect r)
        {
            if (!IsDrawMode) return;

            var roi = new OpenCvSharp.Rect((int)Math.Round(r.X), (int)Math.Round(r.Y),
                                           (int)Math.Round(r.Width), (int)Math.Round(r.Height));
            try
            {
                switch (_target)
                {
                    case DrawTarget.Fiducial:
                        _teach.SetFiducial(roi);
                        Status = "Fiducial set. Step 3: draw the search region (once).";
                        break;

                    case DrawTarget.Search:
                        _teach.SetSearchRegion(roi);
                        Status = "Search region set. Step 4: name a component and add it. Repeat for each type.";
                        break;

                    case DrawTarget.Component:
                        var counts = _teach.AddComponent(NewComponentName, roi);
                        Status = "Component added. Found: " +
                                 string.Join(", ", counts.Select(kv => $"{kv.Key} x{kv.Value}"));
                        break;
                }
                IsTaught = false;
            }
            catch (Exception ex)
            {
                Status = ex.Message;
            }

            CancelDraw();
            RefreshOverlay();
            RaiseFlags();
        }

        private async void ExtractPass()
        {
            var dlg = new Microsoft.Win32.OpenFileDialog
            {
                Filter = "Images|*.jpg;*.jpeg;*.png;*.bmp",
                Title = "Select PASS board images",
                Multiselect = true
            };
            if (dlg.ShowDialog() != true) return;

            string[] files = dlg.FileNames;
            IsBusy = true;
            Status = "Extracting PASS crops...";
            try
            {
                int n = await Task.Run(() => _teach.ExtractPassCrops(files));
                Status = $"Extracted {n} PASS crops from {files.Length} image(s). Step 6: Instant Teach.";
            }
            catch (Exception ex)
            {
                Status = "Extract failed: " + ex.Message;
            }
            finally
            {
                IsBusy = false;
            }
        }

        private async void InstantTeach()
        {
            string? backbonePath = AssetLocator.Find(BackboneFile);
            if (backbonePath == null)
            {
                Status = $"'{BackboneFile}' not found in the Assets folder. Export it once from Python: " +
                         "EdgeLearningEngine(model).export_backbone_onnx('backbone.onnx') and copy backbone.onnx " +
                         "and backbone.onnx.data to Assets.";
                return;
            }

            var save = new Microsoft.Win32.SaveFileDialog
            {
                Filter = "AOI recipe package|*.aoi",
                FileName = "board_recipe.aoi",
                Title = "Save the recipe package"
            };
            if (save.ShowDialog() != true) return;

            string destination = save.FileName;
            string backboneName = BackboneName;
            IsBusy = true;
            Status = "Teaching: extracting golden vectors and writing the package...";
            try
            {
                string result = await Task.Run(() =>
                {
                    using var backbone = new OnnxInferenceService(backbonePath);
                    var (ok, message, _) = _teach.InstantTeach(backbone);
                    if (!ok) throw new InvalidOperationException(message);
                    _teach.SaveAoi(destination, backbonePath, backboneName);
                    return $"Instant Teach complete: {message}. Saved {destination}  (use Load Recipe in the main window)";
                });
                Status = result;
                IsTaught = true;
            }
            catch (Exception ex)
            {
                Status = "Teach failed: " + ex.Message;
            }
            finally
            {
                IsBusy = false;
            }
        }

        // ------------------------------------------------------------------ overlay
        private void RefreshOverlay()
        {
            Overlay.Clear();

            if (_teach.FiducialRect is OpenCvSharp.Rect f)
                Overlay.Add(Box(f, "Cyan", "Fiducial"));

            if (_teach.SearchRect is OpenCvSharp.Rect s)
                Overlay.Add(Box(s, "White", "Search region"));

            int ci = 0;
            foreach (var comp in _teach.Components)
            {
                string color = Palette[ci++ % Palette.Length];
                bool first = true;
                foreach (var inst in comp.Instances)
                {
                    Overlay.Add(Box(inst, color, first ? comp.Name : ""));
                    first = false;
                }
            }

            OnPropertyChanged(nameof(ComponentSummary));
        }

        private static InspectionRegion Box(OpenCvSharp.Rect r, string color, string caption) => new InspectionRegion
        {
            X = r.X,
            Y = r.Y,
            Width = r.Width,
            Height = r.Height,
            BoxColor = color,
            Caption = caption
        };

        private void RaiseFlags()
        {
            OnPropertyChanged(nameof(HasReference));
            OnPropertyChanged(nameof(HasFiducial));
            OnPropertyChanged(nameof(HasSearch));
            OnPropertyChanged(nameof(HasComponents));
            OnPropertyChanged(nameof(HasPassCrops));
            OnPropertyChanged(nameof(CanSetFiducial));
            OnPropertyChanged(nameof(CanDrawSearch));
            OnPropertyChanged(nameof(CanAddComponent));
            OnPropertyChanged(nameof(CanExtract));
            OnPropertyChanged(nameof(CanTeach));
            OnPropertyChanged(nameof(ComponentSummary));
        }

        public event PropertyChangedEventHandler? PropertyChanged;
        protected void OnPropertyChanged([CallerMemberName] string name = null!) =>
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}