using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Globalization;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using ModelTraining.ViewModels;

namespace ModelTraining.ToolLibrary
{
    /// <summary>
    /// Replacement for Cognex CogDisplay.
    /// Draws a BitmapSource fitted to the control, then every InspectionRegion on top in IMAGE pixel
    /// coordinates, so the boxes always line up with the image.
    ///   Mouse wheel          : zoom around the cursor
    ///   Left-drag            : pan          (right/middle drag pans while IsDrawMode is on)
    ///   Double-click         : fit image to window
    ///   IsDrawMode = true    : left-drag draws a box and raises RegionDrawn (image pixel coordinates)
    /// A region with a Caption always shows it (defect names, component names); other regions show the
    /// score when they are big enough on screen.
    /// </summary>
    public class InspectionDisplay : FrameworkElement
    {
        // ---------- Dependency properties ----------
        public static readonly DependencyProperty SourceProperty =
            DependencyProperty.Register(nameof(Source), typeof(BitmapSource), typeof(InspectionDisplay),
                new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender, OnSourceChanged));

        public static readonly DependencyProperty RegionsProperty =
            DependencyProperty.Register(nameof(Regions), typeof(IEnumerable<InspectionRegion>), typeof(InspectionDisplay),
                new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender, OnRegionsChanged));

        public static readonly DependencyProperty ShowScoresProperty =
            DependencyProperty.Register(nameof(ShowScores), typeof(bool), typeof(InspectionDisplay),
                new FrameworkPropertyMetadata(true, FrameworkPropertyMetadataOptions.AffectsRender));

        public static readonly DependencyProperty IsDrawModeProperty =
            DependencyProperty.Register(nameof(IsDrawMode), typeof(bool), typeof(InspectionDisplay),
                new FrameworkPropertyMetadata(false, OnDrawModeChanged));

        public BitmapSource? Source
        {
            get => (BitmapSource?)GetValue(SourceProperty);
            set => SetValue(SourceProperty, value);
        }

        public IEnumerable<InspectionRegion>? Regions
        {
            get => (IEnumerable<InspectionRegion>?)GetValue(RegionsProperty);
            set => SetValue(RegionsProperty, value);
        }

        /// <summary>Draw the similarity score above each box once the box is big enough on screen.</summary>
        public bool ShowScores
        {
            get => (bool)GetValue(ShowScoresProperty);
            set => SetValue(ShowScoresProperty, value);
        }

        /// <summary>When true, left-drag draws a rubber-band box instead of panning.</summary>
        public bool IsDrawMode
        {
            get => (bool)GetValue(IsDrawModeProperty);
            set => SetValue(IsDrawModeProperty, value);
        }

        /// <summary>Raised when the user finishes drawing a box. The rectangle is in image pixels.</summary>
        public event EventHandler<Rect>? RegionDrawn;

        // ---------- View state ----------
        private double _zoom = 1.0;       // 1.0 = fit to window
        private double _panX, _panY;      // screen-pixel offset from the centered position
        private bool _panning;
        private bool _drawing;
        private Point _lastMouse;
        private Point _drawStart, _drawCurrent;   // image pixels

        private const double MinZoom = 0.5;
        private const double MaxZoom = 40.0;
        private const double MinLabelBoxWidth = 24;   // on-screen px

        private readonly Dictionary<string, (Brush brush, Pen pen)> _styleCache = new();
        private static readonly Pen RubberPen = CreateRubberPen();
        private static readonly Brush RubberFill = CreateRubberFill();

        private static Pen CreateRubberPen()
        {
            var pen = new Pen(Brushes.White, 1.5) { DashStyle = DashStyles.Dash };
            pen.Freeze();
            return pen;
        }

        private static Brush CreateRubberFill()
        {
            var b = new SolidColorBrush(Color.FromArgb(40, 255, 255, 255));
            b.Freeze();
            return b;
        }

        public InspectionDisplay()
        {
            ClipToBounds = true;
            Focusable = true;
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.HighQuality);
        }

        public void ResetView()
        {
            _zoom = 1.0;
            _panX = 0;
            _panY = 0;
            InvalidateVisual();
        }

        // ---------- Change handlers ----------
        private static void OnSourceChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            var self = (InspectionDisplay)d;
            var oldImg = e.OldValue as BitmapSource;
            var newImg = e.NewValue as BitmapSource;

            // Keep the user's zoom between images of the same size (batch mode), reset otherwise
            if (oldImg == null || newImg == null ||
                oldImg.PixelWidth != newImg.PixelWidth || oldImg.PixelHeight != newImg.PixelHeight)
            {
                self._zoom = 1.0;
                self._panX = 0;
                self._panY = 0;
            }
        }

        private static void OnRegionsChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            var self = (InspectionDisplay)d;
            if (e.OldValue is INotifyCollectionChanged oldC)
                oldC.CollectionChanged -= self.OnRegionsCollectionChanged;
            if (e.NewValue is INotifyCollectionChanged newC)
                newC.CollectionChanged += self.OnRegionsCollectionChanged;
        }

        private static void OnDrawModeChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            var self = (InspectionDisplay)d;
            self._drawing = false;
            self.Cursor = (bool)e.NewValue ? Cursors.Cross : Cursors.Arrow;
            self.InvalidateVisual();
        }

        private void OnRegionsCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
            => InvalidateVisual();

        // ---------- Geometry ----------
        // image->screen mapping: screen = offset + imagePixel * scale
        private (double scale, double ox, double oy) GetView()
        {
            var src = Source!;
            double w = src.PixelWidth, h = src.PixelHeight;
            double fit = Math.Min(ActualWidth / w, ActualHeight / h);
            double s = fit * _zoom;
            double ox = (ActualWidth - w * s) / 2.0 + _panX;
            double oy = (ActualHeight - h * s) / 2.0 + _panY;
            return (s, ox, oy);
        }

        private Point ScreenToImage(Point p)
        {
            var (s, ox, oy) = GetView();
            return new Point((p.X - ox) / s, (p.Y - oy) / s);
        }

        // ---------- Rendering ----------
        protected override void OnRender(DrawingContext dc)
        {
            base.OnRender(dc);

            var area = new Rect(0, 0, ActualWidth, ActualHeight);
            dc.DrawRectangle(Brushes.Black, null, area);   // also makes the whole control hit-testable

            var src = Source;
            if (src == null || ActualWidth <= 0 || ActualHeight <= 0) return;

            dc.PushClip(new RectangleGeometry(area));

            var (s, ox, oy) = GetView();
            dc.DrawImage(src, new Rect(ox, oy, src.PixelWidth * s, src.PixelHeight * s));

            var regions = Regions;
            if (regions != null)
            {
                double ppd = VisualTreeHelper.GetDpi(this).PixelsPerDip;

                foreach (var r in regions)
                {
                    var (brush, pen) = GetStyle(r.BoxColor);
                    var box = new Rect(ox + r.X * s, oy + r.Y * s, r.Width * s, r.Height * s);
                    dc.DrawRectangle(null, pen, box);

                    string? caption = null;
                    double size = 11;
                    if (!string.IsNullOrEmpty(r.Caption))
                    {
                        caption = r.Caption;      // defect / component names: always shown
                        size = 12;
                    }
                    else if (ShowScores && box.Width >= MinLabelBoxWidth)
                    {
                        caption = r.Score.ToString("F2", CultureInfo.InvariantCulture);
                    }

                    if (caption != null)
                    {
                        var text = new FormattedText(caption, CultureInfo.InvariantCulture,
                            FlowDirection.LeftToRight, new Typeface("Segoe UI"), size, brush, ppd);
                        dc.DrawText(text, new Point(box.X, box.Y - text.Height - 1));
                    }
                }
            }

            if (_drawing)
            {
                double x0 = Math.Min(_drawStart.X, _drawCurrent.X), y0 = Math.Min(_drawStart.Y, _drawCurrent.Y);
                double x1 = Math.Max(_drawStart.X, _drawCurrent.X), y1 = Math.Max(_drawStart.Y, _drawCurrent.Y);
                dc.DrawRectangle(RubberFill, RubberPen,
                    new Rect(ox + x0 * s, oy + y0 * s, (x1 - x0) * s, (y1 - y0) * s));
            }

            dc.Pop();
        }

        private (Brush brush, Pen pen) GetStyle(string colorText)
        {
            colorText = string.IsNullOrWhiteSpace(colorText) ? "LimeGreen" : colorText;
            if (_styleCache.TryGetValue(colorText, out var style)) return style;

            Brush brush;
            try { brush = (Brush)new BrushConverter().ConvertFromString(colorText)!; }
            catch { brush = Brushes.Yellow; }

            if (brush.CanFreeze) brush.Freeze();
            var pen = new Pen(brush, 2);
            pen.Freeze();

            style = (brush, pen);
            _styleCache[colorText] = style;
            return style;
        }

        // ---------- Mouse ----------
        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            base.OnMouseWheel(e);
            if (Source == null || ActualWidth <= 0 || ActualHeight <= 0) return;

            Point p = e.GetPosition(this);
            var (s, ox, oy) = GetView();
            double imgX = (p.X - ox) / s;      // image point under the cursor
            double imgY = (p.Y - oy) / s;

            double factor = e.Delta > 0 ? 1.15 : 1.0 / 1.15;
            _zoom = Math.Max(MinZoom, Math.Min(MaxZoom, _zoom * factor));

            // Keep the same image point under the cursor after zooming
            double fit = Math.Min(ActualWidth / Source.PixelWidth, ActualHeight / Source.PixelHeight);
            double s2 = fit * _zoom;
            _panX = (p.X - imgX * s2) - (ActualWidth - Source.PixelWidth * s2) / 2.0;
            _panY = (p.Y - imgY * s2) - (ActualHeight - Source.PixelHeight * s2) / 2.0;

            InvalidateVisual();
            e.Handled = true;
        }

        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            if (Source == null) return;

            if (IsDrawMode)
            {
                _drawing = true;
                _drawStart = _drawCurrent = ScreenToImage(e.GetPosition(this));
                CaptureMouse();
                InvalidateVisual();
                return;
            }

            if (e.ClickCount == 2) { ResetView(); return; }
            StartPan(e.GetPosition(this));
        }

        protected override void OnMouseRightButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseRightButtonDown(e);
            if (Source != null) StartPan(e.GetPosition(this));
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (Source == null) return;

            Point p = e.GetPosition(this);
            if (_panning)
            {
                _panX += p.X - _lastMouse.X;
                _panY += p.Y - _lastMouse.Y;
                _lastMouse = p;
                InvalidateVisual();
            }
            else if (_drawing)
            {
                _drawCurrent = ScreenToImage(p);
                InvalidateVisual();
            }
        }

        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonUp(e);
            if (_drawing) FinishDrawing();
            else EndPan();
        }

        protected override void OnMouseRightButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseRightButtonUp(e);
            EndPan();
        }

        private void StartPan(Point p)
        {
            _panning = true;
            _lastMouse = p;
            CaptureMouse();
            Cursor = Cursors.Hand;
        }

        private void EndPan()
        {
            if (!_panning) return;
            _panning = false;
            ReleaseMouseCapture();
            Cursor = IsDrawMode ? Cursors.Cross : Cursors.Arrow;
        }

        private void FinishDrawing()
        {
            _drawing = false;
            ReleaseMouseCapture();
            InvalidateVisual();

            var src = Source;
            if (src == null) return;

            double x0 = Math.Max(0, Math.Min(_drawStart.X, _drawCurrent.X));
            double y0 = Math.Max(0, Math.Min(_drawStart.Y, _drawCurrent.Y));
            double x1 = Math.Min(src.PixelWidth, Math.Max(_drawStart.X, _drawCurrent.X));
            double y1 = Math.Min(src.PixelHeight, Math.Max(_drawStart.Y, _drawCurrent.Y));

            if (x1 - x0 >= 4 && y1 - y0 >= 4)
                RegionDrawn?.Invoke(this, new Rect(x0, y0, x1 - x0, y1 - y0));
        }
    }
}