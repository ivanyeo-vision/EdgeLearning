using System;
using System.Collections.Generic;
using OpenCvSharp;

namespace ModelTraining.Services
{
    /// <summary>Tunable parameters, stored in recipe.xml (&lt;DefectParams&gt;).</summary>
    public sealed class DefectParams
    {
        public int ContextMargin { get; set; } = 8;        // px around the pin box (must reach the neighbour to see a short)
        public int AlignSearch { get; set; } = 3;          // +/- px for the fine re-alignment
        public int MinBlobArea { get; set; } = 6;          // smaller blobs are noise
        public int HoleMinArea { get; set; } = 8;          // smallest enclosed region that counts as a hole
        public double HoleFillRatio { get; set; } = 0.5;   // hole is "missing" when >= this fraction of it is copper now
        public double BiteMaxFraction { get; set; } = 0.35;// a bite is smaller than this fraction of its copper piece
        public int HybridMinArea { get; set; } = 12;       // hybrid: a short/open must be this big (px) to fail a pin the AI passed
    }

    /// <summary>
    /// Geometric defect classification (port of DefectClassifier.py, tested there on synthetic boards).
    ///   Missing hole : the golden pad's hole area is copper in the live image
    ///   Short        : extra copper touching two or more separate golden copper pieces
    ///   Open circuit : one golden copper piece became two or more pieces
    ///   Mouse bite   : small piece of copper missing at the outer edge of a copper piece
    /// HYBRID use: the AI similarity model cannot see thin traces, so DetectShortOpen() runs ONLY the short / open
    /// checks on every pin, even when the AI says PASS. ClassifyPin() is the full classification for AI-NG pins.
    /// Keep this file and DefectClassifier.py in sync.
    /// </summary>
    public static class DefectClassifier
    {
        public const string LabelMissingHole = "Missing hole";
        public const string LabelShort = "Short";
        public const string LabelOpen = "Open circuit";
        public const string LabelChipping = "Chipping";
        public const string LabelExtra = "Extra copper";
        public const string LabelMissing = "Missing";
        public const string LabelUnknown = "NG";

        /// <summary>Full classification (for pins the AI flagged NG).</summary>
        /// <param name="box">pin box on the REFERENCE board</param>
        /// <param name="dx">fiducial alignment offset of the live image</param>
        public static string ClassifyPin(Mat refBgr, Mat liveBgr, Rect box, int dx, int dy, DefectParams p)
        {
            var (g, l) = Prepare(refBgr, liveBgr, box, dx, dy, p);
            using (g)
            using (l)
            {
                Binarize(g, l, out byte[] gFg, out byte[] lFg);
                return ClassifyMasks(gFg, lFg, g.Width, g.Height, p);
            }
        }

        
        public static string? DetectShortOpen(Mat refBgr, Mat liveBgr, Rect box, int dx, int dy,
                                              DefectParams p, out int area, int? minAreaOverride = null)
        {
            var (g, l) = Prepare(refBgr, liveBgr, box, dx, dy, p);
            using (g)
            using (l)
            {
                Binarize(g, l, out byte[] gFg, out byte[] lFg);
                return ShortOpenFromMasks(gFg, lFg, g.Width, g.Height, p, minAreaOverride ?? p.HybridMinArea, out area);
            }
        }

        /// <summary>Golden and live gray crops (with context); the live one is fine-aligned to the golden one.</summary>
        private static (Mat g, Mat l) Prepare(Mat refBgr, Mat liveBgr, Rect box, int dx, int dy, DefectParams p)
        {
            int m = p.ContextMargin, s = p.AlignSearch;

            using Mat gColor = Grab(refBgr, box.X - m, box.Y - m, box.Width + 2 * m, box.Height + 2 * m);
            using Mat bigColor = Grab(liveBgr, box.X + dx - m - s, box.Y + dy - m - s,
                                      box.Width + 2 * m + 2 * s, box.Height + 2 * m + 2 * s);

            Mat gGray = new Mat();
            using Mat bigGray = new Mat();
            Cv2.CvtColor(gColor, gGray, ColorConversionCodes.BGR2GRAY);
            Cv2.CvtColor(bigColor, bigGray, ColorConversionCodes.BGR2GRAY);

            // Fine re-alignment: the fiducial alignment is only pixel accurate
            using Mat res = new Mat();
            Cv2.MatchTemplate(bigGray, gGray, res, TemplateMatchModes.CCoeffNormed);
            Cv2.MinMaxLoc(res, out _, out _, out _, out Point loc);

            using Mat view = new Mat(bigGray, new Rect(loc.X, loc.Y, gGray.Width, gGray.Height));
            return (gGray, view.Clone());
        }

        /// <summary>Blur, Otsu on the golden crop, the SAME threshold for the live crop. Masks are 255 = copper.</summary>
        private static void Binarize(Mat gGray, Mat lGray, out byte[] gFg, out byte[] lFg)
        {
            using Mat gBlur = new Mat();
            using Mat lBlur = new Mat();
            Cv2.GaussianBlur(gGray, gBlur, new Size(3, 3), 0);
            Cv2.GaussianBlur(lGray, lBlur, new Size(3, 3), 0);

            using Mat gBin = new Mat();
            double thr = Cv2.Threshold(gBlur, gBin, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
            gBin.GetArray(out gFg);
            lBlur.GetArray(out byte[] lGrayBytes);

            lFg = new byte[lGrayBytes.Length];
            for (int i = 0; i < lFg.Length; i++) lFg[i] = lGrayBytes[i] > thr ? (byte)255 : (byte)0;
        }

        /// <summary>Only the short / open checks (see DefectClassifier.short_open_from_masks in Python).</summary>
        private static string? ShortOpenFromMasks(byte[] gFg, byte[] lFg, int w, int h, DefectParams p,
                                                  int minArea, out int bestArea)
        {
            int minA = p.MinBlobArea;
            gFg = RemoveSmall(gFg, w, h, minA);
            lFg = RemoveSmall(lFg, w, h, minA);

            byte[] gHoles = RemoveSmall(EnclosedHoles(gFg, w, h), w, h, p.HoleMinArea);
            byte[] holeZone = Dilate3(gHoles, w, h);
            byte[] missing = RemoveSmall(Open3(AndNot(AndNot(gFg, lFg), holeZone), w, h), w, h, minA);
            byte[] extra = RemoveSmall(Open3(AndNot(AndNot(lFg, gFg), holeZone), w, h), w, h, minA);

            int[] gLab = Label(gFg, w, h, out int gCount, out _);
            int[] lLab = Label(lFg, w, h, out _, out _);

            string? bestLabel = null;
            bestArea = 0;

            // Short: extra copper touching two or more golden copper pieces
            int[] eLab = Label(extra, w, h, out int eCount, out int[] eAreas);
            for (int i = 1; i <= eCount; i++)
            {
                byte[] ring = Dilate3(MaskOf(eLab, i), w, h);
                var touched = new HashSet<int>();
                for (int k = 0; k < ring.Length; k++)
                    if (ring[k] != 0 && gLab[k] != 0) touched.Add(gLab[k]);

                if (touched.Count >= 2 && eAreas[i] > bestArea)
                {
                    bestLabel = LabelShort;
                    bestArea = eAreas[i];
                }
            }

            // Open circuit: one golden copper piece became two or more pieces (area = size of the gap)
            if (Count(missing) > 0)
            {
                for (int gi = 1; gi <= gCount; gi++)
                {
                    var overlap = new Dictionary<int, int>();
                    for (int k = 0; k < gLab.Length; k++)
                    {
                        if (gLab[k] != gi || lLab[k] == 0) continue;
                        overlap.TryGetValue(lLab[k], out int c);
                        overlap[lLab[k]] = c + 1;
                    }
                    int pieces = 0;
                    foreach (var kv in overlap) if (kv.Value >= minA) pieces++;
                    if (pieces < 2) continue;

                    int gap = 0;
                    for (int k = 0; k < gLab.Length; k++)
                        if (gLab[k] == gi && missing[k] != 0) gap++;

                    if (gap > bestArea)
                    {
                        bestLabel = LabelOpen;
                        bestArea = gap;
                    }
                }
            }

            return (bestLabel != null && bestArea >= minArea) ? bestLabel : null;
        }

        private static string ClassifyMasks(byte[] gFg, byte[] lFg, int w, int h, DefectParams p)
        {
            int minA = p.MinBlobArea;
            gFg = RemoveSmall(gFg, w, h, minA);
            lFg = RemoveSmall(lFg, w, h, minA);

            // 1. Missing hole: the golden hole area is now copper
            byte[] gHoles = RemoveSmall(EnclosedHoles(gFg, w, h), w, h, p.HoleMinArea);
            int gHoleArea = Count(gHoles);
            if (gHoleArea >= p.HoleMinArea)
            {
                int filled = 0;
                for (int i = 0; i < gHoles.Length; i++)
                    if (gHoles[i] != 0 && lFg[i] != 0) filled++;
                if (filled >= p.HoleFillRatio * gHoleArea) return LabelMissingHole;
            }

            // Differences, ignoring the hole surroundings and thin edge noise
            byte[] holeZone = Dilate3(gHoles, w, h);
            byte[] missing = RemoveSmall(Open3(AndNot(AndNot(gFg, lFg), holeZone), w, h), w, h, minA);
            byte[] extra = RemoveSmall(Open3(AndNot(AndNot(lFg, gFg), holeZone), w, h), w, h, minA);

            int[] gLab = Label(gFg, w, h, out int gCount, out int[] gAreas);
            int[] lLab = Label(lFg, w, h, out _, out _);

            // 2. Short: extra copper touching two or more golden copper pieces
            int[] eLab = Label(extra, w, h, out int eCount, out _);
            for (int i = 1; i <= eCount; i++)
            {
                byte[] ring = Dilate3(MaskOf(eLab, i), w, h);
                var touched = new HashSet<int>();
                for (int k = 0; k < ring.Length; k++)
                    if (ring[k] != 0 && gLab[k] != 0) touched.Add(gLab[k]);
                if (touched.Count >= 2) return LabelShort;
            }

            // 3. Open circuit: one golden copper piece became two or more live pieces
            bool anyMissing = Count(missing) > 0;
            if (anyMissing)
            {
                for (int gi = 1; gi <= gCount; gi++)
                {
                    var overlap = new Dictionary<int, int>();
                    for (int k = 0; k < gLab.Length; k++)
                    {
                        if (gLab[k] != gi || lLab[k] == 0) continue;
                        overlap.TryGetValue(lLab[k], out int c);
                        overlap[lLab[k]] = c + 1;
                    }
                    int big = 0;
                    foreach (var kv in overlap) if (kv.Value >= minA) big++;
                    if (big >= 2) return LabelOpen;
                }
            }

            // 4. Mouse bite: small missing piece touching the outer edge of a golden copper piece
            var outerBg = new byte[gFg.Length];
            for (int k = 0; k < outerBg.Length; k++)
                outerBg[k] = (gFg[k] == 0 && gHoles[k] == 0) ? (byte)255 : (byte)0;

            int[] mLab = Label(missing, w, h, out int mCount, out _);
            for (int i = 1; i <= mCount; i++)
            {
                byte[] blob = MaskOf(mLab, i);
                int area = Count(blob);
                byte[] ring = Dilate3(blob, w, h);

                bool touchesOutside = false;
                for (int k = 0; k < ring.Length; k++)
                    if (ring[k] != 0 && outerBg[k] != 0) { touchesOutside = true; break; }
                if (!touchesOutside) continue;

                int compArea = 0;
                for (int k = 0; k < blob.Length; k++)
                    if (blob[k] != 0 && gLab[k] != 0) compArea = Math.Max(compArea, gAreas[gLab[k]]);

                if (compArea > 0 && area <= p.BiteMaxFraction * compArea) return LabelChipping;
            }

            if (Count(extra) > 0) return LabelExtra;
            if (anyMissing) return LabelMissing;
            return LabelUnknown;
        }

       
        private static Mat Grab(Mat img, int x, int y, int w, int h)
        {
            int x0 = Math.Max(x, 0), y0 = Math.Max(y, 0);
            int x1 = Math.Min(x + w, img.Width), y1 = Math.Min(y + h, img.Height);
            if (x1 <= x0 || y1 <= y0)
                return new Mat(h, w, img.Type(), Scalar.All(0));

            using Mat roi = new Mat(img, new Rect(x0, y0, x1 - x0, y1 - y0));
            Mat result = new Mat();
            Cv2.CopyMakeBorder(roi, result, y0 - y, (y + h) - y1, x0 - x, (x + w) - x1, BorderTypes.Replicate);
            return result;
        }

        
        private static int Count(byte[] m)
        {
            int c = 0;
            foreach (byte v in m) if (v != 0) c++;
            return c;
        }

        private static byte[] AndNot(byte[] a, byte[] b)
        {
            var o = new byte[a.Length];
            for (int i = 0; i < a.Length; i++) o[i] = (a[i] != 0 && b[i] == 0) ? (byte)255 : (byte)0;
            return o;
        }

        private static byte[] MaskOf(int[] labels, int id)
        {
            var o = new byte[labels.Length];
            for (int i = 0; i < labels.Length; i++) if (labels[i] == id) o[i] = 255;
            return o;
        }

        private static byte[] Dilate3(byte[] m, int w, int h)
        {
            var o = new byte[m.Length];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    if (m[y * w + x] == 0) continue;
                    for (int yy = Math.Max(0, y - 1); yy <= Math.Min(h - 1, y + 1); yy++)
                        for (int xx = Math.Max(0, x - 1); xx <= Math.Min(w - 1, x + 1); xx++)
                            o[yy * w + xx] = 255;
                }
            return o;
        }

        private static byte[] Erode3(byte[] m, int w, int h)
        {
            var o = new byte[m.Length];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    bool all = true;
                    for (int yy = Math.Max(0, y - 1); yy <= Math.Min(h - 1, y + 1) && all; yy++)
                        for (int xx = Math.Max(0, x - 1); xx <= Math.Min(w - 1, x + 1); xx++)
                            if (m[yy * w + xx] == 0) { all = false; break; }
                    if (all) o[y * w + x] = 255;
                }
            return o;
        }

        private static byte[] Open3(byte[] m, int w, int h) => Dilate3(Erode3(m, w, h), w, h);

        /// <summary>8-connected component labelling. Label 0 = background; areas[label] = pixel count.</summary>
        private static int[] Label(byte[] m, int w, int h, out int count, out int[] areas)
        {
            var lab = new int[m.Length];
            var areaList = new List<int> { 0 };
            var stack = new Stack<int>();
            count = 0;

            for (int i = 0; i < m.Length; i++)
            {
                if (m[i] == 0 || lab[i] != 0) continue;
                count++;
                int area = 0;
                lab[i] = count;
                stack.Push(i);
                while (stack.Count > 0)
                {
                    int cur = stack.Pop();
                    area++;
                    int cx = cur % w, cy = cur / w;
                    for (int yy = Math.Max(0, cy - 1); yy <= Math.Min(h - 1, cy + 1); yy++)
                        for (int xx = Math.Max(0, cx - 1); xx <= Math.Min(w - 1, cx + 1); xx++)
                        {
                            int ni = yy * w + xx;
                            if (m[ni] != 0 && lab[ni] == 0) { lab[ni] = count; stack.Push(ni); }
                        }
                }
                areaList.Add(area);
            }
            areas = areaList.ToArray();
            return lab;
        }

        private static byte[] RemoveSmall(byte[] m, int w, int h, int minArea)
        {
            int[] lab = Label(m, w, h, out _, out int[] areas);
            var o = new byte[m.Length];
            for (int i = 0; i < m.Length; i++)
                if (lab[i] != 0 && areas[lab[i]] >= minArea) o[i] = 255;
            return o;
        }

        /// <summary>Background regions that are not connected to the border (holes inside copper).</summary>
        private static byte[] EnclosedHoles(byte[] fg, int w, int h)
        {
            var outside = new bool[fg.Length];
            var stack = new Stack<int>();

            void Seed(int x, int y)
            {
                int i = y * w + x;
                if (fg[i] == 0 && !outside[i]) { outside[i] = true; stack.Push(i); }
            }
            for (int x = 0; x < w; x++) { Seed(x, 0); Seed(x, h - 1); }
            for (int y = 0; y < h; y++) { Seed(0, y); Seed(w - 1, y); }

            while (stack.Count > 0)
            {
                int cur = stack.Pop();
                int cx = cur % w, cy = cur / w;
                if (cx > 0) Seed(cx - 1, cy);
                if (cx < w - 1) Seed(cx + 1, cy);
                if (cy > 0) Seed(cx, cy - 1);
                if (cy < h - 1) Seed(cx, cy + 1);
            }

            var holes = new byte[fg.Length];
            for (int i = 0; i < fg.Length; i++)
                if (fg[i] == 0 && !outside[i]) holes[i] = 255;
            return holes;
        }
    }
}