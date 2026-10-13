using System;
using System.IO;

namespace ModelTraining.Services
{
    /// <summary>Finds a file in &lt;exe&gt;\Assets first, then next to the exe.</summary>
    public static class AssetLocator
    {
        public static string? Find(string fileName)
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string inAssets = Path.Combine(baseDir, "Assets", fileName);
            if (File.Exists(inAssets)) return inAssets;

            string inBase = Path.Combine(baseDir, fileName);
            return File.Exists(inBase) ? inBase : null;
        }

        public static string Resolve(string fileName) =>
            Find(fileName) ?? throw new FileNotFoundException(
                $"Could not find '{fileName}' in the Assets folder or next to the exe.", fileName);
    }
}