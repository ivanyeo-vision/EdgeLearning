using System.IO;
using System.Windows.Media.Imaging;
using OpenCvSharp;

namespace ModelTraining.Services
{
    public static class ImageConverter
    {
        public static BitmapSource MatToBitmapSource(Mat image)
        {
            using var memoryStream = new MemoryStream(image.ImEncode(".bmp"));
            var bitmapImage = new BitmapImage();
            bitmapImage.BeginInit();
            bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
            bitmapImage.StreamSource = memoryStream;
            bitmapImage.EndInit();
            bitmapImage.Freeze(); 
            return bitmapImage;
        }
    }
}