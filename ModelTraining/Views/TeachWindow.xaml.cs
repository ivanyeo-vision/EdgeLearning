using System.Windows;
using ModelTraining.ViewModels;

namespace ModelTraining.Views
{
    public partial class TeachWindow : Window
    {
        public TeachWindow()
        {
            InitializeComponent();
            DataContext = new TeachViewModel();
        }

        // The display control reports the drawn box in IMAGE pixels, no canvas maths needed
        private void Display_RegionDrawn(object? sender, Rect region)
        {
            if (DataContext is TeachViewModel vm) vm.OnRegionDrawn(region);
        }
    }
}