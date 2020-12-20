using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Text;
using System.Windows.Media.Imaging;
using IronPython;

namespace ImageProcessing
{
    public Interface BaseVideoHandler()
    {   
        public int ImagePixels
        public BaseVideoHandler(int pixels)
        {
            // Init class
            ImagePixels = pixels
        }

        public abstract CaptureImage() { } // Force child classes to include method of capturing image data

        private CenterImage()
        {
            // Finds Center of Mass of input Image, and returns new image with Center of Mass at the center
        }

        private ScaleImage()
        {
            // Rescales input Image
        }

        private ChangeResolution()
        {
            // Changes Image Resolution to ImagePixels x ImagePixels
        }

    }



}