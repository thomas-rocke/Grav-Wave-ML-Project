import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
from scipy.ndimage import zoom
from scipy.optimize import brute
import time
from Utils import meanError
from Gaussian_Beam import Superposition, Hermite

class BaseProcessor(list):
    def __init__(self, target_resolution:tuple = (128, 128), frames_per_reset:int = 10):
        self.frames_processed = 0
        self.target_resolution = target_resolution
        self.frames_per_reset = frames_per_reset
        self.cm = (0, 0)
        self.scale_factor = 20

    def ToGreyscale(self, image):
        '''
        Convert target image to greyscale
        '''
        grey_vec = [0.2989, 0.5870, 0.1140]
        grey_image = np.dot(image[..., :3], grey_vec)
        return grey_image

    def Recenter(self, image):
        '''
        Recenter target greyscale image on the center of mass
        '''
        diff = (int(image.shape[0]/2 - self.cm[0]), int(image.shape[1]/2 - self.cm[1]))
        return shift(image, diff, cval=0)
        

    def Rescale(self, image):
        '''
        Rescale target image
        '''
        zoom_factor = 1.8 * self.scale_factor # 99% of Gaussian within 1.8 sig radius
        dims = image.shape
        new_dims = (dims[0]/zoom_factor, dims[1]/zoom_factor)
        crop_start = (int(dims[0]/2 - new_dims[0]/2), int(dims[1]/2 - new_dims[1]/2))
        crop_end = (int(crop_start[0] + new_dims[0]/2), int(crop_start[1] + new_dims[1]/2))
        return image[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]]

    def _resetCenter(self, image):
        '''
        Resets the coordinate used to center the images
        '''
        threshold_value = threshold_otsu(image)
        labeled_foreground = (image > threshold_value).astype(int)
        properties = regionprops(labeled_foreground, image)
        center_of_mass = properties[0].centroid
        self.cm = tuple(center_of_mass)

    def _resetScale(self, image):
        '''
        Resets the scale factor used to rescale the images
        '''
        normed_image = image/np.linalg.norm(image)
        vals = np.array([WidthModel(w, normed_image) for w in np.arange(1, 100)])
        self.scale_factor = np.argmin(vals)

    def MakeSquare(self, image):
        '''
        Crops the target greyscale image so the aspect ratio is square
        '''
        min_pixels = np.min(image.shape)
        max_pixels = np.max(image.shape)
        offset = int((max_pixels - min_pixels)/2)
        square_image = np.empty((min_pixels, min_pixels))
        if image.shape[0] == max_pixels:
            # Long side is first axis
            square_image = image[offset:(offset+min_pixels), :]
        else:
            square_image = image[:, offset:(offset+min_pixels)]
        return square_image


    def ChangeResolution(self, image):
        '''
        Change resolution of image to self.target_resolution
        '''
        return cv2.resize(image, self.target_resolution, interpolation=cv2.INTER_CUBIC)

    def Normalise(self, image):
        '''
        Normalise target image
        '''
        return image / np.linalg.norm(image)

    def Process(self):
        '''
        Perform all operations to generate an image usable by Neural Net
        '''
        image = self[self.frames_processed]
        grey_image = self.ToGreyscale(image)
        if  not (self.frames_processed % self.frames_per_reset):
            self._resetCenter(grey_image)
            self._resetScale(grey_image)
        centered_image = self.Recenter(grey_image)
        scaled_image = self.Rescale(centered_image)
        squared_image = self.MakeSquare(scaled_image)
        rezzed_image = self.ChangeResolution(squared_image)
        normed_image = self.Normalise(rezzed_image)
        self.frames_processed += 1
        return normed_image


class VideoProcessor(BaseProcessor):

    def __init__(self, video_file, target_resolution:tuple = (128, 128), frames_per_reset=10):
        super().__init__(target_resolution, frames_per_reset)
        self.cap = cv2.VideoCapture(video_file)
        self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.ret = True
        self.buf = np.empty((frameWidth, frameHeight, 3), dtype='uint8') # Define buffer for frame data

    def __getitem__(self, i):
        if self.frames_processed < self.frameCount and self.ret:
            self.ret, self.buf = self.cap.read()
            return self.buf
        else:
            self.cap.release() # Close video file
            raise IndexError('End of video file reached')


def WidthModel(width, image):
    x_center = int(image.shape[0]/2)
    y_center = int(image.shape[1]/2)
    func_im = np.fromfunction(lambda i, j: np.exp(-((i - x_center)**2 + (j - y_center)**2)/width**2), image.shape)
    return ((image - func_im/np.sqrt(2*np.pi*width**2))**2).mean(axis=None)

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out

if __name__ == "__main__":
    fname = r'C:\Users\Tom\Google Drive\corrected second suite\Tom Horn 1.mp4'
    vid = VideoProcessor(fname)
    ts = [0]*50
    for i in range(50):
        t = time.time()
        vid.Process()
        ts[i] = time.time() - t
    print(meanError(ts))
    plt.plot(range(50), ts)
    plt.show()
    
