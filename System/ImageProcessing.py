import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift

class BaseProcessor(list):
    def __init__(self, target_resolution:tuple = (128, 128), frames_per_reset:int = 10):
        self.frames_processed = 0
        self.target_resolution = target_resolution
        self.frames_per_reset = frames_per_reset
        self.cm = (0, 0)

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
        pass

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
        self.scale_factor = 1

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

if __name__ == "__main__":
    fname = r'C:\Users\Tom\Google Drive\corrected second suite\Tom Horn 1.mp4'
    vid = VideoProcessor(fname)

