import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

class BaseProcessor(list):
    def __init__(self, target_resolution:tuple = (128, 128), frames_per_reset:int = 10):
        self.frames_processed = 0
        self.target_resolution = target_resolution
        self.frames_per_reset = frames_per_reset

    def ToGreyscale(self, image):
        '''
        Convert target image to greyscale
        '''
        pass

    def Recenter(self, image):
        '''
        Recenter target greyscale image on the center of mass
        '''
        
        pass

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
        self.cm = [center_of_mass[1], center_of_mass[0]]

    def _resetScale(self, image):
        '''
        Resets the scale factor used to rescale the images
        '''
        self.scale_factor = 1

    def ChangeResolution(self, image):
        '''
        Change resolution of image to self.target_resolution
        '''
        pass

    def Normalise(self, image):
        '''
        Normalise target image
        '''
        pass

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
        rezzed_image = self.ChangeResolution(scaled_image)
        normed_image = self.Normalise(rezzed_image)
        self.frames_processed += 1
        return normed_image


class VideoProcessor(BaseProcessor):

    def __init__(self, video_file, target_resolution:tuple = (128*128), frames_per_reset=10):
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
            self.cap.release()
            raise IndexError('End of video file reached')

if __name__ == "__main__":
    fname = r'C:\Users\Tom\Google Drive\corrected second suite\Tom Horn 1.mp4'
    vid = VideoProcessor(fname)
    img = vid[0]
    plt.imshow(img)
    plt.show()