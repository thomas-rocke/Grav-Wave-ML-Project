import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import ArtistAnimation
from scipy.ndimage.interpolation import shift
from scipy.ndimage import zoom
from scipy.optimize import basinhopping
import time
from Utils import meanError
from Gaussian_Beam import Superposition, Hermite
from multiprocessing import cpu_count, Pool
from skimage.measure import regionprops
from skimage.filters import threshold_otsu

class BaseProcessor(list):
    def __init__(self, target_resolution:tuple = (128, 128), frames_per_reset:int = 10):
        self.frames_processed = 0
        self.target_resolution = target_resolution
        self.frames_per_reset = frames_per_reset
        self.SquareX = 0
        self.SquareY = 0
        self.SquareScale = 0 # Set Bounding box defaults

    def ToGreyscale(self, image):
        '''
        Convert target image to greyscale
        '''
        grey_vec = [0.2989, 0.5870, 0.1140]
        grey_image = np.dot(image[..., :3], grey_vec)
        return grey_image

    def _getCenterOfMass(self, image):
        '''
        Searches for center of mass of target image, and changes center of bounding box to lie on this position
        '''
        threshold_value = threshold_otsu(image)
        labeled_foreground = (image > threshold_value).astype(int)
        properties = regionprops(labeled_foreground, image)
        center_of_mass = properties[0].centroid
        self.SquareX, self.SquareY = center_of_mass[1], center_of_mass[0]

    def _resetSquare(self, image):
        '''
        Resets the Square bounding box size
        '''
        max_sidelength = np.min(image.shape) # Get the length of the shortest image side
        test_sides = np.arange(int(0.3*max_sidelength), max_sidelength)
        least_square_vals = [self._WidthModel(side_length, image) for side_length in test_sides]
        self.SquareSide = test_sides[np.argmax(least_square_vals)]
    
    def _WidthModel(self, SquareSide, image):
        # Model used in maximisation problem to find bounding box
        x_start = int(image.shape[0]/2 + self.SquareX - SquareSide/2)
        y_start = int(image.shape[1]/2 + self.SquareY - SquareSide/2)
        x_end = int(min(x_start + SquareSide, image.shape[0]))
        y_end = int(min(y_start + SquareSide, image.shape[1]))
        mean_square = (image[x_start:x_end, y_start:y_end]**2).mean(axis=None)
        Square_area = SquareSide**2
        return (mean_square - 1e-5*(Square_area/image.size))

    def MakeSquare(self, image):
        '''
        Crops the target greyscale image so the aspect ratio is the square given by self.SquareX, self.SquareY and self.SquareScale. Image pixels outside the original image default to 0
        '''
        new_image = np.zeros((self.SquareSide, self.SquareSide))
        x_start = int(image.shape[0]/2 + self.SquareX - self.SquareSide/2)
        y_start = int(image.shape[1]/2 + self.SquareY - self.SquareSide/2)
        for i in range(self.SquareSide):
            for j in range(self.SquareSide):
                x = i + x_start
                y = j + y_start
                if x > 0 and y > 0:
                    try:
                        new_image[i, j] = image[x, y]
                    except:
                        pass
        return new_image

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

    def getImages(self, batch_size:int = 1):
        '''
        Perform all operations to generate an image usable by Neural Net, and return a batch of batch_size images
        '''
        images = [self[self.frames_processed + i] for i in range(batch_size)]
        self._resetSquare(self.ToGreyscale(images[0])) # Resets the size of the bounding box based on the first image of the batch
        processed_images = [0]*len(images)
        for i, image in enumerate(images):
            grey_image = self.ToGreyscale(image)
            squared_image = self.MakeSquare(grey_image)
            rezzed_image = self.ChangeResolution(squared_image)
            normed_image = self.Normalise(rezzed_image)
            processed_images[i] = normed_image
            self.frames_processed += 1
        return processed_images

class VideoProcessor(BaseProcessor):

    def __init__(self, video_file, target_resolution:tuple = (128, 128), frames_per_reset=10):
        super().__init__(target_resolution, frames_per_reset)
        self.cap = cv2.VideoCapture(video_file)
        self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.framerate = int(self.cap.get(cv2.CAP_PROP_FPS))
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
    fname = r'C:\Users\Tom\Downloads\EditedBeamModes.mp4'
    vid = VideoProcessor(fname, frames_per_reset=1)
    
    fig, ax = plt.subplots(ncols=5)
    vid.target_resolution = (1920, 1080)
    ims = vid.getImages(5)
    print(vid.SquareSide)
    for i in range(5):
        ax[i].imshow(ims[i])
    
    plt.show()