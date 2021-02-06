import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.filters import threshold_otsu, gaussian
from Gaussian_Beam import Superposition, Hermite, Laguerre
from multiprocessing import cpu_count, Pool
import matplotlib.pyplot as plt
import time
from Utils import meanError, get_cams

import Logger

LOG = Logger.get_logger(__name__)


class BaseProcessor(list):
    def __init__(self, target_resolution:tuple = (128, 128)):
        self.frames_processed = 0
        self.target_resolution = target_resolution

    def toGreyscale(self, image):
        '''
        Convert target image to greyscale
        '''
        if len(image.shape) == 3:
            grey_vec = [0.2989, 0.5870, 0.1140]
            grey_image = np.dot(image[..., :3], grey_vec)
            return grey_image
        else:
            return image

    def _getCenterOfMass(self, image):
        '''
        Searches for center of mass of target image
        '''
        threshold_value = threshold_otsu(image)
        labeled_foreground = (image > threshold_value).astype(int)
        properties = regionprops(labeled_foreground, image)
        center_of_mass = properties[0].centroid
        return center_of_mass[1], center_of_mass[0]

    def _resetSquare(self, image):
        '''
        Resets the Square bounding box size
        '''
        SquareX, SquareY = self._getCenterOfMass(image)
        max_sidelength = np.min(image.shape) # Get the length of the shortest image side
        test_sides = np.arange(5, max_sidelength)
        least_square_vals = [self._widthModel(side_length, SquareX, SquareY, image) for side_length in test_sides]
        SquareSide = test_sides[np.argmax(least_square_vals)]
        return SquareSide, SquareX, SquareY
    
    def _widthModel(self, SquareSide, SquareX, SquareY, image):
        # Model used in maximisation problem to find bounding box
        x_start = int(SquareX - SquareSide/2)
        y_start = int(SquareY - SquareSide/2)
        x_end = int(min(x_start + SquareSide, image.shape[0]))
        y_end = int(min(y_start + SquareSide, image.shape[1]))
        mean_square = (image[x_start:x_end, y_start:y_end]**2).mean(axis=None)
        return (mean_square - 100*(SquareSide/max(image.shape)))

    def makeSquare(self, image, SquareSide=0, SquareX=0, SquareY=0):
        '''
        Crops the target greyscale image so the aspect ratio is the square given by SquareX, SquareY and SquareSide (defaulting to min(image.shape)). Image pixels outside the original image default to 0
        '''
        if SquareSide == 0:
            SquareSide = min(image.shape)

        new_image = np.zeros((SquareSide, SquareSide))
        x_start = int(SquareX - SquareSide/2)
        y_start = int(SquareY - SquareSide/2)
        for i in range(SquareSide):
            for j in range(SquareSide):
                x = i + x_start
                y = j + y_start
                if x >= 0 and y >= 0:
                    try:
                        new_image[i, j] = image[x, y]
                    except:
                        pass
        return new_image

    def changeResolution(self, image, target_resolution:tuple=(0, 0)):
        '''
        Change resolution of image to self.target_resolution
        '''
        if target_resolution == (0, 0):
            target_resolution = self.target_resolution
        return cv2.resize(image, target_resolution, interpolation=cv2.INTER_CUBIC)

    def normalise(self, image):
        '''
        Normalise target image
        '''
        return image / np.linalg.norm(image)

    def processImage(self, image, SquareSide=0, SquareX=0, SquareY=0):
        grey_image = self.toGreyscale(image)
        squared_image = self.makeSquare(grey_image, SquareSide, SquareX, SquareY)
        rezzed_image = self.changeResolution(squared_image)
        normed_image = self.normalise(rezzed_image)
        return normed_image

    def getImages(self, batch_size:int = 1):
        '''
        Perform all operations to generate an image usable by Neural Net, and return a batch of batch_size images
        '''
        msg = "Generating {} images".format(batch_size)
        LOG.info(msg)
        images = [self[self.frames_processed + i] for i in range(batch_size)]
        SquareSide, SquareX, SquareY = self._resetSquare(self.toGreyscale(images[0])) # Resets the size of the bounding box based on the first image of the batch
        processed_images = [self.processImage(image, SquareSide, SquareX, SquareY) for image in images]
        self.frames_processed += batch_size
        return processed_images

class VideoProcessor(BaseProcessor):

    def __init__(self, video_file, target_resolution:tuple = (128, 128)):
        super().__init__(target_resolution)
        msg = "Opening video file '{}'".format(video_file)
        LOG.info(msg)
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

class ModeProcessor(BaseProcessor):
    def __init__(self, camera:dict = {}, target_resolution:tuple=(128, 128)):
        self.change_camera(camera)
        super().__init__(target_resolution)

    def _reset_bins(self):
        raw_bins = np.zeros((2**self.bit_depth - 1, 2**self.bit_depth - 1, 2**self.bit_depth - 1)) # (R, G, B) matrix quantised to self.bit_depth
        shape = raw_bins.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    raw_bins[i, j, k] = 0.2989 * i + 0.5870 * j + 0.1140 * k # Convert quantisation to greyscale
        self.raw_bins = np.sort(raw_bins.flatten()) # Array of all greyscale intensity values possibly with bit_depth quantization

    def change_camera(self, camera:dict):
        msg = "Changing camera"
        LOG.info(msg)
        camera_keys = camera.keys()

        # Set up camera properties
        # Assume ideal camera if property not defined
        if 'noise_variance' in camera_keys:
            msg = "New 'noise_variance' is {}".format(camera['noise_variance'])
            LOG.debug(msg)
            self.noise_variance = camera['noise_variance']
        else:
            msg = "'noise_variance' not explicitly defined by new camera, defaulting to 0"
            LOG.warning(msg)
            self.noise_variance = 0
        
        if 'exposure_limits' in camera_keys:
            msg = "New 'exposure_limits' is {}".format(camera['exposure_limits'])
            LOG.debug(msg)
            self.exposure_limits = camera['exposure_limits']
        else:
            msg = "'exposure_limits' not explicitly defined by new camera, defaulting to (0, 1)"
            LOG.warning(msg)
            self.exposure_limits = (0, 1)
        
        if 'bit_depth' in camera_keys:
            msg = "New 'bit_depth' is {}".format(camera['bit_depth'])
            LOG.debug(msg)
            self.bit_depth = camera['bit_depth']
            self._reset_bins()
        else:
            msg = "'bit_depth' not explicitly defined by new camera, defaulting to 0"
            LOG.warning(msg)
            self.bit_depth = 0
        
        if 'blur_variance' in camera_keys:
            msg = "New 'blur_variance' is {}".format(camera['blur_variance'])
            LOG.debug(msg)
            self.blur_variance = camera['blur_variance']
        else:
            msg = "'blur_variance' not explicitly defined by new camera, defaulting to 0"
            LOG.warning(msg)
            self.blur_variance = 0
        
        if 'rotational_variance' in camera_keys:
            msg = "New 'rotational_variance' is {}".format(camera['rotational_variance'])
            LOG.debug(msg)
            self.rotational_variance = camera['rotational_variance']
        else:
            msg = "'rotational_variance' not explicitly defined by new camera, defaulting to 0"
            LOG.warning(msg)
            self.rotational_variance = 0
        
        if 'stretch_variance' in camera_keys:
            msg = "New 'stretch_variance' is {}".format(camera['stretch_variance'])
            LOG.debug(msg)
            self.stretch_variance = camera['stretch_variance']
        else:
            msg = "'stretch_variance' not explicitly defined by new camera, defaulting to 0"
            LOG.warning(msg)
            self.stretch_variance = 0

    def errorEffects(self, raw_image):
        '''
        Performs all image processing for noise effects on target image, using params from class init
        '''
        #shifted_image = shift_image(image,) # Shift the image in x and y coords
        rotated_image = self.add_rotational_error(raw_image, self.rotational_variance) # Perform rotation
        stretched_image = self.add_random_stretch(rotated_image, self.stretch_variance) # Add rstretch warping in random direction
        noisy_image = self.add_noise(stretched_image, self.noise_variance) # Add Gaussian Noise to the image
        blurred_image = self.blur_image(noisy_image, self.blur_variance) # Add gaussian blur
        exposed_image = self.add_exposure(blurred_image, self.exposure_limits) # Add exposure
        quantized_image = self.quantize_image(exposed_image, self.bit_depth) # Quantize
        return quantized_image

    def getImage(self, raw_image):
        '''
        Perform all processing on target superposition image to preprare it for training.
        '''
        noisy_image = self.errorEffects(raw_image)
        SquareSide, SquareX, SquareY = self._resetSquare(noisy_image) # Relocation of the square bounding boix should be unique for each superposition, as the center of mass movesd
        resized_image = self.processImage(noisy_image, SquareSide, SquareX, SquareY)
        return resized_image

    # Error/Noise functions:
    def add_noise(self,image, noise_variance: float = 0.0):
        '''
        Adds random noise to a copy of the image according to a normal distribution of variance 'noise_variance'.
        Noise Variance defined as a %age of maximal intensity
        '''

        actual_variance = np.abs(np.random.normal(0, noise_variance)) 
        # Noise Variance parameter gives maximum noise level for whole dataset
        # Actual Noise is the gaussian noise variance used for a specific add_noise call

        max_val = np.max(image)
        return np.random.normal(loc=image, scale=actual_variance*max_val) # Variance then scaled as fraction of brightest intensity

    def add_exposure(self, image, exposure:tuple = (0.0, 1.0)):
        '''
        Adds in exposure limits to the image, using percentile limits defined by exposure.
        exposure[0] is the x% lower limit of detection, exposure[1] is the upper.
        Percents calculated as a function of the maximum image intensity.
        '''
        max_val = np.max(image)
        lower_bound = max_val * exposure[0]
        upper_bound = max_val * exposure[1]
        image = np.clip(image, lower_bound, upper_bound)
        image -= lower_bound
        return image
    
    def quantize_image(self, image, bits):
        '''
        Quantize the image, so that only 2**bits - 1 evenly spaced values possible
        '''
        if bits:
            bins = self.raw_bins * np.max(image)
            quantized_image = np.digitize(image, bins)
            return quantized_image
        else:
            return image
    
    def blur_image(self, image, blur_variance:float=0):
        blur_amount = np.random.normal(0, self.blur_variance)**2
        return gaussian(image, blur_amount)
    
    def rotate_image(self, image, angle):
        '''
        Rotate image by [angle] radians
        '''
        rows, cols = image.shape

        M = cv2.getRotationMatrix2D((cols/2,rows/2), 360*(angle/(2*np.pi)), 1)
        rotated_img = cv2.warpAffine(image, M, (cols,rows))
        return rotated_img
    
    def add_rotational_error(self, image, rotational_variance):
        '''
        Rotate the image by a random amount according to rotational_variance
        '''
        angle = np.random.normal(0, rotational_variance)
        rotated_image = self.rotate_image(image, angle)
        return rotated_image
    
    def add_random_stretch(self, image, stretch_variance):
        '''
        Stretch the image randomly in a random direction, according to stretch_variance
        '''
        stretch_factor = np.abs(np.random.normal(1, stretch_variance))
        angle = np.random.uniform(0, 2*np.pi)
        rotated_im = self.rotate_image(image, angle)
        stretched_dims = (rotated_im.shape[0], int(rotated_im.shape[1]*stretch_factor))
        stretched_im = cv2.resize(rotated_im, stretched_dims, interpolation=cv2.INTER_CUBIC)
        restored_im = self.rotate_image(stretched_im, -angle)
        return restored_im

if __name__ == "__main__":
    camera = get_cams('poor_exposure')
    mode_processor = ModeProcessor(camera)
    s = Superposition(Hermite(1, 1), Laguerre(3, 3))
    img = s.superpose()
    minimum = np.min(img)
    maximum = np.max(img)
    plt.imshow(mode_processor.getImage(img))
    plt.show()
    
