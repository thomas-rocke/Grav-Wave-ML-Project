import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.filters import threshold_otsu, gaussian
from Gaussian_Beam import Superposition, Hermite
from multiprocessing import cpu_count, Pool
import matplotlib.pyplot as plt
import time
from Utils import meanError

class BaseProcessor(list):
    def __init__(self, target_resolution:tuple = (128, 128), frames_per_reset:int = 10):
        self.frames_processed = 0
        self.target_resolution = target_resolution
        self.frames_per_reset = frames_per_reset
        self.SquareX = 0
        self.SquareY = 0
        self.SquareScale = 0 # Set Bounding box defaults

    def toGreyscale(self, image):
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
        least_square_vals = [self._widthModel(side_length, image) for side_length in test_sides]
        self.SquareSide = test_sides[np.argmax(least_square_vals)]
        return self.SquareSide
    
    def _widthModel(self, SquareSide, image):
        # Model used in maximisation problem to find bounding box
        x_start = int(image.shape[0]/2 + self.SquareX - SquareSide/2)
        y_start = int(image.shape[1]/2 + self.SquareY - SquareSide/2)
        x_end = int(min(x_start + SquareSide, image.shape[0]))
        y_end = int(min(y_start + SquareSide, image.shape[1]))
        mean_square = (image[x_start:x_end, y_start:y_end]**2).mean(axis=None)
        Square_area = SquareSide**2
        return (mean_square - 1e-5*(Square_area/image.size))

    def makeSquare(self, image, SquareSide=0):
        '''
        Crops the target greyscale image so the aspect ratio is the square given by self.SquareX, self.SquareY and SquareSide (defaulting to self.SquareSide). Image pixels outside the original image default to 0
        '''
        if SquareSide == 0:
            SquareSide = self.SquareSide
        new_image = np.zeros((SquareSide, SquareSide))
        x_start = int(image.shape[0]/2 + self.SquareX - self.SquareSide/2)
        y_start = int(image.shape[1]/2 + self.SquareY - self.SquareSide/2)
        for i in range(SquareSide):
            for j in range(SquareSide):
                x = i + x_start
                y = j + y_start
                if x > 0 and y > 0:
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

    def processImage(self, image):
        grey_image = self.toGreyscale(image)
        squared_image = self.makeSquare(grey_image)
        rezzed_image = self.changeResolution(squared_image)
        normed_image = self.normalise(rezzed_image)
        return normed_image

    def getImages(self, batch_size:int = 1):
        '''
        Perform all operations to generate an image usable by Neural Net, and return a batch of batch_size images
        '''
        images = [self[self.frames_processed + i] for i in range(batch_size)]
        self._resetSquare(self.ToGreyscale(images[0])) # Resets the size of the bounding box based on the first image of the batch
        processed_images = [self.processImage(image) for image in images]
        self.frames_processed += batch_size
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


class ModeProcessor(BaseProcessor):
    def __init__(self, camera_params:dict = {}, model_params:dict = {}):
        camera_keys = camera_params.keys()
        model_keys = model_params.keys()

        # Set up camera properties
        # Assume ideal camera if property not defined
        if 'noise_variance' in camera_keys:
            self.noise_variance = camera_params['noise_variance']
        else:
            self.noise_variance = 0
        
        if 'exposure_limits' in camera_keys:
            self.exposure_limits = camera_params['exposure_limits']
        else:
            self.exposure_limits = (0, 1)
        
        if 'bit_depth' in camera_keys:
            self.bit_depth = camera_params['bit_depth']
        else:
            self.bit_depth = 0
        
        if 'blur_variance' in camera_keys:
            self.blur_variance = camera_params['blur_variance']
        else:
            self.blur_variance = 0
        
        # Set up properties from the model
        if 'resolution' in model_keys:
            self.resolution = model_params['resolution']
        else:
            self.resolution = (128, 128)

        super().__init__(self.resolution)
        self.expose = np.vectorize(self._exposure_comparison) # Create function to handle exposure

    def processImage(self, raw_image):
        '''
        Performs all image processing on target image, using params from class init
        '''
        #shifted_image = shift_image(image,) # Shift the image in x and y coords
        noisy_image = self.add_noise(image, self.noise_variance) # Add Gaussian Noise to the image
        blurred_image = self.blur_image(noisy_image, self.blur_variance)
        exposed_image = self.add_exposure(blurred_image, self.exposure_limits) # Add exposure

        if self.bit_depth: # Bits > 0 therefore quantize
            quantized_image = self.quantize_image(exposed_image, self.bit_depth)
            return quantized_image
        else:
            return exposed_image

    # Error/Noise functions:
    def randomise_amp_and_phase(self, mode):
        '''
        Randomise the amplitude and phase of mode according to normal distributions of self.amplitude_variation and self.phase_variation width.
        Returns new mode with randomised amp and phase.
        '''
        x = mode.copy()
        x *= np.random.random() # Change amp by random amount
        x.add_phase(np.random.random() * 2 * np.pi) # Add random amount of phase
        return x

    def vary_w_0(self, modes, w_0_variance):
        '''
        Varies w_0 param for all modes within a superposition
        '''
        new_w_0 = np.random.normal(modes[0].w_0, w_0_variance)
        new_modes = [mode.copy() for mode in modes]
        for m in new_modes:
            m.w_0 = new_w_0
        return new_modes

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
        image = self.expose(image, upper_bound, lower_bound)
        return image

    def _exposure_comparison(self, val, upper_bound, lower_bound):
        if val > upper_bound:
            val = upper_bound
        elif val < lower_bound:
            val = lower_bound
        return val

    def shift_image(self, image, max_pixel_shift):
        '''
        Will translate target image in both x and y by integer resolution by random numbers in the range (-max_pixel_shift, max_pixel_shift)
        '''
        copy = np.zeros_like(image)
        x_shift = random.randint(-max_pixel_shift, max_pixel_shift)
        y_shift = random.randint(-max_pixel_shift, max_pixel_shift)
        shape = np.shape(image)
        for i in range(shape[0]):
            for j in range(shape[1]):
                new_coords = [i + x_shift, j + y_shift]
                if new_coords[0] in range(shape[0]) and new_coords[1] in range(shape[1]): # New coordinates still within image bounds
                    copy[i, j] = image[new_coords[0], new_coords[1]]
                else:
                    copy[i, j] = 0
        return copy
    
    def quantize_image(self, image, bits):
        '''
        Quantize the image, so that only 255 evenly spaced values possible
        '''
        max_val = np.max(image)
        vals = 2**bits - 1
        bins = np.linspace(0, max_val, vals, endpoint=1)
        quantized_image = np.digitize(image, bins)
        return quantized_image
    
    def blur_image(self, image):
        blur_amount = np.random.normal(0, self.blur_variance)**2
        return gaussian(image, blur_amount)



if __name__ == "__main__":
    fname = r'C:\Users\Tom\Downloads\EditedBeamModes.mp4'
    vid = VideoProcessor(fname, frames_per_reset=1)
    
    fig, ax = plt.subplots(ncols=5)
    ims = vid.getImages(5)
    for i in range(5):
        ax[i].imshow(ims[i])
    
    plt.show()