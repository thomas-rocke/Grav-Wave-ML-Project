import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.filters import threshold_otsu, gaussian
from Gaussian_Beam import Superposition, Hermite, Laguerre
from multiprocessing import cpu_count, Pool
import matplotlib.pyplot as plt
import time
from Utils import meanError

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
        test_sides = np.arange(0, max_sidelength)
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
        return (mean_square - (SquareSide/max(image.shape)))

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
        images = [self[self.frames_processed + i] for i in range(batch_size)]
        SquareSide, SquareX, SquareY = self._resetSquare(self.toGreyscale(images[0])) # Resets the size of the bounding box based on the first image of the batch
        processed_images = [self.processImage(image, SquareSide, SquareX, SquareY) for image in images]
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
    def __init__(self, camera:dict = {}, target_resolution:tuple=(128, 128)):
        self.change_camera(camera)

        super().__init__(target_resolution)
        self.expose = np.vectorize(self._exposure_comparison) # Create function to handle exposure

    def change_camera(self, camera:dict):
        camera_keys = camera.keys()

        # Set up camera properties
        # Assume ideal camera if property not defined
        if 'noise_variance' in camera_keys:
            self.noise_variance = camera['noise_variance']
        else:
            self.noise_variance = 0
        
        if 'exposure_limits' in camera_keys:
            self.exposure_limits = camera['exposure_limits']
        else:
            self.exposure_limits = (0, 1)
        
        if 'bit_depth' in camera_keys:
            self.bit_depth = camera['bit_depth']
        else:
            self.bit_depth = 0
        
        if 'blur_variance' in camera_keys:
            self.blur_variance = camera['blur_variance']
        else:
            self.blur_variance = 0

    def errorEffects(self, raw_image):
        '''
        Performs all image processing for noise effects on target image, using params from class init
        '''
        #shifted_image = shift_image(image,) # Shift the image in x and y coords
        noisy_image = self.add_noise(raw_image, self.noise_variance) # Add Gaussian Noise to the image
        blurred_image = self.blur_image(noisy_image, self.blur_variance)
        exposed_image = self.add_exposure(blurred_image, self.exposure_limits) # Add exposure

        if self.bit_depth: # Bits > 0 therefore quantize
            quantized_image = self.quantize_image(exposed_image, self.bit_depth)
            return quantized_image
        else:
            return exposed_image

    def getImage(self, raw_image):
        '''
        Perform all processing on target superposition image to preprare it for training.
        '''
        noisy_image = self.errorEffects(raw_image)
        SquareSide, SquareX, SquareY = self._resetSquare(noisy_image) # Relocation of the square bounding boix should be unique for each superposition, as the center of mass movesd
        resized_image = self.processImage(noisy_image, SquareSide, SquareX, SquareY)
        return resized_image

    # Error/Noise functions:
    def randomise_amp_and_phase(self, mode):
        '''
        Randomise the amplitude and phase of mode according to normal distributions of self.amplitude_variation and self.phase_variation width.
        Returns new mode with randomised amp and phase.
        '''
        x = mode.copy()
        x *= np.random.rand() # Change amp by random amount
        x.add_phase(np.random.rand() * 2 * np.pi) # Add random amount of phase
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
        image -= lower_bound
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
        x_shift = np.random.randint(-max_pixel_shift, max_pixel_shift)
        y_shift = np.random.randint(-max_pixel_shift, max_pixel_shift)
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
    
    def blur_image(self, image, blur_variance:float=0):
        blur_amount = np.random.normal(0, self.blur_variance)**2
        return gaussian(image, blur_amount)

camera_presets = {
    'ideal_camera' : {
        'noise_variance' : 0,
        'exposure_limits' : (0, 1),
        'bit_depth' : 0,
        'blur_variance' : 0
    },

    'poor_noise' : {
        'noise_variance' : 0.4,
        'exposure_limits' : (0, 1),
        'bit_depth' : 0,
        'blur_variance' : 0
    },

    'poor_exposure' : {
        'noise_variance' : 0,
        'exposure_limits' : (0.3, 0.7),
        'bit_depth' : 0,
        'blur_variance' : 0
    },

    'poor_bit_depth' : {
        'noise_variance' : 0,
        'exposure_limits' : (0, 1),
        'bit_depth' : 8,
        'blur_variance' : 0
    },

    'poor_blur' : {
        'noise_variance' : 0,
        'exposure_limits' : (0, 1),
        'bit_depth' : 0,
        'blur_variance' : 0.5
    },
}

if __name__ == "__main__":
    camera = camera_presets['poor_noise']
    mode_processor = ModeProcessor(camera)
    s = Superposition(Hermite(1, 1), Laguerre(3, 3), resolution=480)
    img = s.superpose()
    processed_img = mode_processor.getImage(img)
    fig, ax = plt.subplots(ncols=3)
    ax[0].imshow(img)
    ax[1].imshow(processed_img)
    ax[2].imshow(mode_processor.changeResolution(img) - processed_img)
    plt.show()
    
