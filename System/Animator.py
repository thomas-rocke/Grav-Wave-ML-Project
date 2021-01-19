import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.image import AxesImage
import numpy as np

class ImageAnimator():
    '''
    Class used with matplotlib to animate image data
    '''
    def __init__(self, axis_shape, image_shape=(128, 128), framerate=30):
        '''
        Initialize class
        '''
        self.framerate=framerate
        self.axes = []
        self.animation_length = 0
        self.axis_shape = axis_shape
        self.ims = np.empty((axis_shape), dtype=AxesImage)

        blank_image = np.zeros(image_shape)
        if type(axis_shape) is not int:
            self.fig, self.axis = plt.subplots(nrows=axis_shape[0], ncols=axis_shape[1]) # create empty figure and axis
            for i in range(axis_shape[0]):
                for j in range(axis_shape[1]):
                    self.ims[i, j] = self.axis[i, j].imshow(blank_image, animated=True) # Create images on each subplot
        else:
            self.fig, self.axis = plt.subplots(nrows=axis_shape) # create empty figure and axis
            for i in range(axis_shape):
                self.ims[i] = self.axis[i].imshow(blank_image, animated=True) # Create images on each subplot
        
        self.data = []
        self.current_frame = 0

    def addImage(self, im_array):
        '''
        Adds im image data as the next animation frame
        im_array should be the same shape as the axis_shape defined in the initialiser
        '''
        self.data.append(im_array)
        self.animation_length += 1
    
    def _updateAni(self, *args):
        '''
        Update script called between animation frames.
        '''
        if type(self.axis_shape) is not int:
            for i in range(self.axis_shape[0]):
                for j in range(self.axis_shape[1]):
                    self.ims[i, j].set_array(self.data[self.current_frame][i, j]) # Update each image with current frame data
        else:
            for i in range(self.axis_shape):
                self.ims[i].set_array(self.data[self.current_frame][i]) # Update each image with current frame data
        self.current_frame += 1
        self.current_frame % self.animation_length # Wrap back to first frame on animation completion
        return self.axis.get_children()

    def show(self):
        '''
        Shows animation
        '''
        FuncAnimation(self.fig, self._updateAni, interval=int(1/self.framerate), blit=True)
        plt.show()



from Gaussian_Beam import Hermite, Superposition

s = Superposition(Hermite(0, 0), Hermite(0, 1))
Ani = ImageAnimator((2), framerate=60)
for i in range(100):
    s[0].add_phase(0.1)
    im = s.superpose()
    phase = s.phase_map()
    Ani.addImage([im, phase])
Ani.show()