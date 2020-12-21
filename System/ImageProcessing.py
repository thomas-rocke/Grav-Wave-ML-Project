import numpy as np
import cv2

class BaseProcessor(list):
    def __init__(self, target_resolution:tuple = (128, 128)):
        self.frames_processed = 0
        self.target_resolution = target_resolution

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
        centered_image = self.Recenter(grey_image)
        scaled_image = self.Rescale(centered_image)
        rezzed_image = self.ChangeResolution(scaled_image)
        normed_image = self.Normalise(rezzed_image)
        self.frames_processed += 1
        return normed_image


class VideoProcessor(BaseProcessor):

    def __init__(self, video_file, target_resolution:tuple = (128*128)):
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

        fc = 0
        ret = True

        while (fc < frameCount  and ret):
            ret, buf[fc] = cap.read()
            fc += 1

        cap.release()

        self.video = buf
        super().__init__(target_resolution)

    def __getitem__(self, i):
        return self.video[i, :, :, :]

