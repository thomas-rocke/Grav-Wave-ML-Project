import numpy as np
import matplotlib.pyplot as plt
from ImageProcessing import VideoProcessor, BaseProcessor, ModeProcessor
from ML_Identification import ML
from DataHandling import Dataset, BasicGenerator
from Gaussian_Beam import Hermite, Superposition
import Logger
import os
LOG = Logger.get_logger(__name__)

def visualise_video_predictions(video_file: str, model:ML):
    resolution = model.data_generator.mode_processor.target_resolution
    fig, ax = plt.subplots(ncols=2)
    ax[0].set_title("Processed input image")
    ax[1].set_title("Image reconstruction of predictions")
    processor = VideoProcessor(video_file)
    predictions = np.zeros((resolution[0], resolution[1], processor.frameCount))
    for i in range(processor.frameCount):
        img = processor[i]
        if i == 0:
            params = processor.get_bounding_box(processor.toGreyscale(img/np.linalg.norm(img)))
        processed_frame = processor.processImage(img, *params) # Get and process next frame
        predicted_image = model.predict(processed_frame).superpose()
        predictions[:,  :, i] = predicted_image

        ax[0].imshow(processed_frame)
        ax[1].imshow(predicted_image)
        plt.pause(0.05)
        #LOG.warning("Unexpected EOF")
    plt.show()
    return predictions

def mode_sweep_test(model, modes, freqs, iterations):
    fig, ax = plt.subplots(ncols=2)
    ax[0].set_title("Processed input image")
    ax[1].set_title("Image reconstruction of predictions")
    processor = ModeProcessor()
    for step in range(iterations):
        for i, mode in enumerate(modes):
            mode.amplitude = np.sin(freqs[i] * step)
            mode.phase = 2 * np.pi * np.cos(freqs[i] * step)
        img = Superposition(*modes).superpose()
        processed_frame = processor.getImage(img) # Get and process next frame
        predicted_image = model.predict(processed_frame).superpose()
        ax[0].imshow(processed_frame)
        ax[1].imshow(predicted_image)
        plt.pause(0.05)
    plt.show()

#os.chdir("System")
fname = r"C:\Users\Tom\Documents\EditedBeamModes.mp4"
model = ML(BasicGenerator(3, 3, 0.5, 0, 0.1, (0.0, 1.0), 64, 64, 64, False, 1), 'Adamax', 0.0001, False)
model.load()

model.data_generator.new_stage()
model.compare(model.data_generator.get_random())

#dat = visualise_video_predictions(fname, model)
#mode_sweep_test(model, [Hermite(0, 0), Hermite(0, 1), Hermite(1, 1), Hermite(1, 0)], [1, 1.5, 0.25, 1/3], 20)