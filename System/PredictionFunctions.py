import numpy as np
import matplotlib.pyplot as plt
from ImageProcessing import VideoProcessor
from ML_Identification import ML
from DataHandling import Dataset, BasicGenerator
import Logger
LOG = Logger.get_logger(__name__)

def visualise_video_predictions(video_file: str, model:ML):
    resolution = model.data_generator.mode_processor.resolution
    fig, ax = plt.subplots(ncols=2)
    ax[0].set_title("Processed input image")
    ax[1].set_title("Image reconstruction of predictions")
    processor = VideoProcessor(video_file)
    predictions = np.zeros((resolution[1], resolution[2], processor.frameCount))
    for i in range(processor.frameCount):
        try:
            processed_frame = processor[i] # Get and process next frame
            predicted_image = model.predict(processed_frame).superpose()
            predictions[:,  :, i] = predicted_image

            ax[0].imshow(processed_frame)
            ax[1].imshow(predicted_image)
            plt.pause(0.05)
        except:
            LOG.warn("Unexpected EOF")
    return predictions


fname = r"C:\Users\Tom\Documents\EditedBeamModes.mp4"
ds = BasicGenerator(batch_size=64, amplitude_variation=0.2, phase_variation=0.2)
model = ML(data_generator=ds)
model.load()

dat = visualise_video_predictions(fname, model)
plt.show()