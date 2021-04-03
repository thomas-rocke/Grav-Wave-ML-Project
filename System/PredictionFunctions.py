import numpy as np
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from ImageProcessing import VideoProcessor, BaseProcessor, ModeProcessor
from ML_Identification import ML
from Old_Data_Generators import Dataset, BasicGenerator
from Gaussian_Beam import Hermite, Superposition
import Logger
import os
LOG = Logger.get_logger(__name__)

def visualise_video_predictions(video_file: str, model:ML):
    resolution = (64, 64)#model.data_generator.mode_processor.target_resolution
    fig, ax = plt.subplots(ncols=2)
    ax[0].set_title("Processed Input Image")
    ax[1].set_title("Image Reconstruction")
    ax[0].axis('off')
    ax[1].axis('off')
    processor = VideoProcessor(video_file, resolution)
    predictions = np.zeros((resolution[0], resolution[1], processor.frameCount))

    frames = []
    for i in tqdm(range(processor.frameCount - 10), desc=pathlib.PurePath(video_file).name):
        processed_frame = processor.getImages(batch_size=1)[0] # Get and process next frame
        predicted = model.predict(processed_frame, info=False)
        predicted_image = predicted.superpose()
        predictions[:,  :, i] = predicted_image

        f1 = ax[0].imshow(processed_frame, cmap='jet', animated=True)
        ax[1].set_xlabel(predicted.latex_print())
        f2 = ax[1].imshow(predicted_image, cmap='jet', animated=True)
        frames.append([f1, f2])
        plt.pause(1e-9)

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    os.makedirs(f"Animations/{model}", exist_ok=True) # Create directory for model
    ani.save(f'Animations/{model}/{pathlib.PurePath(video_file).name[:-4]}.gif', writer=animation.PillowWriter(fps=30))
    plt.show()
    return predictions

def mode_sweep_test(model, modes, freqs, iterations):
    fig, ax = plt.subplots(ncols=2)
    ax[0].set_title("Processed Input Image")
    ax[1].set_title("Image Reconstruction")
    processor = ModeProcessor()

    frames = []
    for step in range(iterations):
        for i, mode in enumerate(modes):
            mode.amplitude = np.sin(freqs[i] * step)
            mode.phase = 2 * np.pi * np.cos(freqs[i] * step)
        img = Superposition(*modes).superpose()
        processed_frame = processor.getImage(img) # Get and process next frame
        predicted_image = model.predict(processed_frame).superpose()

        f1 = ax[0].imshow(processed_frame, cmap='jet', animated=True)
        f2 = ax[1].imshow(predicted_image, cmap='jet', animated=True)
        frames.append([f1, f2])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    ani.save('movie.mp4')
    plt.show()

if __name__ == '__main__':
    model = ML(BasicGenerator(3, 3, 0.2, 0.4, 0.1, (0.0, 1.0), 64, 64, 64, 1, False), 'default', 'Adamax', 0.0001, False)
    model.load()
    visualise_video_predictions(r'C:\Users\Tom\Documents\GitHub\Grav-Wave-ML-Project\Cavity\video.mov', model)



model = ML(BasicGenerator(3, 3, 1.0, 2.0, 0.1, (0.0, 0.6), 64, 64, 64, 1, False), 'VGG16', 'Adam', 0.0001, False) # ML(BasicGenerator(3, 3, 0.5, 1.0, 0.1, (0.0, 1.0), 64, 64, 64, False, 1), 'Adamax', 0.0001, False)
model.train()
model.save()
model.load()
model.evaluate()

while model.data_generator.new_stage(): pass

dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video2.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited2.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\LIGO2.mp4", model)

model = ML(BasicGenerator(5, 5, 1.0, 2.0, 0.1, (0.0, 0.6), 64, 64, 64, 1, False), 'VGG16', 'Adam', 0.0001, False) # ML(BasicGenerator(3, 3, 0.5, 1.0, 0.1, (0.0, 1.0), 64, 64, 64, False, 1), 'Adamax', 0.0001, False)
model.train()
model.save()
model.load()
model.evaluate()

# while model.data_generator.new_stage(): pass

dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video2.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited2.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\LIGO2.mp4", model)

model = ML(BasicGenerator(3, 9, 0.5, 1.0, 0.1, (0.0, 1.0), 64, 64, 64, 1, False), 'VGG16', 'Adam', 0.0001, False) # ML(BasicGenerator(3, 3, 0.5, 1.0, 0.1, (0.0, 1.0), 64, 64, 64, False, 1), 'Adamax', 0.0001, False)
model.train()
model.save()
model.load()

while model.data_generator.new_stage(): pass

dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video2.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Test.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited2.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\LIGO2.mp4", model)

model = ML(BasicGenerator(3, 3, 0.5, 1.0, 0.1, (0.0, 1.0), 64, 64, 64, 1, False), 'VGG16', 'Adam', 0.0001, False) # ML(BasicGenerator(3, 3, 0.5, 1.0, 0.1, (0.0, 1.0), 64, 64, 64, False, 1), 'Adamax', 0.0001, False)
model.train()
model.save()
model.load()

while model.data_generator.new_stage(): pass

dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video2.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited2.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\LIGO2.mp4", model)

model = ML(BasicGenerator(3, 3, 1.0, 2.0, 0.1, (0.0, 1.0), 64, 64, 64, 1, False), 'VGG16', 'Adam', 0.0001, False) # ML(BasicGenerator(3, 3, 0.5, 1.0, 0.1, (0.0, 1.0), 64, 64, 64, False, 1), 'Adamax', 0.0001, False)
model.train()
model.save()
model.load()

while model.data_generator.new_stage(): pass

dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video2.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited2.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\LIGO2.mp4", model)

model = ML(BasicGenerator(3, 3, 0.2, 0.4, 0.1, (0.0, 1.0), 64, 64, 64, 1, False), 'VGG16', 'Adam', 0.0001, False) # ML(BasicGenerator(3, 3, 0.5, 1.0, 0.1, (0.0, 1.0), 64, 64, 64, False, 1), 'Adamax', 0.0001, False)
model.train()
model.save()
model.load()

while model.data_generator.new_stage(): pass

dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video2.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited2.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\LIGO2.mp4", model)

model = ML(BasicGenerator(5, 5, 0.5, 1.0, 0.1, (0.0, 1.0), 8, 32, 64, 1, False), 'default', 'Adam', 0.0001, False) # ML(BasicGenerator(3, 3, 0.5, 1.0, 0.1, (0.0, 1.0), 64, 64, 64, False, 1), 'Adamax', 0.0001, False)
model.train()
model.save()
model.load()

while model.data_generator.new_stage(): pass

dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Video2.mov", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\Edited2.mp4", model)
# dat = visualise_video_predictions(r"C:\Users\Jake\OneDrive - University of Birmingham\Uni\Year 4\Project\Grav-Wave-ML-Project\Cavity\LIGO2.mp4", model)



# model.get_errs_of_model()
# for i in range(1):
#     model.compare(model.data_generator.get_random())
# print(model.errs)
