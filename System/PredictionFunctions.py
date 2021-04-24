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
from collections import deque
from Utils import meanError
import time
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

def get_rot_matrix(length, theta):
    rot = np.diagflat([1]*length).astype(float)
    rot[0, 0] = np.sin(theta)
    rot[1, 0] = np.cos(theta)
    rot[0, 1] = np.cos(theta)
    rot[1, 1] = -np.sin(theta)
    
    
    off = np.zeros((length, length))   
    ls = deque(range(length))
    for i in range(length):
        off[i, :] = ls
        ls.rotate(1)
    
    off /= np.linalg.norm(off)
    return np.dot(off, rot)

def mode_sweep_test(model, its):
    model.load()
    while model.data_generator.new_stage(): pass
    sup = model.data_generator.get_random()

    model.get_errs_of_model(n_test_points=100)
    errs = model.errs
    hermite_modes = model.data_generator.hermite_modes
    vec = np.array([sup.contains(j).amplitude for j in hermite_modes] + [sup.contains(j).phase for j in hermite_modes])
    ln = len(vec)

    fig, ax = plt.subplots(nrows=2, sharex=True)

    data = np.zeros((its, ln))

    for step in tqdm(range(its)):
        """ theta = 2*np.pi*freq*i/its
        rot = get_rot_matrix(ln, theta)
        new_vec = np.dot(rot, vec)

        for j, mode in enumerate(sup):
            mode.amplitude = np.abs(new_vec[j])
            mode.phase = new_vec[(ln//2) + j]
        
        sup.renormalize() """
        sup = model.data_generator.get_random()

        true_amplitudes = [sup.contains(j).amplitude for j in model.data_generator.hermite_modes]
        true_phases = [sup.contains(j).phase for j in model.data_generator.hermite_modes]
        true_phases = [phase if (phase !=-10) else 0 for phase in true_phases]
        
        img = sup.superpose()#model.data_generator.mode_processor.getImage(sup.superpose())
        pred = model.predict(img, threshold=0, info=False)
        pred_amps = [pred.contains(j).amplitude for j in model.data_generator.hermite_modes]
        pred_phases = [pred.contains(j).phase for j in model.data_generator.hermite_modes]
        
        diff_amps = [np.abs(true_amplitudes[i] - pred_amps[i]) for i in range(len(pred_amps))]
        diff_phases = [0]*len(pred_phases)

        for i in range(len(pred_phases)):
            phase_diff = np.min([(true_phases[i] - 2*np.pi - pred_phases[i])**2, (true_phases[i] - pred_phases[i])**2, (true_phases[i] + 2*np.pi - pred_phases[i])**2]) # Account for phase wrapping massively changing the error
            diff_phases[i] = np.sqrt(phase_diff) if not np.isnan(phase_diff) else 0

        diffs = np.array(diff_amps + diff_phases)
        dat = diffs/errs
        data[step, :] = np.array([d if not np.isnan(d) else 0 for d in dat])

    for i, mode in enumerate(hermite_modes):
        ax[0].scatter(range(its), data[:, i], label=mode.latex_print())
        ax[1].scatter(range(its),  data[:, (ln//2) + i], label=mode.latex_print())
    
    ax[0].set_title("Amplitude errors over time")
    ax[1].set_title("Phase errors over time")
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel("Absolute Amplitude Error")
    ax[1].set_xlabel("Timestep")
    ax[1].set_ylabel("Absolute Phase Error")

    ax[0].fill_between(range(its), 0, 1, facecolor="green", alpha=0.3)
    ax[0].fill_between(range(its), 1, 2, facecolor="yellow", alpha=0.3)
    ax[1].fill_between(range(its), 0, 1, facecolor="green", alpha=0.3)
    ax[1].fill_between(range(its), 1, 2, facecolor="yellow", alpha=0.3)

    max_amp = int(np.max(np.abs(data[:, :ln//2]))) + 1
    max_phase = int(np.max(np.abs(data[:, ln//2:]))) + 1

    ax[0].set_ylim(0, max_amp)
    ax[1].set_ylim(0, max_phase)

    amp_labels = ["{}$\sigma$".format(val) for val in range(max_amp + 1)]
    phase_labels = ["{}$\sigma$".format(val) for val in range(max_phase + 1)]

    ax[0].set_yticks(range(max_amp + 1))
    ax[0].set_yticklabels(amp_labels)
    ax[1].set_yticks(range(max_phase + 1))
    ax[1].set_yticklabels(phase_labels)
    ax[1].set_xlim(0, its)



    plt.show()

if __name__ == '__main__':

    model = ML(BasicGenerator(4, 4, 0.5, 1.0, 0.1, (0.0, 1.0), 32, 32, 64, 1, False), 'VGG16', 'Adam', 0.0001, False)
    model.load()
    while model.data_generator.new_stage(): pass
    model.data_generator.mode_processor.target_resolution = (64, 64)

    mode_sweep_test(model, 100)
    
    
