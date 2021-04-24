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

def real_data_stability(model, video_file):
    plt.title("MSE difference between Real frame and reconstructed Prediction ")
    processor = VideoProcessor(video_file, model.data_generator.mode_processor.target_resolution)
    predictions = np.zeros((processor.frameCount))

    for i in tqdm(range(processor.frameCount), desc=pathlib.PurePath(video_file).name):
        processed_frame = processor.getImages(batch_size=1)[0] # Get and process next frame
        predicted = model.predict(processed_frame, threshold=0, info=False).superpose()
        predicted_image = processor.processImage(predicted, *processor._resetSquare(predicted))
        predictions[i] = np.sum((predicted_image - processed_frame)**2)

    plt.scatter(range(processor.frameCount), predictions)
    plt.xlabel("Frame Number")
    plt.ylabel("MSE error")
    plt.show()

def format_func(value, tick_num):
    return "{}$\sigma$".format(value)


def mode_sweep_test(model, its):
    while model.data_generator.new_stage(): pass
    sup = model.data_generator.get_random()

    model.get_errs_of_model(n_test_points=its)
    errs = model.errs
    hermite_modes = model.data_generator.hermite_modes
    vec = np.array([sup.contains(j).amplitude for j in hermite_modes] + [sup.contains(j).phase for j in hermite_modes])
    ln = len(vec)

    fig, ax = plt.subplots(nrows=2)

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
        
        diff_amps = [(true_amplitudes[i] - pred_amps[i]) for i in range(len(pred_amps))]
        diff_phases = [0]*len(pred_phases)

        for i in range(len(pred_phases)):
            phase_errs = [(true_phases[i] - 2*np.pi - pred_phases[i]), (true_phases[i] - pred_phases[i]), (true_phases[i] + 2*np.pi - pred_phases[i])]
            phase_diff = phase_errs[np.argmin(np.abs(phase_errs))] # Account for phase wrapping massively changing the error
            diff_phases[i] = phase_diff if not np.isnan(phase_diff) else 0

        diffs = np.array(diff_amps + diff_phases)
        dat = diffs/errs
        data[step, :] = np.array([d if not np.isnan(d) else 0 for d in dat])

    max_amp = int(np.max(np.abs(data[:, :ln//2]))) + 1
    max_phase = int(np.max(np.abs(data[:, ln//2:]))) + 1


    amp_bins = 4*max_amp
    phase_bins = 4*max_phase

    max_amp_freq = 0
    max_phase_freq = 0

    for i, mode in enumerate(hermite_modes):
        amp_freqs, _, __ = ax[0].hist(data[:, i], amp_bins, label=mode.latex_print(), histtype="step", align="mid")
        phase_freqs, _, __ = ax[1].hist(data[:, (ln//2) + i], phase_bins, label=mode.latex_print(), histtype="step", align="mid")

        max_amp_freq = max([max_amp_freq, amp_freqs.max()])
        max_phase_freq = max([max_phase_freq, phase_freqs.max()])
    
    ax[0].set_title("Amplitude Error Distribution")
    ax[1].set_title("Phase Error Distribution")
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel("Relative Frequency")
    ax[0].set_xlabel("Amplitude Error")
    ax[1].set_xlabel("Phase Error")

    ax[0].fill_betweenx([0, max_amp_freq], -1, 1, facecolor="green", alpha=0.3)
    ax[0].fill_betweenx([0, max_amp_freq], 1, 2, facecolor="yellow", alpha=0.3)
    ax[0].fill_betweenx([0, max_amp_freq], -1, -2, facecolor="yellow", alpha=0.3)
    ax[1].fill_betweenx([0, max_phase_freq], -1, 1, facecolor="green", alpha=0.3)
    ax[1].fill_betweenx([0, max_phase_freq], 1, 2, facecolor="yellow", alpha=0.3)
    ax[1].fill_betweenx([0, max_phase_freq], -1, -2, facecolor="yellow", alpha=0.3)

    ax[0].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax[1].xaxis.set_major_formatter(plt.FuncFormatter(format_func))



    plt.show()

if __name__ == '__main__':

    model = ML(BasicGenerator(4, 4, 0.5, 1.0, 0.1, (0.0, 1.0), 32, 32, 64, 1, False), 'VGG16', 'Adam', 0.0001, False)
    model.load()
    while model.data_generator.new_stage(): pass
    model.data_generator.mode_processor.target_resolution = (64, 64)
    #fname = r"C:\Users\Tom\Documents\GitHub\Grav-Wave-ML-Project\Cavity\Edited.mp4"
    #real_data_stability(model, fname)
    mode_sweep_test(model, 100)
    
    
