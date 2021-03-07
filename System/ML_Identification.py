##################################################
##########                              ##########
##########              ML              ##########
##########        CLASSIFICATION        ##########
##########                              ##########
##################################################

# TODO Header for file

# Imports
from re import ASCII
import sys
import os
import shutil
import gc
import logging
import argparse
import textwrap

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide Tensorflow info, warning and error messages

from Gaussian_Beam import Hermite, Superposition, Laguerre
from DataHandling import GenerateData, BasicGenerator, Dataset
import Logger
from time import perf_counter
from datetime import datetime
from math import isnan
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, Convolution2D, ZeroPadding2D
from keras.constraints import maxnorm
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam
from itertools import combinations, chain
from multiprocessing import Pool, cpu_count
from ImageProcessing import ModeProcessor

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').disabled = True
LOG = Logger.get_logger(__name__)


##################################################
##########                              ##########
##########           CLASSES            ##########
##########                              ##########
##################################################


class Colour:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'




class ML:
    '''
    The class 'ML' that represents a Keras model using datasets from Gaussian modes.
    '''
    def __init__(self,
                 data_generator: keras.utils.Sequence = BasicGenerator(),
                 optimiser: str = "Adamax",
                 learning_rate: float = 0.0001,
                 use_multiprocessing: bool = True):
        '''
        Initialise the class.
        '''
        LOG.info("Initialising ML model.")

        self.data_generator = data_generator
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.use_multiprocessing = use_multiprocessing

        LOG.debug(f"Locals: {locals()}")

        self.max_epochs = 100 # Max epochs before training is terminated
        self.success_loss = 0.001 # Loss at which the training is considered successful
        self.stagnation = 5 # Epochs of stagnation before terminating training stage
        self.history = {"time": [], "stage": [], "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        self.model = None
        self.errs = None

        print(Colour.HEADER + Colour.BOLD + "____________________| " + str(self) + " |____________________\n" + Colour.ENDC)

        LOG.info("ML model initialised!")

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return repr(self)

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + f"({self.data_generator}, '{self.optimiser}', {self.learning_rate}, {self.use_multiprocessing})"

    def copy(self):
        '''
        Copy the model object.
        '''
        return ML(self.data_generator.copy(), self.optimiser, self.learning_rate, self.use_multiprocessing)

    def exists(self):
        '''
        Check if the model exits in the file system.
        '''
        LOG.debug("Checking if ML object exists in the file system.")

        return os.path.exists(f"Models/{self}")

    def trained(self):
        '''
        Check if the model has been trained before.
        '''
        LOG.debug("Checking if ML model has been trained before.")

        return os.path.exists(f"Models/{self}/model.h5")

    def accuracy(self, y_true, y_pred):
        '''
        Custom metric to determine the accuracy of our regression problem using rounded accuracy.
        '''
        mask = K.cast(K.greater_equal(y_true, 0), K.floatx())

        return K.mean(K.equal(K.round(y_true * mask), K.round(y_pred * mask)))

    def loss(self, y_true, y_pred):
        '''
        Custom loss function to mask out modes that don't exist in the superposition.
        '''
        if type(self.data_generator) == Dataset:
            mask = K.cast(K.greater(y_true, 0), K.floatx())
        else:
            mask = K.cast(K.greater_equal(y_true, 0), K.floatx())

        diff = K.abs((y_pred * mask) - (y_true * mask))

        amplitudes = diff * K.constant(np.array([1 if i < len(self.classes) // 2 else 0 for i in range(len(self.classes))]))
        phases = diff * K.constant(np.array([0 if i < len(self.classes) // 2 else 1 for i in range(len(self.classes))]))

        reduced_phases = K.minimum(phases, K.abs(diff - 1))
        loss = K.square(amplitudes + reduced_phases)

        # K.print_tensor(phases)
        # K.print_tensor(reduced_phases)

        return K.mean(loss, axis=-1)

    def create_model(self, summary: bool = True):
        '''
        Create the Keras model in preparation for training.
        '''
        LOG.debug("Creating the architecture of the Keras model.")

        # Conv2D: Matrix that traverses the image and blurs it
        # Dropout: Randomly sets input units to 0 to help prevent overfitting
        # MaxPooling2D: Downsamples the input representation

        # The VGG16 convolutional neural net (CNN) architecture was used to win ILSVR (Imagenet) competition in 2014
        # It is considered to be one of the best vision model architectures to date

        # Our custom architecture:

        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=self.input_shape, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.2))

        model.add(Dense(512, kernel_constraint=maxnorm(3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(128, kernel_constraint=maxnorm(3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(units=len(self.classes)))
        model.add(Activation('sigmoid'))

        LOG.debug("Keras model layers created.")
        LOG.debug("Compiling the model.")

        # model = keras.applications.VGG16(include_top=False, input_shape=self.input_shape, pooling='avg', classes=len(self.classes))
        # model = VGG16(self.input_shape, len(self.classes)) # Override model with VGG16 model
        model.compile(loss=self.loss, optimizer=eval(f"{self.optimiser}(learning_rate={self.learning_rate})"), metrics=[self.accuracy])

        LOG.debug(f"Model compiled. Optimiser: {self.optimiser}(learning_rate={self.learning_rate}).")

        # We choose sigmoid and binary_crossentropy here because we have a multilabel neural network, which becomes K binary classification problems
        # Using softmax would be wrong as it raises the probabiity on one class and lowers others

        if summary: print(model.summary())
        if summary: keras.utils.plot_model(model, str(self), show_shapes=True)

        LOG.debug("Keras model architecture created successfully!")
        return model

    def train(self, info: bool = False):
        '''
        Train the model.
        '''
        LOG.info("Initialising the training process for the model.")

        # Check if the model has already been trained

        if self.trained():
            LOG.warning("Trained model already exists.")
            print(log("[WARN] Trained model already exists!\n"))

            self.load()
            return

        # Timing how long the training process took

        start_time = perf_counter()

        # Create the data generator object for the dataset

        LOG.debug("Creating the data generator object for the dataset.")
        print(log("[INIT] Initialising dataset... "), end='')

        self.classes = self.data_generator.get_classes()
        self.input_shape = (self.classes[0].resolution, self.classes[0].resolution, 1)

        print("Done!")
        LOG.debug(f"Classes: {list(self.classes)}")
        LOG.debug(f"Input shape: {self.input_shape}")

        # Create the Keras model

        LOG.debug("Creating the Keras model.")
        print(log("[INIT] Generating model (input shape = " + str(self.input_shape) + ", classes = " + str(len(self.classes)) + ", optimiser: " + self.optimiser + ")... "), end='')

        self.model = self.create_model(summary=False)

        print("Done!\n")

        # Begin training

        LOG.info("Starting the training process.")
        print(log("[TRAIN] Training..."))
        print(log("[TRAIN] |"))

        # For every stage / complexity of data in the dataset

        while self.data_generator.new_stage():
            LOG.info(f"Stage {self.data_generator.stage} / {self.data_generator.max_stage} of training.")
            print(log(f"[TRAIN] |-> Stage               : {self.data_generator.stage} / {self.data_generator.max_stage}."))
            print(log(f"[TRAIN] |-> Dataset             : {self.data_generator}."))
            print(log(f"[TRAIN] |-> Size                : {len(self.data_generator)}."))
            print(log(f"[TRAIN] |-> Cores               : {cpu_count()}."))
            print(log(f"[TRAIN] |-> Classes             : {' '.join([str(i).replace(' ', '') for i in self.classes])}."))
            print(log(f"[TRAIN] |-> Success Condition   : A loss of {self.success_loss}."))
            print(log(f"[TRAIN] |-> Terminate Condition : Reaching epoch {len(self.history['loss']) + self.max_epochs} or {self.stagnation} consecutive epochs of stagnation."))
            print(log(f"[TRAIN] |"))

            # For every epoch of training on that stage

            n = 0
            try:
                iterator = tqdm(range(self.max_epochs), log("[TRAIN] |-> Training "))
                for n in iterator:
                    # LOG.debug(f"Epoch {n + 1}:")

                    # Fit the model to the stage of the data generator

                    history_callback = self.model.fit_generator(self.data_generator,
                                                                validation_data=self.data_generator,
                                                                validation_steps=1,
                                                                steps_per_epoch=len(self.data_generator),
                                                                max_queue_size=cpu_count(),
                                                                use_multiprocessing=self.use_multiprocessing,
                                                                workers=cpu_count(),
                                                                verbose=int(info))

                    # Save the performance of this epoch

                    for i in self.history:
                        if i == "time": self.history[i].append(perf_counter() - start_time) # Save time elapsed since training began
                        elif i == "stage": self.history[i].append(self.data_generator.stage) # Save what stage the data generator was on for this epoch
                        else: self.history[i].append(history_callback.history[i][0]) # Save performance of epoch

                        # LOG.debug(f"{i.replace('_', ' ').title()}: {self.history[i][-1]}")

                    # Compute the stagnation of the training run

                    stagnates = len(np.where(np.round(self.history['loss'][-1], 4) >= np.round(self.history['loss'][-min(n + 1, self.stagnation + 1):-1], 4))[0])
                    if stagnates == 0: indicator = Colour.OKGREEN + '++' + Colour.ENDC
                    else: indicator = (Colour.WARNING if stagnates < self.stagnation - 1 else Colour.FAIL) + '-' + str(stagnates) + Colour.ENDC

                    LOG.debug(f"Epoch: {n + 1} - Time: {self.history['time'][-1] :.2f}s - Loss: {self.history['loss'][-1] :.4f} - Accuracy: {self.history['accuracy'][-1] :.4f} - Val Loss: {self.history['val_loss'][-1] :.4f} - Val Accuracy: {self.history['val_accuracy'][-1] :.4f} - Stagnation: {stagnates}")

                    # Update the loading bar description with the current losses

                    iterator.set_description(log(f"[TRAIN] |-> {indicator} Loss: {self.history['loss'][-1] :.4f} - Val Loss: {self.history['val_loss'][-1] :.4f} "))

                    # Check if gradient descent has diverged so training has failed

                    if isnan(self.history['loss'][-1]): # Loss is nan so training has failed
                        LOG.critical("Loss is 'nan' so training has diverged and failed.")
                        print(log("\n[TRAIN] V "))
                        print(log("[FATAL] Training failed! Gradient descent diverged at epoch " + str(len(self.history['loss'])) + ".\n"))

                        sys.exit()

                    # Check if loss has reached an acceptable level

                    elif self.history['loss'][-1] < self.success_loss: # Loss has reached success level
                        LOG.debug("Loss has reached success level. Will break from training stage.")
                        iterator.close()

                        print(log("[TRAIN] |"))
                        print(log("[TRAIN] |-> " + str(self.success_loss) + " loss achieved at epoch " + str(len(self.history['loss'])) + "."))

                        break

                    # Check if learning has stagnated

                    elif stagnates >= self.stagnation: # Training has stagnated so is not learning anymore
                        LOG.warning("Training has stagnated, so model is no longer learning.")
                        iterator.close()

                        print(log("[TRAIN] |"))
                        print(log("[WARN]  |-> Learning stagnated at epoch " + str(len(self.history['loss'])) + "."))

                        break

            # Check if stage has been aborted by Ctrl-C input

            except KeyboardInterrupt:
                LOG.warning("Keyboard interrupt detected. Will abort the training stage.")
                print(log("[TRAIN] |"))
                print(log("[WARN]  |-> Aborted at epoch " + str(len(self.history['loss']) + 1) + "!"))

            # Check if stage has reached the max epoch

            if n == self.max_epochs - 1: # Reached max epoch
                LOG.warning(f"Reached max epoch of {len(self.history['loss'])}.")

                print(log("[TRAIN] |"))
                print(log("[WARN]  |-> Reached max epoch of " + str(len(self.history['loss'])) + "!"))

            print(log("[TRAIN] |"))

        # Evaluate the training performance

        LOG.debug("Evaluating the training performance.")
        print(log("[TRAIN] |-> Evaluating : "), end='')

        scores = self.model.evaluate_generator(self.data_generator,
                                               steps=len(self.data_generator),
                                               max_queue_size=cpu_count(),
                                               use_multiprocessing=self.use_multiprocessing,
                                               workers=cpu_count(),
                                               verbose=int(info))

        LOG.debug(f"Loss: {scores[0]} - Accuracy: {scores[1]}")
        print(f"Loss: {scores[0] :.4f} - Accuracy: {scores[1] * 100 :.2f}%")

        # Training complete

        LOG.info(f"Training complete after {int((perf_counter() - start_time) // 60)} minutes and {int((perf_counter() - start_time) % 60)} seconds.")
        print(log("[TRAIN] V "))
        print(log("[TRAIN] Done!\n"))
        print(log(f"[INFO] Training complete after {int((perf_counter() - start_time) // 60)} minutes and {int((perf_counter() - start_time) % 60)} seconds.\n"))

    # def load_data(self, number_of_modes: int = 1):
    #     '''
    #     Load training and testing data.
    #     '''
    #     LOG.info(f"Loading training and validation data for {number_of_modes} modes.")

    #     try:
    #         LOG.debug("Generating training dataset.")
    #         print(log("[DATA] Generating data for superpositions of " + str(number_of_modes) + " different modes..."))
    #         print(log("[DATA] |"))

    #         train_data = GenerateData(self.max_order, number_of_modes, self.amplitude_variation, self.phase_variation, self.noise_variation, self.exposure, self.repeats, info=False)
    #         train_inputs = train_data.get_inputs(log("[DATA] |-> " + str(self.repeats) + " datasets of training data"))
    #         train_outputs = train_data.get_outputs()

    #         LOG.debug("Generating validation dataset.")
    #         print(log("[DATA] |"))

    #         val_data = GenerateData(self.max_order, number_of_modes, self.amplitude_variation, self.phase_variation, self.noise_variation, self.exposure, 1, info=False)
    #         val_inputs = val_data.get_inputs(log("[DATA] |-> 1 dataset of validation data"))
    #         val_outputs = val_data.get_outputs()

    #         LOG.info("Successfully generated training and validation data!")
    #         print(log("[DATA] V "))
    #         print(log("[DATA] Done! Training size: " + str(len(train_inputs)) + ", validation size: " + str(len(val_inputs)) + ".\n"))

    #     except MemoryError:
    #         LOG.critical("Memory overflow.")
    #         print(log("[DATA] V "))
    #         print(log("[FATAL] Memory overflow!\n"))

    #         sys.exit()

    #     return (train_inputs, train_outputs), (val_inputs, val_outputs)

    def plot(self, info: bool = True, axes: tuple = None, label: str = False, elapsed_time: bool = False):
        '''
        Plot the history of the model whilst training.
        '''
        LOG.info("Ploting model history.")
        LOG.debug(f"Locals: {locals()}")

        if not self.exists():
            LOG.warning("Model does not exist!")
            print(log("[WARN] Model does not exist!\n"))

            return

        if info: print(log("[PLOT] Plotting history..."))

        if elapsed_time: t = np.array(self.history['time']) / 60
        else: t = np.arange(1, len(self.history['loss']) + 1)

        if axes == None:
            LOG.debug("Generating axes as none were given.")

            fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
            fig.suptitle(f"Training and Validation History for {str(self)}")
            ax1.grid()
            ax2.grid()

            ax1.plot(t, self.history['loss'], label="Training Loss")[0]
            ax2.plot(t, self.history['accuracy'], label="Training Accuracy")[0]
            ax1.plot(t, self.history['val_loss'], label="Validation Loss")[0]
            ax2.plot(t, self.history['val_accuracy'], label="Validation Accuracy")[0]

            ax2.set_ylim(0, 1)

        else:
            LOG.debug("Plotting history using axes given.")

            ax1, ax2 = axes
            if not label: label = str(self)

            ax1.plot(t, self.history['loss'], label=label)[0]
            ax2.plot(t, self.history['val_loss'], label=label)[0]

            if np.max(self.history['val_loss']) > ax2.get_ylim()[1]: ax2.set_ylim(0, np.max(self.history['val_loss']))

        stage_change_indexes = [i for i in range(1, len(self.history['stage'])) if self.history['stage'][i] != self.history['stage'][i-1]]
        for i in stage_change_indexes:
            ax1.axvline(self.history['time'][i-1] / 60 if elapsed_time else i, color=ax1.get_lines()[-1].get_color(), linestyle='--')
            ax2.axvline(self.history['time'][i-1] / 60 if elapsed_time else i, color=ax2.get_lines()[-1].get_color(), linestyle='--')

        LOG.debug("Formatting plot.")

        if t[-1] > ax1.get_xlim()[1]: plt.xlim(0, t[-1])
        if np.max(self.history['loss']) > ax1.get_ylim()[1]: ax1.set_ylim(0, np.max(self.history['loss']))

        ax1.set_ylim(0)
        ax2.set_ylim(0)

        ax1.set_ylabel("Loss")
        ax2.set_xlabel(f"{'Elapsed Time (mins)' if elapsed_time else 'Epoch'}")
        ax2.set_ylabel(f"{'Accuracy' if axes == None else 'Validation Loss'}")

        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")

        if info:
            plt.show()
            print(log("[PLOT] Done!\n"))

        LOG.info("Model history plotted successfully!")
        return (ax1, ax2)

    def save(self, save_trained: bool = True):
        '''
        Save the ML model and history to files.
        '''
        if save_trained: LOG.info("Saving ML object model and history to files.")
        else: LOG.info("Saving ML object history to files.")
        print(log("[SAVE] Saving model... "), end='')

        os.makedirs(f"Models/{str(self)}", exist_ok=True) # Create directory for model

        if save_trained:
            LOG.debug(f"Saving Keras model to 'Models/{str(self)}/model.h5'.")
            self.model.save(f"Models/{str(self)}/model.h5")

        for i in self.history:
            LOG.debug(f"Saving performance history to 'Models/{str(self)}/{i}.txt'.")
            np.savetxt(f"Models/{str(self)}/{i}.txt", self.history[i], delimiter=",")

        LOG.debug(f"Saving classes to 'Models/{str(self)}/classes.txt'.")
        np.savetxt(f"Models/{str(self)}/classes.txt", self.classes, fmt="%s", delimiter=",")

        LOG.debug("Generating history plot for epochs.")
        self.plot(info=False, elapsed_time=False)
        plt.savefig("Models/" + str(self) + "/history_epoch.png", bbox_inches="tight", pad_inches=0)

        LOG.debug("Generating history plot for elapsed time.")
        self.plot(info=False, elapsed_time=True)
        plt.savefig("Models/" + str(self) + "/history_elapsed_time.png", bbox_inches="tight", pad_inches=0)

        LOG.info("ML object saved successfully!")
        print("Done!\n")

    def load(self, save_trained: bool = True, info: bool = False):
        '''
        Load a saved model.
        '''
        LOG.info("Loading ML object.")

        if not self.exists():
            LOG.warning("Model does not exist! Will now train and save.")
            print(log("[WARN] Model does not exist! Will now train and save.\n"))

            self.train(info)
            self.save(save_trained)

        print(log("[LOAD] Loading model... "), end='')
        LOG.debug("Loading ML object from files.")

        if self.trained():
            LOG.debug(f"Loading Keras model from 'Models/{str(self)}/model.h5'.")

            try:
                self.model = keras.models.load_model(f"Models/{str(self)}/model.h5", custom_objects={"loss": self.loss, "metrics": [self.accuracy]})
            except:
                LOG.error("Model corrupted! Will now delete and reload.")
                print("Model corrupted! Will now delete and reload.\n")

                shutil.rmtree(f"Models/{str(self)}")
                self.load(save_trained, info)

                return

        for i in self.history:
            LOG.debug(f"Loading performance history from 'Models/{str(self)}/{i}.txt'.")
            self.history[i] = np.loadtxt(f"Models/{str(self)}/{i}.txt", delimiter=",")

        LOG.debug(f"Loading classes from 'Models/{str(self)}/classes.txt'.")
        self.classes = np.loadtxt(f"Models/{str(self)}/classes.txt", dtype=str, delimiter="\n")
        self.classes = [eval(i.replace("H", "Hermite")) for i in self.classes]

        LOG.info("ML object loaded successfully!")
        print("Done!\n")

    def predict(self, data, threshold: float = 0.1, info: bool = True):
        '''
        Predict the superposition based on a 2D numpy array of the unknown optical cavity.
        '''
        LOG.info("Using model to make a prediction.")

        if not self.trained():
            LOG.warning("Model has not been trained!")
            print(log("[WARN] Model has not been trained!\n"))

            return

        start_time = perf_counter()

        if info: print(log("[PRED] Predicting... (shape = " + str(data.shape) + ")"))
        if info: print(log("[PRED] |"))

        LOG.debug("Formatting data.")
        formatted_data = np.array([data[..., np.newaxis]]) # Convert to the correct format for our neural network

        LOG.debug("Making prediction.")
        prediction = self.model.predict(formatted_data)[0] # Make prediction using model

        LOG.debug(f"Prediction: {list(prediction)}")
        LOG.debug(f"Generating superposition of modes above threshold of {threshold} and asigning the respective amplitudes and phases.")

        modes = []
        for i in range(len(prediction) // 2): # For all values of prediction

            # LOG.debug(f"{self.classes[i]}: {prediction[i] :.3f}" + int(prediction[i] > threshold) * " ***")
            if info: print(log(f"[PRED] |-> {self.classes[i]}: {prediction[i] :.3f}" + Colour.FAIL + int(prediction[i] > threshold) * " ***" + Colour.ENDC))

            if prediction[i] > threshold: # If the prediction is above a certain threshold
                modes.append(self.classes[i].copy()) # Copy the corresponding solution to modes

                amplitude = prediction[i]
                normalised_phase = prediction[i + (len(prediction) // 2)]
                actual_phase = (normalised_phase * (2 * np.pi)) - np.pi

                LOG.debug(f"{self.classes[i]}: {amplitude :.3f}, {normalised_phase :.3f}, {actual_phase :.3f}")

                modes[-1].amplitude = amplitude # Set that modes amplitude to the prediction value
                modes[-1].phase = actual_phase # Set the phase to the corresponding modes phase

        if info: print(log("[PRED] V "))

        if len(modes) == 0:
            LOG.critical(f"Prediction failed! A threshold of {threshold} is likely too high.")
            print(log(f"[FATAL] Prediction failed! A threshold of {threshold} is likely too high.\n"))

            sys.exit()

        answer = Superposition(*modes) # Normalise the amplitudes

        # self.calculate_phase(data, answer)

        LOG.info(f"Prediction complete! Took {round((perf_counter() - start_time) * 1000, 3)} milliseconds.")
        LOG.info(f"Reconstructed: {repr(answer)}")
        if info: print(log(f"[PRED] Done! Took {round((perf_counter() - start_time) * 1000, 3)} milliseconds."))
        if info: print(log(f"[PRED] Reconstructed: {str(answer)}\n"))

        return answer

    def compare(self, sup: Superposition, camera: dict = None, threshold: float = 0.1, info: bool = True, save: bool = False):
        '''
        Plot given superposition against predicted superposition for visual comparison.
        '''
        LOG.info(f"Comparing test superposition: {repr(sup)}")

        raw_amp_errs = self.errs[:int(len(self.errs)/2)]
        raw_phase_errs = self.errs[int(len(self.errs)/2):]

        if camera is not None:
            processor = ModeProcessor(camera)
        else:
            processor = self.data_generator.mode_processor

        if info: print(log("[PRED] Actual: " + str(sup)))
        raw_image = sup.superpose()
        noisy_image = processor.errorEffects(raw_image)
        pred = self.predict(raw_image, threshold=threshold, info=info)

        '''
        labels = [i.latex_print() for i in sup]
        sup_amps = [i.amplitude for i in sup]
        pred_amps = [pred.contains(i).amplitude for i in sup]
        sup_phases = [i.phase for i in sup]
        raw_pred_phases = [pred.contains(i).phase for i in sup]
        pred_phases = [0 if i == -10 else i for i in raw_pred_phases] # If mode does not exist, give it a phase of 0

        pred_strings = [str(mode) for mode in pred]
        amp_errs = [raw_amp_errs[i] for i in range(len(sup)) if str(sup[i]) in pred_strings]
        phase_errs = [raw_phase_errs[i] for i in range(len(sup)) if str(sup[i]) in pred_strings]
        '''
        true_strings = [str(m) for m in sup]
        pred_strings = [str(m) for m in pred]
        all_strings = true_strings + pred_strings

        sup_amps = []
        sup_phases = []
        pred_amps = []
        pred_phases = []
        amp_errs = []
        phase_errs = []
        labels = []

        for i, test_mode in enumerate(self.data_generator.hermite_modes): # Iterate over all possible modes
            if str(test_mode) in all_strings: # test_mode in input superposition and/or predicted superposition
                
                labels.append(test_mode.latex_print())

                # Input Superposition
                if str(test_mode) in true_strings: # test_mode present in input sup
                    sup_amps.append(sup.contains(test_mode).amplitude)
                    sup_phases.append(sup.contains(test_mode).phase)
                else: # test_mode not present in input superposition, but is in prediction
                    sup_amps.append(0)
                    sup_phases.append(0)
                
                # Predicted Superposition
                if str(test_mode) in pred_strings: # test_mode is in predicted superposition
                    pred_amps.append(pred.contains(test_mode).amplitude)
                    pred_phases.append(pred.contains(test_mode).phase)
                else: # test_mode not present in prediction, but is in input superposition
                    pred_amps.append(0)
                    pred_phases.append(0)
                
                amp_errs.append(raw_amp_errs[i])
                phase_errs.append(raw_phase_errs[i])


        x = np.arange(len(labels)) # Label locations
        width = 0.35 # Width of the bars

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(14, 10))
        fig.suptitle(r"$\bf{" + str(self) + "}$")

        ax4.set_xlabel(r"$\bf{Actual: }$" + "\n".join(textwrap.wrap(sup.latex_print())))
        ax5.set_xlabel(r"$\bf{Reconst: }$" + "\n".join(textwrap.wrap(pred.latex_print())))
        ax1.set_ylabel(r"$\bf{Amplitude}$")
        ax4.set_ylabel(r"$\bf{Phase}$")
        ax3.set_title(r"$\bf{Mode}$ $\bf{Amplitudes}$")
        ax6.set_title(r"$\bf{Mode}$ $\bf{Phases}$")

        ax1.imshow(noisy_image, cmap='jet')
        ax2.imshow(pred.superpose(), cmap='jet')
        ax4.imshow(sup.phase_map(), cmap='jet')
        ax5.imshow(pred.phase_map(), cmap='jet')
        rects1 = ax3.bar(x - (width / 2), sup_amps, width, label='Actual', zorder=3)
        rects2 = ax3.bar(x + (width / 2), pred_amps, width, yerr=amp_errs,  label='Reconstucted', zorder=3, capsize=10)
        rects3 = ax6.bar(x - (width / 2), sup_phases, width, label='Actual', zorder=3)
        rects4 = ax6.bar(x + (width / 2), pred_phases, width, yerr=phase_errs, label='Reconstucted', zorder=3, capsize=10)
        ax3.axhline(threshold, color='r', linestyle='--', zorder=5)

        # ax1.colorbar()
        # ax2.colorbar()
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax3.grid(zorder=0)
        ax6.grid(zorder=0)

        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.set_ylim(0.0, 1.1)
        ax3.legend(loc="upper right")
        ax6.set_xticks(x)
        ax6.set_xticklabels(labels)
        ax6.set_ylim(-np.pi, np.pi)

        ax6.set_yticks([-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax6.set_yticklabels(["$-\\pi$", "$-\\frac{3}{4}\\pi$", "$-\\frac{1}{2}\\pi$", "$-\\frac{1}{4}\\pi$", "$0$", "$\\frac{1}{4}\\pi$", "$\\frac{1}{2}\\pi$", "$\\frac{3}{4}\\pi$", "$\\pi$"])
        ax6.legend(loc="upper right")

        auto_label(rects1, ax3)
        auto_label(rects2, ax3)
        auto_label(rects3, ax6)
        auto_label(rects4, ax6)

        # fig.tight_layout()
        if save:
            if os.path.exists(f"Comparisons/{self}"):
                id = max([int(name[:-4]) for name in os.listdir(f"Comparisons/{self}")]) + 1
            else:
                os.makedirs(f"Comparisons/{self}") # Create directory for image
                id = 1

            LOG.debug(f"Saving to 'Comparisons/{self}/{id}.png'.")
            plt.savefig(f"Comparisons/{self}/{id}.png", bbox_inches="tight", pad_inches=0) # Save image

        else: plt.show()

        plt.close(fig)
        LOG.info("Comparison complete!")

    def evaluate(self, N: int = 500, info: bool = False):
        '''
        Evaluate the model by comparing against N randomly generated superpositions.
        '''
        LOG.info(f"Evaluating model using {N} randomly generated superpositions.")

        if not self.trained():
            LOG.warning("Model has not been trained!")
            print(log("[WARN] Model has not been trained!\n"))

            return

        # Check if the max number of comparison plots have already been generated

        if os.path.exists(f"Comparisons/{self}"):
            num_comps = len([name for name in os.listdir(f"Comparisons/{self}")])
            LOG.debug(f"Found {num_comps} existing comparison plots in 'Comparisons/{self}'.")

            if num_comps >= N:
                LOG.warning(f"Found {num_comps} comparison plots which exceeds the maximum of {N} plots.")
                print(log(f"[WARN] Found {num_comps} comparison plots which exceeds the maximum of {N} plots.\n"))

                return
        else: num_comps = 0

        print(log("[EVAL] Evaluating..."))
        print(log("[EVAL] |"))

        try:
            while self.data_generator.new_stage(): pass # Move to the last stage of training

            if self.errs is None:
                LOG.warning("Model errors have not yet been computed. Computing errors now")
                self.get_errs_of_model()
                LOG.warning("Model errors computed, resuming comparison.")

            for i in tqdm(range(N - num_comps), desc=log("[EVAL] |-> Generating comparison plots ")): self.compare(self.data_generator.get_random(), info=False, save=True) # Generate comparison plots

        except KeyboardInterrupt: LOG.info("Stopped evaluation due to keyboard interrupt.")

        LOG.info("Evaluation complete!")
        print(log("[EVAL] V "))
        print(log("[EVAL] Done!\n"))

    def calculate_phase(self, data, superposition: Superposition):
        '''
        Calculate the phases for modes in the superposition that minimises the MSE to the original data.
        '''
        for mode in superposition:
            sups = []
            for phase in range(11):
                mode.phase = ((phase / 10) * (2 * np.pi)) - np.pi
                sups.append(np.mean(np.square(superposition.superpose() - data)))
            mode.phase = ((np.argmin(sups) / 10) * (2 * np.pi)) - np.pi

    def free(self):
        '''
        Free GPU memory of this model.
        '''
        LOG.debug("Deleting model and freeing GPU memory.")
        print(log("[INFO] Deleting model and freeing GPU memory... "), end='')

        del self.model
        self.model = None
        K.clear_session()
        collected = gc.collect()

        LOG.debug(f"GPU memory deleted. Collected: {collected}.")
        print(f"Done! Collected: {collected}.\n")

    def optimise(self, param_name: str, param_range: str, plot: bool = True, save: bool = False) -> None:
        '''
        Loading / training multiple models and plotting comparison graphs of their performances.
        '''
        LOG.info(f"Optimising parameter '{param_name}' across range '{param_range}'.")
        print(log(f"[INFO] Optimising parameter '{param_name}' across range '{param_range}'.\n"))

        models = []
        for test in param_range:

            m = self.copy()

            if param_name in dir(m):
                setattr(m, param_name, test)

                LOG.debug(f"New model: {m}")
                print(log(f"[INFO] New model: {m}\n"))

            elif param_name in dir(m.data_generator):
                setattr(m.data_generator, param_name, test)
                m.data_generator = m.data_generator.copy()

                LOG.debug(f"New data generator: {m.data_generator}")
                print(log(f"[INFO] New data generator: {m.data_generator}\n"))

            else:
                LOG.critical(f"Parameter {param_name} does not exist!")
                print(log(f"[FATAL] Parameter {param_name} does not exist!"))

                sys.exit()

            m.load(save_trained=False) # Load the model, and if the model does not exist then train and save it
            models.append(m) # Add the model to the list 

        if plot:
            LOG.debug("Plotting optimisation graphs.")

            for time in (True, False):
                fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8), sharex=True, gridspec_kw={'hspace': 0})
                fig.suptitle(f"Comparing {param_name.replace('_', ' ').title()} by {'Elapsed Time' if time else 'Epoch'}")
                ax1.grid()
                ax2.grid()

                plt.xlim(0, 1E-9)
                plt.ylim(0, 1E-9)

                for m in models: m.plot(info=False, axes=(ax1, ax2), label=f"{param_name.replace('_', ' ').title()}: {getattr(m, param_name) if param_name in dir(m) else getattr(m.data_generator, param_name)}", elapsed_time=time)

                if save:
                    LOG.debug(f"Saving to 'Optimisation/{self.data_generator.__class__.__name__}({self.data_generator.max_order}){' on BlueBear' if self.use_multiprocessing else ' on Desktop'}/{param_name.replace('_', ' ').title()} by {'Elapsed Time' if time else 'Epoch'}{f' for {param_range}' if not param_name == 'training_strategy_name' else ''}.png'.")

                    os.makedirs(f"Optimisation/{self.data_generator.__class__.__name__}({self.data_generator.max_order}){' on BlueBear' if self.use_multiprocessing else ' on Desktop'}", exist_ok=True) # Create directory for optimisations
                    plt.savefig(f"Optimisation/{self.data_generator.__class__.__name__}({self.data_generator.max_order}){' on BlueBear' if self.use_multiprocessing else ' on Desktop'}/{param_name.replace('_', ' ').title()} by {'Elapsed Time' if time else 'Epoch'}{f' for {param_range}' if not param_name == 'training_strategy_name' else ''}.png", bbox_inches="tight", pad_inches=0) # Save image
                else:
                    plt.show()

        LOG.info("Optimisation complete.\n")

    def get_errs_of_model(self, n_test_points:int=1000):
        cumulative_error = np.zeros(len(self.classes))

        for i in tqdm(range(n_test_points), desc=log("[EVAL] |-> Computing model errors ")):
            test_sup = self.data_generator.get_random()
            true_amplitudes = [test_sup.contains(j).amplitude for j in self.data_generator.hermite_modes]
            true_phases = [test_sup.contains(j).phase for j in self.data_generator.hermite_modes]

            test_img = self.data_generator.mode_processor.errorEffects(test_sup.superpose())
            pred = self.predict(test_img, threshold=0, info=False)
            pred_amps = [pred.contains(j).amplitude for j in self.data_generator.hermite_modes]
            pred_phases = [pred.contains(j).phase for j in self.data_generator.hermite_modes]
            
            diff_amps = [(true_amplitudes[i] - pred_amps[i])**2 for i in range(len(pred_amps))]
            diff_phases = [0]*len(pred_phases)

            for i in range(len(pred_phases)):
                diff_phases[i] = np.min([(true_phases[i] - 2*np.pi - pred_phases[i])**2, (true_phases[i] - pred_phases[i])**2, (true_phases[i] + 2*np.pi - pred_phases[i])**2]) # Account for phase wrapping massively changing the error

            diffs = diff_amps + diff_phases
            cumulative_error += diffs

        self.errs = cumulative_error / (np.sqrt(n_test_points)*(n_test_points - 1))
        print(log("[EVAL] |"))




##################################################
##########                              ##########
##########          FUNCTIONS           ##########
##########                              ##########
##################################################


def log(message):
    '''
    Return message in the format given.
    '''
    # if level == "DEBUG":
    #     LOG.debug(message)
    # if level == "INFO":
    #     LOG.info(message)
    #     print(Colour.OKBLUE + "[INFO] " + Colour.ENDC + message)
    # if level == "WARNING":
    #     LOG.warning(message)
    #     print(Colour.WARNING + "[WARN] " + Colour.ENDC + message)
    # if level == "ERROR":
    #     LOG.error(message)
    #     print(Colour.FAIL + "[FATAL] " + Colour.ENDC + message)
    # if level == "CRITICAL":
    #     LOG.critical(message)
    #     print(Colour.FAIL + "[FATAL] " + Colour.ENDC + message)

    # text = message.replace(Colour.OKGREEN, '').replace(Colour.WARNING, '').replace(Colour.FAIL, '').replace(Colour.ENDC, '')

    # if text.contains("[INFO] "): LOG.info(text.replace("[INFO] ", ''))
    # if text.contains("[WARN] "): LOG.warning(text.replace("[WARN] ", ''))
    # if text.contains("[WARN] "): LOG.warning(text.replace("[WARN] ", ''))

    message = message.replace("->",         Colour.OKCYAN   + "->"      + Colour.ENDC)
    message = message.replace(" |",         Colour.OKCYAN   + " |"      + Colour.ENDC)
    message = message.replace(" V ",        Colour.OKCYAN   + " V "     + Colour.ENDC)
    message = message.replace("[INFO]",     Colour.OKBLUE   + "[INFO]"  + Colour.ENDC)
    message = message.replace("[WARN]",     Colour.WARNING  + "[WARN]"  + Colour.ENDC)
    message = message.replace("[FATAL]",    Colour.FAIL     + "[FATAL]" + Colour.ENDC)
    message = message.replace("[INIT]",     Colour.OKGREEN  + "[INIT]"  + Colour.ENDC)
    message = message.replace("[DATA]",     Colour.OKGREEN  + "[DATA]"  + Colour.ENDC)
    message = message.replace("[TRAIN]",    Colour.OKGREEN  + "[TRAIN]" + Colour.ENDC)
    message = message.replace("[PLOT]",     Colour.OKGREEN  + "[PLOT]"  + Colour.ENDC)
    message = message.replace("[EVAL]",     Colour.OKGREEN  + "[EVAL]"  + Colour.ENDC)
    message = message.replace("[SAVE]",     Colour.OKGREEN  + "[SAVE]"  + Colour.ENDC)
    message = message.replace("[LOAD]",     Colour.OKGREEN  + "[LOAD]"  + Colour.ENDC)
    message = message.replace("[PRED]",     Colour.OKGREEN  + "[PRED]"  + Colour.ENDC)

    return message

def VGG16(input_shape, classes):
    '''
    Returns the VGG16 model.
    '''
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='sigmoid'))

    return model

def auto_label(rects, ax):
    '''
    Attach a log label above each bar in rects displaying its height.
    '''
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -15),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom")

def get_model_error(model, data_object:GenerateData, test_number:int=10, sup:Superposition=None):
    '''
    Tests the accuracy of the model from data contained within data_object
    
    Assumes a gaussian resultant error through the Central Limit Theorem
    
    Tests on test_percent of the data given in the data_object
    '''
    if sup is None:
        test_data = np.array(random.sample(data_object.combs, test_number)) # Select superpositions randomly from combs
    else:
        test_data = np.array([sup.copy() for i in range(test_number)]) # Make several copies of the target superposition

    test_data = np.array([Superposition(*[data_object.randomise_amp_and_phase(j) for j in i]) for i in test_data]) # Randomise all amps and phases
    model_predictions = np.array([model.predict(data.superpose()) for data in test_data]) # Predict superpositions through models

    test_amps = np.array([[mode.amplitude for mode in sup] for sup in test_data]).flatten()
    model_amps = np.array([[mode.amplitude for mode in sup] for sup in model_predictions]).flatten()

    amp_err = (np.sum([(model_amps[i] - test_amps[i])**2 for i in range(len(test_amps))])/(len(test_amps) - 1))**0.5 # Predicts amplitude error assuming error is constant throughout multivariate space

    test_phases = np.array([[mode.phase for mode in sup] for sup in test_data]).flatten()
    model_phases = np.array([[mode.phase for mode in sup] for sup in model_predictions]).flatten()

    phase_err = (np.sum([(model_phases[i] - test_phases[i])**2 for i in range(len(test_phases))])/(len(test_phases) - 1))**0.5 # Predicts phase error assuming error is constant throughout multivariate space

    test_imgs = np.array([sup.superpose() for sup in test_data])
    model_imgs = np.array([sup.superpose() for sup in model_predictions])

    img_err = (np.sum([(model_imgs[i] - test_imgs[i])**2 for i in range(len(test_imgs))])/(len(test_imgs) - 1))**0.5 # Predicts img error assuming error is constant throughout multivariate space

    return amp_err, phase_err, img_err

def optimise(param_name: str, param_range: str, plot: bool = True, save: bool = False) -> None:
    '''
    Loading / training multiple models and plotting comparison graphs of their performances.
    '''
    LOG.info(f"Optimising parameter '{param_name}' across range '{param_range}'.")
    print(log(f"[INFO] Optimising parameter '{param_name}' across range '{param_range}'.\n"))

    models = []
    for test in param_range:
        m = ML()

        if param_name in dir(m):
            setattr(m, param_name, test)

            LOG.debug(f"New model: {m}")
            print(log(f"[INFO] New model: {m}\n"))

        elif param_name in dir(m.data_generator):
            setattr(m.data_generator, param_name, test)

            LOG.debug(f"New model: {m}")
            print(log(f"[INFO] New data generator: {m.data_generator}\n"))

        else:
            LOG.critical(f"Parameter {param_name} does not exist!")
            print(log(f"[FATAL] Parameter {param_name} does not exist!"))

            sys.exit()

        m.load(save_trained=False) # Load the model, and if the model does not exist then train and save it
        models.append(m) # Add the model to the list 

    if plot:
        LOG.debug("Plotting optimisation graphs.")

        for time in (True, False):
            fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8), sharex=True, gridspec_kw={'hspace': 0})
            fig.suptitle(f"Comparing {param_name.replace('_', ' ').title()} by {'Elapsed Time' if time else 'Epoch'}")
            ax1.grid()
            ax2.grid()

            plt.xlim(0, 1E-9)
            plt.ylim(0, 1E-9)

            for m in models: m.plot(info=False, axes=(ax1, ax2), label=param_name.replace('_', ' ').title() + ": " + str(getattr(m, param_name)), elapsed_time=time)

            if save:
                LOG.debug(f"Saving to 'Optimisation/Comparing {param_name.replace('_', ' ').title()} by {'Elapsed Time' if time else 'Epoch'}.png'.")
                plt.savefig(f"Optimisation/Comparing {param_name.replace('_', ' ').title()} by {'Elapsed Time' if time else 'Epoch'} across {param_range}.png", bbox_inches="tight", pad_inches=0) # Save image
            else:
                plt.show()

    LOG.info("Optimisation complete.\n")




##################################################
##########                              ##########
##########            MAIN              ##########
##########                              ##########
##################################################


# This is supposed to automatically allocate memory to the GPU when it is needed instead of reserving the full space.
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')

    print("█████▀█████████████████████████████████████████████████████████████████████████████\n"
          "█─▄▄▄▄██▀▄─██▄─██─▄█─▄▄▄▄█─▄▄▄▄█▄─▄██▀▄─██▄─▀█▄─▄███▄─▀█▀─▄█─▄▄─█▄─▄▄▀█▄─▄▄─█─▄▄▄▄█\n"
          "█─██▄─██─▀─███─██─██▄▄▄▄─█▄▄▄▄─██─███─▀─███─█▄▀─█████─█▄█─██─██─██─██─██─▄█▀█▄▄▄▄─█\n"
          "▀▄▄▄▄▄▀▄▄▀▄▄▀▀▄▄▄▄▀▀▄▄▄▄▄▀▄▄▄▄▄▀▄▄▄▀▄▄▀▄▄▀▄▄▄▀▀▄▄▀▀▀▄▄▄▀▄▄▄▀▄▄▄▄▀▄▄▄▄▀▀▄▄▄▄▄▀▄▄▄▄▄▀\n")

    # max_order = 3
    # number_of_modes = 3
    # amplitude_variation = 0.2
    # phase_variation = 0.2
    # noise_variation = 0.0
    # exposure = (0.0, 1.0)
    # repeats = 32

    # Loading saved model

    # Training and saving

    # m = ML(data_generator=BasicGenerator(amplitude_variation=0.2, phase_variation=0.2)) # Dataset(training_strategy_name="stage_change_test")
    # m.train()
    # m.save()

    # data = BasicGenerator(amplitude_variation=0.2, phase_variation=0.2)
    # data.new_stage() # Init stage 1
    # data.new_stage() # Init stage 2
    # data.new_stage() # Init stage 3
    # data.new_stage() # Init stage 4

    # for i in tqdm(range(1000)): m.compare(data.get_random(), info=False, save=True)

    # m = ML(data_generator=BasicGenerator(amplitude_variation=0.5, phase_variation=1.0))
    # m.train()
    # m.save()

    # data = BasicGenerator(amplitude_variation=0.5, phase_variation=1.0)
    # data.new_stage() # Init stage 1
    # data.new_stage() # Init stage 2
    # data.new_stage() # Init stage 3
    # data.new_stage() # Init stage 4

    # for i in tqdm(range(1000)): m.compare(data.get_random(), info=False, save=True)

    # datas = [data.get_random() for i in tqdm(range(1000))]
    # p = Pool(cpu_count())
    # p.map(m.compare, sup=tqdm(datas))

    # for i in range(10):
    #     sup = data.get_random()
    #     m.compare(sup)

    optimise("repeats", [2**n for n in range(1, 10)], plot=True, save=True)
    optimise("batch_size", [2**n for n in range(9)], plot=True, save=True)
    optimise("optimiser", ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"], plot=True, save=True)
    optimise("learning_rate", [round(0.1**n, n) for n in range(8)], plot=True, save=True)
    optimise("learning_rate", [0.001 * n for n in range(1, 9)], plot=True, save=True)
    optimise("learning_rate", [round(0.0001 * n, 4) for n in range(1, 9)], plot=True, save=True)

    # print(tf.config.list_physical_devices())
    # with tf.device("gpu:0"):
    #     model = ML(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure, repeats)
    #     try:
    #         model.load()
    #     except:
    #         model.train()
    #         model.save

    #     # Generating test data for comparisons

    #     data = GenerateData(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure)

    # sup = Superposition(Hermite(1,2), Hermite(2,0), Hermite(0,1))
    # prediction = model.predict(sup.superpose())
    # sup.show()
    # prediction.show()

    # max_order = 3
    # number_of_modes = 5
    # amplitude_variation = 0.2
    # phase_variation = 0.0
    # noise_variation = 0.0
    # exposure = (0.0, 1.0)
    # repeats = 50

    # model = ML(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure, repeats)
    # #sys.path.insert(1, '../System') # Move to directory containing simulation files
    # model.load()

    # data = GenerateData(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure)

    # errs = get_model_error(model, data, 0.5)

    # print(errs[0])
    # print(errs[1])
    # plt.imshow(errs[2])
    # plt.show()
