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
import gc
import logging

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide Tensorflow info, warning and error messages

from Gaussian_Beam import Hermite, Superposition, Laguerre
from DataHandling import Dataset, GenerateData
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
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from itertools import combinations, chain
from multiprocessing import Pool, cpu_count

# logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('matplotlib.font_manager').disabled = True

log_format = "%(asctime)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s"
logging.basicConfig(filename="Logs" + os.sep + "{:%d-%m-%Y}.log".format(datetime.now()), level="DEBUG", format=log_format)




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




class DataGenerator(keras.utils.Sequence):
    '''
    The class 'DataGenerator' that generates data for Keras in training in real-time.
    '''

    def __init__(self,
                 max_order: int = 3,
                 number_of_modes: int = 3,
                 amplitude_variation: float = 0.2,
                 phase_variation: float = 0.2,
                 noise_variation: float = 0.0,
                 exposure: tuple = (0.0, 1.0),
                 repeats: int = 50,
                 batch_size: int = 64):
        '''
        Initialise the class with the required complexity.

        'max_order': Max order of Guassian modes in superpositions (x > 0).
        'number_of_modes': How many modes you want to superimpose together (x > 0).
        'ampiltude_variation': How much you want to vary the amplitude of the Gaussian modes by (x > 0).
        '''
        logging.info("Initialising generator.")

        self.max_order = max_order
        self.max_number_of_modes = number_of_modes
        self.amplitude_variation = amplitude_variation
        self.phase_variation = phase_variation
        self.noise_variation = noise_variation
        self.exposure = exposure
        self.repeats = repeats
        self.batch_size = batch_size

        logging.debug(f"Locals: {locals()}")

        self.hermite_modes = [Hermite(l=i, m=j) for i in range(max_order) for j in range(max_order)]
        self.laguerre_modes = [Laguerre(p=i, m=j) for i in range(max_order // 2) for j in range(max_order // 2)]
        self.gauss_modes = self.hermite_modes + self.laguerre_modes
        self.number_of_modes = 1

        logging.info("Generator initialised!")

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return repr(self)

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + f"({self.max_order}, {self.max_number_of_modes}, {self.amplitude_variation}, {self.phase_variation}, {self.noise_variation}, {self.exposure}, {self.repeats}, {self.batch_size})"

    def __len__(self):
        '''
        Denotes the number of batches per epoch.
        For us this is the (total number of combinations * repeats) / batch size.
        '''
        return int(len(self.combs) / self.batch_size)

    def __getitem__(self, index):
        '''
        Generates and returns one batch of data.
        '''
        # logging.debug(f"Getting item {index}.")

        combs = [self.combs[i] for i in range(index * self.batch_size, (index + 1) * self.batch_size)] # Take combs in order
        # combs = [self.combs[np.random.randint(len(self.combs))] for i in range(self.batch_size)] # Take random combs from self.combs
        sups = [self.generate_superposition(comb) for comb in combs]

        X = np.array(self.get_inputs(*sups))[..., np.newaxis]
        Y = np.array([[i.contains(j).amplitude for j in self.hermite_modes] + [np.cos(i.contains(j).phase) for j in self.hermite_modes] for i in sups])

        return X, Y

    def new_stage(self):
        '''
        Sets the stage of training for this generator.
        This generates all the combinations for that specified stage.
        '''
        logging.debug(f"Incrementing dataset to stage {self.number_of_modes}.")

        self.number_of_modes += 1
        if self.number_of_modes > self.max_number_of_modes: return False

        self.combs = [list(combinations(self.gauss_modes, i + 1)) for i in range(self.number_of_modes)]
        self.combs = [i[j] for i in self.combs for j in range(len(i))] * self.repeats
        random.shuffle(self.combs) # Shuffle the combinations list

        return True

    def generate_superposition(self, comb):
        '''
        Generates the superposition with randomised amplitudes, phase, noise and exposure for a given combination.
        '''
        return Superposition(*[self.randomise_amp_and_phase(i) for i in comb])

    def get_inputs(self, *sups):
        '''
        Get inputs from list of superpositions.
        '''
        inputs = []
        for sup in sups:
            sup.noise_variation = self.noise_variation
            sup.exposure = self.exposure

            inputs.append(sup.superpose())

        return inputs

    def get_classes(self):
        '''
        Get the num_classes result required for model creation.
        '''
        logging.debug("Getting classes.")

        return np.array(self.hermite_modes * 2, dtype=object)

    def randomise_amp_and_phase(self, mode):
        '''
        Randomise the amplitude and phase of mode according to normal distributions of self.amplitude_variation and self.phase_variation width.
        Returns new mode with randomised amp and phase.
        '''
        x = mode.copy()

        x *= np.abs(np.random.normal(scale=self.amplitude_variation) + 1)
        x.add_phase(np.random.normal(scale=self.phase_variation))

        return x

    def get_random(self):
        '''
        Returns a random superposition from the dataset.
        '''
        logging.debug("Getting a random superposition from generator.")

        comb = self.combs[np.random.randint(len(self.combs))]
        sup = self.generate_superposition(comb)

        return sup




class ML:
    '''
    The class 'ML' that represents a Keras model using datasets from Gaussian modes.
    '''
    def __init__(self,
                 max_order: int = 3,
                 number_of_modes: int = 3,
                 amplitude_variation: float = 0.2,
                 phase_variation: float = 0.2,
                 noise_variation: float = 0.0,
                 exposure: tuple = (0.0, 1.0),
                 repeats: int = 50,
                 batch_size: int = 64,
                 optimiser: str = "Adamax",
                 learning_rate: float = 0.0001):
        '''
        Initialise the class.
        '''
        logging.info("Initialising ML model.")

        self.max_order = max_order
        self.number_of_modes = number_of_modes
        self.amplitude_variation = amplitude_variation
        self.phase_variation = phase_variation
        self.noise_variation = noise_variation
        self.exposure = exposure
        self.repeats = repeats
        self.batch_size = batch_size
        self.optimiser = optimiser
        self.learning_rate = learning_rate

        logging.debug(f"Locals: {locals()}")

        self.max_epochs = 100 # Max epochs before training is terminated
        self.success_loss = 0.003 # Loss at which the training is considered successful
        self.stagnation = 5 # Epochs of stagnation before terminating training stage
        self.history = {"time": [], "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        self.model = None

        print(Colour.HEADER + Colour.BOLD + "____________________| " + str(self) + " |____________________\n" + Colour.ENDC)

        logging.info("ML model initialised!")

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return repr(self)

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + f"({self.max_order}, {self.number_of_modes}, {self.amplitude_variation}, {self.phase_variation}, {self.noise_variation}, {self.exposure}, {self.repeats}, {self.batch_size}, '{self.optimiser}', {self.learning_rate})"

    def exists(self):
        '''
        Check if the model exits in the file system.
        '''
        logging.debug("Checking if ML object exists in the file system.")

        return os.path.exists("Models/" + str(self))

    def trained(self):
        '''
        Check if the model has been trained before.
        '''
        logging.debug("Checking if ML model has been trained before.")

        return os.path.exists("Models/" + str(self) + "/" + str(self) + ".h5")

    def accuracy(self, y_true, y_pred):
        '''
        Custom metric to determine the accuracy of our regression problem using rounded accuracy.
        '''
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    def create_model(self, summary: bool = True):
        '''
        Create the Keras model in preparation for training.
        '''
        logging.debug("Creating the architecture of the Keras model.")

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

        logging.debug("Keras model layers created.")
        logging.debug("Compiling the model.")

        # model = VGG16(self.input_shape, len(self.classes)) # Override model with VGG16 model
        model.compile(loss="mse", optimizer=eval(f"{self.optimiser}(learning_rate={self.learning_rate})"), metrics=[self.accuracy])

        logging.debug(f"Model compiled. Optimiser: {self.optimiser}(learning_rate={self.learning_rate}).")

        # We choose sigmoid and binary_crossentropy here because we have a multilabel neural network, which becomes K binary classification problems
        # Using softmax would be wrong as it raises the probabiity on one class and lowers others

        if summary: print(model.summary())
        if summary: keras.utils.plot_model(model, str(self), show_shapes=True)

        logging.debug("Keras model architecture created successfully!")
        return model

    def train(self, info: bool = False):
        '''
        Train the model.
        '''
        logging.info("Initialising the training process for the model.")

        # Check if the model has already been trained

        if self.trained():
            logging.warning("Trained model already exists.")
            print(log("[WARN] Trained model already exists!\n"))

            self.load()
            return

        # Timing how long the training process took

        start_time = perf_counter()

        # Create the data generator object for the dataset

        logging.debug("Creating the data generator object for the dataset.")
        print(log("[INIT] Initialising dataset... "), end='')

        dataset = DataGenerator(self.max_order, self.number_of_modes, self.amplitude_variation, self.phase_variation, self.noise_variation, self.exposure, self.repeats, self.batch_size)
        self.classes = dataset.get_classes()
        self.input_shape = (self.classes[0].resolution, self.classes[0].resolution, 1)

        print("Done!")
        logging.debug(f"Classes: {list(self.classes)}")
        logging.debug(f"Input shape: {self.input_shape}")

        # Create the Keras model

        logging.debug("Creating the Keras model.")
        print(log("[INIT] Generating model (input shape = " + str(self.input_shape) + ", classes = " + str(len(self.classes)) + ", optimiser: " + self.optimiser + ")... "), end='')

        self.model = self.create_model(summary=False)

        print("Done!\n")

        # Begin training

        logging.info("Starting the training process.")
        print(log("[TRAIN] Training..."))
        print(log("[TRAIN] |"))

        # For every stage / complexity of data in the dataset

        while dataset.new_stage():
            logging.info(f"Stage {dataset.number_of_modes - 1} / {dataset.max_number_of_modes - 1} of training.")
            print(log("[TRAIN] |-> Stage               : " + str(dataset.number_of_modes - 1) + " / " + str(dataset.max_number_of_modes - 1) + "."))
            print(log("[TRAIN] |-> Dataset             : " + str(dataset) + "."))
            print(log("[TRAIN] |-> Size                : " + str(len(dataset)) + "."))
            print(log("[TRAIN] |-> Success Condition   : A loss of " + str(self.success_loss) + "."))
            print(log("[TRAIN] |-> Terminate Condition : Reaching epoch " + str(len(self.history["loss"]) + self.max_epochs) + " or " + str(self.stagnation) + " consecutive epochs of stagnation."))
            print(log("[TRAIN] |"))

            # For every epoch of training on that stage

            n = 0
            try:
                iterator = tqdm(range(self.max_epochs), log("[TRAIN] |-> Training "))
                for n in iterator:
                    # logging.debug(f"Epoch {n + 1}:")

                    # Fit the model to the stage of the data generator

                    history_callback = self.model.fit(dataset,
                                                      validation_data=dataset,
                                                      validation_steps=1,
                                                      batch_size=self.batch_size,
                                                      use_multiprocessing=False,
                                                      workers=cpu_count(),
                                                      verbose=int(info))

                    # Save the performance of this epoch

                    for i in self.history:
                        if i == "time": self.history[i].append(perf_counter() - start_time) # Save time elapsed since training began
                        else: self.history[i].append(history_callback.history[i][0]) # Save performance of epoch

                        # logging.debug(f"{i.replace('_', ' ').title()}: {self.history[i][-1]}")

                    # Compute the stagnation of the training run

                    stagnates = len(np.where(np.round(self.history["loss"][-1], 4) >= np.round(self.history["loss"][-min(n + 1, self.stagnation + 1):-1], 4))[0])
                    if stagnates == 0: indicator = Colour.OKGREEN + '++' + Colour.ENDC
                    else: indicator = (Colour.WARNING if stagnates < self.stagnation - 1 else Colour.FAIL) + '-' + str(stagnates) + Colour.ENDC

                    logging.debug(f"Epoch: {n + 1} - Time: {self.history['time'][-1] :.2f}s - Loss: {self.history['loss'][-1] :.4f} - Accuracy: {self.history['accuracy'][-1] :.4f} - Val Loss: {self.history['val_loss'][-1] :.4f} - Val Accuracy: {self.history['val_accuracy'][-1] :.4f} - Stagnation: {stagnates}")

                    # Update the loading bar description with the current losses

                    iterator.set_description(log("[TRAIN] |-> " + indicator + " Loss: %0.4f - Val Loss: %0.4f " % (self.history["loss"][-1], self.history["val_loss"][-1])))

                    # Check if gradient descent has diverged so training has failed

                    if isnan(self.history["loss"][-1]): # Loss is nan so training has failed
                        logging.critical("Loss is 'nan' so training has diverged and failed.")
                        print(log("\n[TRAIN] V "))
                        print(log("[FATAL] Training failed! Gradient descent diverged at epoch " + str(len(self.history["loss"])) + ".\n"))

                        sys.exit()

                    # Check if loss has reached an acceptable level

                    elif self.history["loss"][-1] < self.success_loss: # Loss has reached success level
                        logging.debug("Loss has reached success level. Will break from training stage.")
                        iterator.close()

                        print(log("[TRAIN] |"))
                        print(log("[TRAIN] |-> " + str(self.success_loss) + " loss achieved at epoch " + str(len(self.history["loss"])) + "."))

                        break

                    # Check if learning has stagnated

                    elif stagnates >= self.stagnation: # Training has stagnated so is not learning anymore
                        logging.warning("Training has stagnated, so model is no longer learning.")
                        iterator.close()

                        print(log("[TRAIN] |"))
                        print(log("[WARN]  |-> Learning stagnated at epoch " + str(len(self.history["loss"])) + "."))

                        break

            # Check if stage has been aborted by Ctrl-C input

            except KeyboardInterrupt:
                logging.warning("Keyboard interrupt detected. Will abort the training stage.")
                print(log("[TRAIN] |"))
                print(log("[WARN]  |-> Aborted at epoch " + str(len(self.history["loss"]) + 1) + "!"))

            # Check if stage has reached the max epoch

            if n == self.max_epochs - 1: # Reached max epoch
                logging.warning(f"Reached max epoch of {len(self.history['loss'])}.")

                print(log("[TRAIN] |"))
                print(log("[WARN]  |-> Reached max epoch of " + str(len(self.history["loss"])) + "!"))

            print(log("[TRAIN] |"))

        # Evaluate the training performance

        logging.debug("Evaluating the training performance.")
        print(log("[TRAIN] |-> Evaluating : "), end='')

        scores = self.model.evaluate(dataset,
                                     batch_size=self.batch_size, 
                                     use_multiprocessing=False,
                                     workers=cpu_count(),
                                     verbose=int(info))

        logging.debug(f"Loss: {scores[0]} - Accuracy: {scores[1]}")
        print("Loss: %0.4f - Accuracy: %0.4f " % (scores[0], scores[1]))

        # Training complete

        logging.info(f"Training complete after {int((perf_counter() - start_time) // 60)} minutes and {int((perf_counter() - start_time) % 60)} seconds.")
        print(log("[TRAIN] V "))
        print(log("[TRAIN] Done!\n"))
        print(log(f"[INFO] Training complete after {int((perf_counter() - start_time) // 60)} minutes and {int((perf_counter() - start_time) % 60)} seconds.\n"))

    def load_data(self, number_of_modes: int = 1):
        '''
        Load training and testing data.
        '''
        logging.info(f"Loading training and validation data for {number_of_modes} modes.")

        try:
            logging.debug("Generating training dataset.")
            print(log("[DATA] Generating data for superpositions of " + str(number_of_modes) + " different modes..."))
            print(log("[DATA] |"))

            train_data = GenerateData(self.max_order, number_of_modes, self.amplitude_variation, self.phase_variation, self.noise_variation, self.exposure, self.repeats, info=False)
            train_inputs = train_data.get_inputs(log("[DATA] |-> " + str(self.repeats) + " datasets of training data"))
            train_outputs = train_data.get_outputs()

            logging.debug("Generating validation dataset.")
            print(log("[DATA] |"))

            val_data = GenerateData(self.max_order, number_of_modes, self.amplitude_variation, self.phase_variation, self.noise_variation, self.exposure, 1, info=False)
            val_inputs = val_data.get_inputs(log("[DATA] |-> 1 dataset of validation data"))
            val_outputs = val_data.get_outputs()

            logging.info("Successfully generated training and validation data!")
            print(log("[DATA] V "))
            print(log("[DATA] Done! Training size: " + str(len(train_inputs)) + ", validation size: " + str(len(val_inputs)) + ".\n"))

        except MemoryError:
            logging.critical("Memory overflow.")
            print(log("[DATA] V "))
            print(log("[FATAL] Memory overflow!\n"))

            sys.exit()

        return (train_inputs, train_outputs), (val_inputs, val_outputs)

    def plot(self, info: bool = True, axes: tuple = None, label: str = False, elapsed_time: bool = False):
        '''
        Plot the history of the model whilst training.
        '''
        logging.info("Ploting model history.")
        logging.debug(f"Locals: {locals()}")

        if info: print(log("[PLOT] Plotting history..."))

        if elapsed_time: t = self.history["time"]
        else: t = np.arange(1, len(self.history["loss"]) + 1)

        if axes == None:
            logging.debug("Generating axes as none were given.")

            fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
            fig.suptitle(f"Training and Validation History for {str(self)}")
            ax1.grid()
            ax2.grid()

            ax1.plot(t, self.history["loss"], label="Training Loss")[0]
            ax2.plot(t, self.history["accuracy"], label="Training Accuracy")[0]
            ax1.plot(t, self.history["val_loss"], label="Validation Loss")[0]
            ax2.plot(t, self.history["val_accuracy"], label="Validation Accuracy")[0]

            ax2.set_ylim(0, 1)

        else:
            logging.debug("Plotting history using axes given.")

            ax1, ax2 = axes
            if not label: label = str(self)

            ax1.plot(t, self.history["loss"], label=label)[0]
            ax2.plot(t, self.history["val_loss"], label=label)[0]

            if np.max(self.history["val_loss"]) > ax2.get_ylim()[1]: ax2.set_ylim(0, np.max(self.history["val_loss"]))

        logging.debug("Formatting plot.")

        if t[-1] > ax1.get_xlim()[1]: plt.xlim(0, t[-1])
        if np.max(self.history["loss"]) > ax1.get_ylim()[1]: ax1.set_ylim(0, np.max(self.history["loss"]))

        ax1.set_ylim(0)
        ax2.set_ylim(0)

        ax1.set_ylabel("Loss")
        ax2.set_xlabel(f"{'Elapsed Time (s)' if elapsed_time else 'Epoch'}")
        ax2.set_ylabel(f"{'Accuracy' if axes == None else 'Validation Loss'}")

        ax1.legend()
        ax2.legend()

        if info:
            plt.show()
            print(log("[PLOT] Done!\n"))
        
        logging.info("Model history plotted successfully!")
        return (ax1, ax2)

    def save(self, save_trained: bool = True):
        '''
        Save the ML model and history to files.
        '''
        if save_trained: logging.info("Saving ML object model and history to files.")
        else: logging.info("Saving ML object history to files.")
        print(log("[SAVE] Saving model... "), end='')

        os.makedirs(f"Models/{str(self)}", exist_ok=True) # Create directory for model

        if save_trained:
            logging.debug(f"Saving Keras model to 'Models/{str(self)}/{str(self)}.h5'.")
            self.model.save(f"Models/{str(self)}/{str(self)}.h5")

        for i in self.history:
            logging.debug(f"Saving performance history to 'Models/{str(self)}/{i}.txt'.")
            np.savetxt(f"Models/{str(self)}/{i}.txt", self.history[i], delimiter=",")

        logging.debug(f"Saving classes to 'Models/{str(self)}/classes.txt'.")
        np.savetxt(f"Models/{str(self)}/classes.txt", self.classes, fmt="%s", delimiter=",")

        logging.debug("Generating history plot for epochs.")
        self.plot(info=False, elapsed_time=False)
        plt.savefig("Models/" + str(self) + "/history_epoch.png", bbox_inches='tight', pad_inches=0)

        logging.debug("Generating history plot for elapsed time.")
        self.plot(info=False, elapsed_time=True)
        plt.savefig("Models/" + str(self) + "/history_elapsed_time.png", bbox_inches='tight', pad_inches=0)

        logging.info("ML object saved successfully!")
        print("Done!\n")

    def load(self, save_trained: bool = True):
        '''
        Load a saved model.
        '''
        logging.info("Loading ML object.")

        if not self.exists():
            logging.warning("Model does not exist! Will now train and save.")
            print(log("[WARN] Model does not exist! Will now train and save.\n"))

            self.train()
            self.save(save_trained)
            if not save_trained: self.free()

            return

        elif not self.trained():
            logging.warning("Model exists but has not been trained! Will only load history.")
            print(log("[WARN] Model exists but has not been trained! Will only load history.\n"))

        print(log("[LOAD] Loading model... "), end='')
        logging.debug("Loading ML object from files.")

        if self.trained():
            logging.debug(f"Loading Keras model from 'Models/{str(self)}/{str(self)}.h5'.")
            self.model = keras.models.load_model(f"Models/{str(self)}/{str(self)}.h5", custom_objects={"metrics": [self.accuracy]})

        for i in self.history:
            logging.debug(f"Loading performance history from 'Models/{str(self)}/{i}.txt'.")
            self.history[i] = np.loadtxt(f"Models/{str(self)}/{i}.txt", delimiter=",")

        logging.debug(f"Loading classes from 'Models/{str(self)}/classes.txt'.")
        self.classes = np.loadtxt(f"Models/{str(self)}/classes.txt", dtype=str, delimiter="\n")
        self.classes = [eval(i.replace("HG", "Hermite")) for i in self.classes]

        logging.info("ML object loaded successfully!")
        print("Done!\n")

    def predict(self, data, threshold: float = 0.2, info: bool = True):
        '''
        Predict the superposition based on a 2D numpy array of the unknown optical cavity.
        '''
        logging.info("Using model to make a prediction.")
        logging.debug(f"Locals: {locals()}")

        if not self.exists():
            logging.critical("Model does not exist!")
            print(log("[FATAL] Model does not exist!\n"))

            return

        elif not self.trained():
            logging.error("Model has not been trained!")
            print(log("[WARN] Model has not been trained!\n"))

            return

        start_time = perf_counter()

        if info: print(log("[PRED] Predicting... (shape = " + str(data.shape) + ")"))
        if info: print(log("[PRED] |"))

        logging.debug("Formatting data.")
        formatted_data = np.array([data[..., np.newaxis]]) # Convert to the correct format for our neural network

        logging.debug("Making prediction.")
        prediction = self.model.predict(formatted_data)[0] # Make prediction using model

        logging.debug(f"Prediction: {list(prediction)}")
        logging.debug(f"Generating superposition of modes above threshold of {threshold} and asigning the respective amplitudes and phases.")

        modes = []
        for i in range(len(prediction) // 2): # For all values of prediction

            logging.debug(f"{self.classes[i]}: {prediction[i] :.3f}" + int(prediction[i] > threshold) * " ***")
            if info: print(log(f"[PRED] |-> {self.classes[i]}: {prediction[i] :.3f}" + Colour.FAIL + int(prediction[i] > threshold) * " ***" + Colour.ENDC))

            if prediction[i] > threshold: # If the prediction is above a certain threshold
                modes.append(self.classes[i].copy()) # Copy the corresponding solution to modes
                modes[-1].amplitude = prediction[i] # Set that modes amplitude to the prediction value
                modes[-1].phase = np.arccos(prediction[i + (len(prediction) // 2)]) # Set the phase to the corresponding modes phase

        if info: print(log("[PRED] V "))

        if len(modes) == 0:
            logging.critical(f"Prediction failed! A threshold of {threshold} is likely too high.")
            print(log(f"[FATAL] Prediction failed! A threshold of {threshold} is likely too high.\n"))

            sys.exit()

        answer = Superposition(*modes) # Normalise the amplitudes

        # self.calculate_phase(data, answer)

        logging.info(f"Prediction complete! Took {round((perf_counter() - start_time) * 1000, 3)} milliseconds.")
        logging.info(f"Reconstructed: {repr(answer)}")
        if info: print(log(f"[PRED] Done! Took {round((perf_counter() - start_time) * 1000, 3)} milliseconds."))
        if info: print(log(f"[PRED] Reconstructed: {str(answer)}\n"))

        return answer

    def compare(self, sup: Superposition, info: bool = True, save: bool = False):
        '''
        Plot given superposition against predicted superposition for visual comparison.
        '''
        logging.info(f"Comparing test superposition: {repr(sup)}")

        if info: print(log("[PRED] Actual: " + str(sup)))
        pred = self.predict(sup.superpose(), info=info)

        labels = [str(i) for i in sup]
        sup_amps = [i.amplitude for i in sup]
        pred_amps = [pred.contains(i).amplitude for i in sup]
        sup_phases = [i.phase for i in sup]
        pred_phases = [pred.contains(i).phase for i in sup]

        x = np.arange(len(labels))  # Label locations
        width = 0.35  # Width of the bars

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(14, 10))
        fig.suptitle(r"$\bf{" + str(self) + "}$")

        ax4.set_xlabel(r"$\bf{Actual: }$" + str(sup))
        ax5.set_xlabel(r"$\bf{Reconst: }$" + str(pred))
        ax1.set_ylabel(r"$\bf{Amplitude}$")
        ax4.set_ylabel(r"$\bf{Phase}$")
        ax3.set_title(r"$\bf{Mode}$ $\bf{Amplitudes}$")
        ax6.set_title(r"$\bf{Mode}$ $\bf{Phases}$")

        ax1.imshow(sup.superpose(), cmap='jet')
        ax2.imshow(pred.superpose(), cmap='jet')
        ax4.imshow(sup.phase_map(), cmap='jet')
        ax5.imshow(pred.phase_map(), cmap='jet')
        rects1 = ax3.bar(x - (width / 2), sup_amps, width, label='Actual', zorder=3)
        rects2 = ax3.bar(x + (width / 2), pred_amps, width, label='Reconstucted', zorder=3)
        rects3 = ax6.bar(x - (width / 2), sup_phases, width, label='Actual', zorder=3)
        rects4 = ax6.bar(x + (width / 2), pred_phases, width, label='Reconstucted', zorder=3)

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
        ax3.legend()
        ax6.set_xticks(x)
        ax6.set_xticklabels(labels)
        ax6.set_ylim(-np.pi, np.pi)
        ax6.legend()

        # auto_label(rects1, ax3)
        # auto_label(rects2, ax3)
        # auto_label(rects3, ax6)
        # auto_label(rects4, ax6)

        # fig.tight_layout()
        if save:
            logging.debug(f"Saving to 'Comparisons/{str(self)}/{str(sup)}.png'.")

            os.makedirs(f"Comparisons/{str(self)}", exist_ok=True) # Create directory for image
            plt.savefig(f"Comparisons/{str(self)}/{str(sup)}.png", bbox_inches='tight', pad_inches=0) # Save image

        else:
            plt.show()
        
        logging.info("Comparison complete!")

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
        logging.debug("Deleting model and freeing GPU memory.")
        print(log("[INFO] Deleting model and freeing GPU memory... "), end='')

        del self.model
        self.model = None
        K.clear_session()
        collected = gc.collect()

        logging.debug(f"GPU memory deleted. Collected: {collected}.")
        print(f"Done! Collected: {collected}.\n")




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
    #     logging.debug(message)
    # if level == "INFO":
    #     logging.info(message)
    #     print(Colour.OKBLUE + "[INFO] " + Colour.ENDC + message)
    # if level == "WARNING":
    #     logging.warning(message)
    #     print(Colour.WARNING + "[WARN] " + Colour.ENDC + message)
    # if level == "ERROR":
    #     logging.error(message)
    #     print(Colour.FAIL + "[FATAL] " + Colour.ENDC + message)
    # if level == "CRITICAL":
    #     logging.critical(message)
    #     print(Colour.FAIL + "[FATAL] " + Colour.ENDC + message)

    # text = message.replace(Colour.OKGREEN, '').replace(Colour.WARNING, '').replace(Colour.FAIL, '').replace(Colour.ENDC, '')

    # if text.contains("[INFO] "): logging.info(text.replace("[INFO] ", ''))
    # if text.contains("[WARN] "): logging.warning(text.replace("[WARN] ", ''))
    # if text.contains("[WARN] "): logging.warning(text.replace("[WARN] ", ''))

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
                    xylog=(0, 3 if height > 0 else -15),  # 3 points vertical offset
                    logcoords="offset points",
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

def optimize(param_name: str, param_range: str, plot: bool = True, save: bool = False) -> None:
    '''
    Loading / training multiple models and plotting comparison graphs of their performances.
    '''
    logging.info(f"Optimizing parameter '{param_name}' across range '{param_range}'.")
    print(log(f"[INFO] Optimizing parameter '{param_name}' across range '{param_range}'.\n"))

    models = []
    for test in param_range:
        m = ML(**{param_name: test})
        m.load(save_trained=False) # Load the model, and if the model does not exist then train and save it
        models.append(m) # Add the model to the list 

    if plot:
        logging.debug("Plotting optimisation graphs.")

        for time in (True, False):
            fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6), sharex=True, gridspec_kw={'hspace': 0})
            fig.suptitle(f"Comparing {param_name.replace('_', ' ').title()} by {'Elapsed Time' if time else 'Epoch'}")
            ax1.grid()
            ax2.grid()

            plt.xlim(0, 1E-9)
            plt.ylim(0, 1E-9)

            for m in models: m.plot(info=False, axes=(ax1, ax2), label=param_name.replace('_', ' ').title() + ": " + str(getattr(m, param_name)), elapsed_time=time)

            if save:
                logging.debug(f"Saving to 'Optimisation/Comparing {param_name.replace('_', ' ').title()} by {'Elapsed Time' if time else 'Epoch'}.png'.")
                plt.savefig(f"Optimisation/Comparing {param_name.replace('_', ' ').title()} by {'Elapsed Time' if time else 'Epoch'}.png", bbox_inches='tight', pad_inches=0) # Save image
            else:
                plt.show()

    logging.info("Optimisation complete.")




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
    logging.info("End of program.\n\n\n")
    logging.info("Start of program.")

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

    # Training and saving

    m = ML()
    m.train()
    m.save()

    # Loading saved model

    # data = GenerateData(3, 3, 0.2, 0.2)
    # data.new_stage()
    # data.new_stage()

    # for i in tqdm(data): model.compare(i, info=False, save=True)

    # for i in range(10):
    #     sup = data.get_random()
    #     m.compare(sup)

    optimize("repeats", [2**n for n in range(1, 9)], plot=True, save=True)
    optimize("batch_size", [2**n for n in range(9)], plot=True, save=True)
    optimize("optimiser", ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"], plot=True, save=True)
    optimize("learning_rate", [round(0.1**n, n) for n in range(8)], plot=True, save=True)
    optimize("learning_rate", [0.001 * n for n in range(1, 9)], plot=True, save=True)

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