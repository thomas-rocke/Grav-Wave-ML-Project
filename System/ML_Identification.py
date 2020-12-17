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

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide Tensorflow info, warning and error messages

from Gaussian_Beam import Hermite, Superposition, Laguerre
from DataHandling import Generate_Data, Dataset
from time import perf_counter
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
    def __init__(self, max_order: int = 3, number_of_modes: int = 3, amplitude_variation: float = 0.5, phase_variation: float = 1.0, noise_variation: float = 0.1, exposure: tuple = (0.0, 1.0), repeats: int = 100, batch_size: int = 128, optimizer: str = "Adamax", learning_rate: float = 0.002):
        '''
        Initialise the class.
        '''
        self.max_order = max_order
        self.number_of_modes = number_of_modes
        self.amplitude_variation = amplitude_variation
        self.phase_variation = phase_variation
        self.noise_variation = noise_variation
        self.exposure = exposure
        self.repeats = repeats
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.max_epochs = 100
        self.start_number = 2
        self.step_speed = 0.072
        self.success_loss = 0.001
        self.history = {"time": [], "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        self.model = None

        print(Colour.HEADER + Colour.BOLD + "____________________| " + str(self) + " |____________________\n" + Colour.ENDC)

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return repr(self)

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + "(" + str(self.max_order) + ", " + str(self.number_of_modes) + ", " + str(self.amplitude_variation) + ", " + str(self.phase_variation) + ", " + str(self.noise_variation) + ", " + str(self.exposure) + ", " + str(self.repeats) + ", " + str(self.batch_size) + ", " + str(self.optimizer) + ", " + str(self.learning_rate) + ")"

    def exists(self):
        '''
        Check if the model has been trained before.
        '''
        return os.path.exists("Models/" + str(self))

    def trained(self):
        '''
        Check if the model has been trained before.
        '''
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
        print(text("[INIT] Generating preliminary data for model generation... "), end='')

        prelim_data = Generate_Data(self.max_order, self.number_of_modes, info=False)
        self.solutions = prelim_data.get_classes()
        self.input_shape = (prelim_data[0].pixels, prelim_data[0].pixels, 1)

        print("Done!")
        print(text("[INIT] Generating model (input shape = " + str(self.input_shape) + ", classes = " + str(len(self.solutions)) + ", optimizer: " + self.optimizer + ")... "), end='')

        model = Sequential()

        # Conv2D: Matrix that traverses the image and blurs it
        # Dropout: Randomly sets input units to 0 to help prevent overfitting
        # MaxPooling2D: Downsamples the input representation

        # Using the VGG16 convolutional neural net (CNN) architecture which was used to win ILSVR (Imagenet) competition in 2014.
        # It is considered to be one of the best vision model architectures to date.

        # Our custom architecture

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

        model.add(Dense(units=len(self.solutions)))
        model.add(Activation('sigmoid'))

        # model = VGG16(self.input_shape, len(self.solutions)) # Override model with VGG16 model
        model.compile(loss="mse", optimizer=eval(self.optimizer + "(learning_rate=" + str(self.learning_rate) + ")"), metrics=[self.accuracy])

        # We choose sigmoid and binary_crossentropy here because we have a multilabel neural network, which becomes K binary
        # classification problems. Using softmax would be wrong as it raises the probabiity on one class and lowers others.

        print("Done!\n")
        if summary: text(model.summary())
        if summary: keras.utils.plot_model(model, str(self), show_shapes=True) # TODO Make work using packages

        return model

    def train(self, info: bool = False):
        '''
        Train the model.
        '''
        if self.trained():
            print(text("[WARN] Trained model already exists!\n"))
            self.load()
            return

        start_time = perf_counter()
        self.model = self.create_model(summary=False) # Generate preliminary data to determine all solutions (classes) and create the model

        for number_of_modes in range(self.start_number, self.number_of_modes + 1):
            (train_inputs, train_outputs), (val_inputs, val_outputs) = self.load_data(number_of_modes) # Load training and validation data

            etl = (((len(train_inputs) / self.batch_size) * self.step_speed) * self.max_epochs) / 60
            print(text("[TRAIN] Training stage " + str(number_of_modes - 1) + "/" + str(self.number_of_modes - 1) + "..."))
            print(text("[TRAIN] |"))
            print(text("[TRAIN] |-> Dataset             : " + str(len(train_inputs)) + " data elements in batches of " + str(self.batch_size) + "."))
            print(text("[TRAIN] |-> Success Condition   : A loss of " + str(self.success_loss) + "."))
            print(text("[TRAIN] |-> Terminate Condition : Reaching epoch " + str(len(self.history["loss"]) + self.max_epochs) + " or 5 consecutive epochs of stagnation."))
            print(text("[TRAIN] |-> Maximum Duration    : " + str(int(round(etl / 60, 0))) + " hours " + str(int(round(etl % 60, 0))) + " minutes."))
            print(text("[TRAIN] |"))

            n = 0
            try:
                iterator = tqdm(range(self.max_epochs), text("[TRAIN] |-> Training "))
                for n in iterator:
                    history_callback = self.model.fit(train_inputs, train_outputs, validation_data=(val_inputs, val_outputs), batch_size=self.batch_size, verbose=int(info))

                    for i in self.history:
                        if i == "time": self.history[i].append(perf_counter() - start_time) # Save time elapsed since training began
                        else: self.history[i].append(history_callback.history[i][0]) # Save performance of epoch

                    iterator.set_description(text("[TRAIN] |-> Loss: " + str(round(self.history["loss"][-1], 3)) + " - Accuracy: " + str(round(self.history["accuracy"][-1] * 100, 1)) + "% "))

                    if isnan(self.history["loss"][-1]): # Loss is nan so training has failed
                        print(text("\n[TRAIN] V"))
                        print(text("[FATAL] Training failed! Gradient descent diverged at epoch " + str(len(self.history["loss"])) + ".\n"))
                        sys.exit()
                    elif self.history["loss"][-1] < self.success_loss: # Loss has reached success level
                        iterator.close()
                        print(text("[TRAIN] |"))
                        print(text("[TRAIN] |-> " + str(self.success_loss) + " loss achieved at epoch " + str(len(self.history["loss"])) + "."))
                        break
                    elif n >= 4: # Check there is enough history to check for stagnation
                        if np.all(round(self.history["loss"][-5], 3) <= np.round(self.history["loss"][-4:], 3)): # Learning has stagnated
                            iterator.close()
                            print(text("[TRAIN] |"))
                            print(text("[WARN]  |-> Learning stagnated at epoch " + str(len(self.history["loss"])) + "."))
                            break

            except KeyboardInterrupt:
                print(text("[TRAIN] |"))
                print(text("[WARN]  |-> Aborted at epoch " + str(len(self.history["loss"]) + 1) + "!"))

            if n == self.max_epochs - 1: # Reached max epoch
                print(text("[TRAIN] |"))
                print(text("[WARN]  |-> Reached max epoch of " + str(len(self.history["loss"])) + "!"))

            print(text("[TRAIN] |-> Evaluating : "), end='')
            scores = self.model.evaluate(val_inputs, val_outputs, verbose=0)
            print("Loss: " + str(round(scores[0], 3)) + " - Accuracy: " + str(round(scores[1] * 100, 1)) + "%.")
            print(text("[TRAIN] V"))
            print(text("[TRAIN] Done!\n"))

            del train_inputs, train_outputs, val_inputs, val_outputs # Releasing RAM memory

        print(text("[INFO] Training complete after " + str(int((perf_counter() - start_time) // 60)) + " minutes " + str(int((perf_counter() - start_time) % 60)) + " seconds.\n"))

    def load_data(self, number_of_modes: int = 1):
        '''
        Load training and testing data.
        '''
        try:
            print(text("[DATA] Generating data for superpositions of " + str(number_of_modes) + " different modes..."))
            print(text("[DATA] |"))

            train_data = Generate_Data(self.max_order, number_of_modes, self.amplitude_variation, self.phase_variation, self.noise_variation, self.exposure, self.repeats, info=False)
            train_inputs = train_data.get_inputs(text("[DATA] |-> " + str(self.repeats) + " datasets of training data"))
            train_outputs = train_data.get_outputs()

            print(text("[DATA] |"))

            val_data = Generate_Data(self.max_order, number_of_modes, self.amplitude_variation, self.phase_variation, self.noise_variation, self.exposure, 1, info=False)
            val_inputs = val_data.get_inputs(text("[DATA] |-> 1 dataset of validation data"))
            val_outputs = val_data.get_outputs()

            print(text("[DATA] V"))
            print(text("[DATA] Done!\n"))

        except MemoryError:
            print(text("[DATA] V"))
            print(text("[FATAL] Memory overflow!\n"))
            sys.exit()

        return (train_inputs, train_outputs), (val_inputs, val_outputs)

    def plot(self, info: bool = True, axes: tuple = None, label: str = False, elapsed_time: bool = False):
        '''
        Plot the history of the model whilst training.
        '''
        if info: print(text("[PLOT] Plotting history..."))

        if elapsed_time: t = self.history["time"]
        else: t = np.arange(1, len(self.history["loss"]) + 1)

        if axes == None:
            fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
            fig.suptitle("Training and Validation History for " + str(self))
            ax1.grid()
            ax2.grid()

            ax1.plot(t, self.history["loss"], label="Training Loss")[0]
            ax2.plot(t, self.history["accuracy"], label="Training Accuracy")[0]
            ax1.plot(t, self.history["val_loss"], label="Validation Loss")[0]
            ax2.plot(t, self.history["val_accuracy"], label="Validation Accuracy")[0]
        else:
            ax1, ax2 = axes
            if not label: label = str(self)

            ax1.plot(t, self.history["loss"], label=label)[0]
            ax2.plot(t, self.history["accuracy"], label=label)[0]

        ax1.set_ylabel("Loss")
        if elapsed_time: ax2.set_xlabel("Elapsed Time (s)")
        else: ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")

        plt.xlim(0, t[-1])
        # ax1.set_ylim(0, np.max(self.history["loss"]))
        ax2.set_ylim(0, 1)
        ax1.legend()
        ax2.legend()

        if info:
            plt.show()
            print(text("[PLOT] Done!\n"))

        return (ax1, ax2)

    def save(self, save_trained: bool = True):
        '''
        Save the history of the training to text files.
        '''
        os.makedirs("Models/" + str(self), exist_ok=True) # Create directory for model

        print(text("[SAVE] Saving model... "), end='')

        if save_trained: self.model.save("Models/" + str(self) + "/" + str(self) + ".h5")

        for i in self.history: np.savetxt("Models/" + str(self) + "/" + i + ".txt", self.history[i], delimiter=",")
        np.savetxt("Models/" + str(self) + "/solutions.txt", self.solutions, fmt="%s", delimiter=",")

        self.plot(info=False, elapsed_time=False)
        plt.savefig("Models/" + str(self) + "/history_epoch.png", bbox_inches='tight', pad_inches=0)
        self.plot(info=False, elapsed_time=True)
        plt.savefig("Models/" + str(self) + "/history_elapsed_time.png", bbox_inches='tight', pad_inches=0)

        print("Done!\n")

    def load(self, save_trained: bool = True):
        '''
        Load a saved model.
        '''
        if not self.exists():
            print(text("[WARN] Model does not exist! Will now train and save.\n"))
            self.train()
            self.save(save_trained)
            if not save_trained: self.free()
            return
        elif not self.trained():
            print(text("[WARN] Model exists but has not been trained! Will only load history.\n"))

        print(text("[LOAD] Loading model... "), end='')

        if self.trained(): self.model = keras.models.load_model("Models/" + str(self) + "/" + str(self) + ".h5", custom_objects={"metrics": [self.accuracy]})

        for i in self.history: self.history[i] = np.loadtxt("Models/" + str(self) + "/" + i + ".txt", delimiter=",")
        self.solutions = np.loadtxt("Models/" + str(self) + "/solutions.txt", dtype=str, delimiter="\n")
        self.solutions = [eval(i.replace("HG", "Hermite")) for i in self.solutions]

        print("Done!\n")
    
    def predict(self, data, threshold: float = 0.2, info: bool = True):
        '''
        Predict the superposition based on a 2D numpy array of the unknown optical cavity.
        '''
        if not self.exists():
            print(text("[WARN] Model does not exist!\n"))
            return
        elif not self.trained():
            print(text("[WARN] Model has not been trained!\n"))
            return

        start_time = perf_counter()
        if info: print(text("[PRED] Predicting... (shape = " + str(data.shape) + ")"))
        if info: print(text("[PRED] |"))

        formatted_data = np.array([data[..., np.newaxis]]) # Convert to the correct format for our neural network
        prediction = self.model.predict(formatted_data)[0] # Make prediction using model (return index of superposition)

        modes = []
        for i in range(len(prediction) // 2): # For all values of prediction
            if info: print(text("[PRED] |-> " + str(self.solutions[i]) + ": " + str(round(prediction[i], 3)) + Colour.FAIL + int(prediction[i] > threshold) * " ***" + Colour.ENDC))

            if prediction[i] > threshold: # If the prediction is above a certain threshold
                modes.append(self.solutions[i].copy()) # Copy the corresponding solution to modes
                modes[-1].amplitude = prediction[i] # Set that modes amplitude to the prediction value
                modes[-1].phase = np.arccos(prediction[i + (len(prediction) // 2)]) # Set the phase to the corresponding modes phase

        if info: print(text("[PRED] V"))
        if len(modes) == 0:
            print(text("[FATAL] Prediction failed! A threshold of " + str(threshold) + " is likely too high.\n"))
            sys.exit()

        answer = Superposition(*modes) # Normalise the amplitudes

        # self.calculate_phase(data, answer)

        if info: print(text("[PRED] Done! Took " + str(round((perf_counter() - start_time) * 1000, 3)) + " milliseconds."))
        if info: print(text("[PRED] Reconstructed: " + str(answer) + "\n"))

        return answer

    def compare(self, sup: Superposition, info: bool = True, save: bool = False):
        '''
        Plot given superposition against predicted superposition for visual comparison.
        '''
        if info: print(text("[PRED] Actual: " + str(sup)))
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

        auto_label(rects1, ax3)
        auto_label(rects2, ax3)
        auto_label(rects3, ax6)
        auto_label(rects4, ax6)

        # fig.tight_layout()
        if save:
            os.makedirs("Comparisons/" + str(self), exist_ok=True) # Create directory for image
            plt.savefig("Comparisons/" + str(self) + "/" + str(sup) + ".png", bbox_inches='tight', pad_inches=0)
        else:
            plt.show()

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
        print(text("[INFO] Deleting model and freeing GPU memory... "), end='')

        del self.model
        self.model = None
        K.clear_session()
        collected = gc.collect()

        print("Done! Collected: " + str(collected) + ".\n")




##################################################
##########                              ##########
##########          FUNCTIONS           ##########
##########                              ##########
##################################################


def text(message):
    '''
    Return message in the format given.
    '''
    message = message.replace("->",         Colour.OKCYAN   + "->"      + Colour.ENDC)
    message = message.replace(" |",         Colour.OKCYAN   + " |"      + Colour.ENDC)
    message = message.replace(" V",         Colour.OKCYAN   + " V"      + Colour.ENDC)
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
    Attach a text label above each bar in rects displaying its height.
    '''
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -15),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom")

def process(**kwargs):
    '''
    Runs a process that creates a model, trains it and then saves it. Can be run on a separate thread to free GPU memory after training for multiple training runs.
    '''
    print("Done!\n")

    model = ML(**kwargs)
    model.train()
    model.save(save_trained=False)

def train_and_save(**kwargs):
    '''
    Starts a thread for training and saving of a model to ensure GPU memory is freed after training is complete.
    '''
    print(text("[INFO] Starting process to ensure GPU memory is freed after training is complete... "), end='')

    p = multiprocessing.Process(target=process, args=kwargs)
    p.start()
    p.join()

def get_model_error(model, data_object:Generate_Data, test_number:int=10, sup:Superposition=None):
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

def optimize(param_name: str, param_range: str, plot: bool = True) -> None:
    '''
    Loading / training multiple models and plotting comparison graphs of their performances.
    '''
    print(text("[INFO] Optimizing parameter '" + param_name + "' across range '" + str(param_range) + "'.\n"))

    models = []
    for test in param_range:
        m = ML(**{param_name: test})
        m.load(save_trained=False) # Load the model, and if the model does not exist then train and save it
        models.append(m) # Add the model to the list 

    if plot:
        for time in (True, False):
            fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
            fig.suptitle(f"Comparing {param_name} by {'Elapsed Time' if time else 'Epoch'}")
            ax1.grid()
            ax2.grid()

            for m in models: m.plot(info=False, axes=(ax1, ax2), label=param_name + ": " + str(getattr(m, param_name)), elapsed_time=time)
            plt.show()




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

    max_order = 3
    number_of_modes = 3
    amplitude_variation = 0.5
    phase_variation = 1.0
    noise_variation = 0.1
    exposure = (0.0, 1.0)
    repeats = 128

    # Training and saving

    # train_and_save(3, 3, amplitude_variation, phase_variation, noise_variation, exposure, 20, 128)

    optimize("batch_size", [2**n for n in range(9)], plot=False)
    optimize("optimizer", ["SGD", "RMSprop", "Adam", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"], plot=False)
    optimize("learning_rate", [round(0.1**n, n) for n in range(8)], plot=False)
    optimize("learning_rate", [0.001 * n for n in range(1, 9)], plot=False)
    optimize("repeats", [2**n for n in range(9)], plot=False)

    # for r in [20, 50, 100]:
    #     train_and_save(3, 3, amplitude_variation, phase_variation, noise_variation, exposure, r)
    #     train_and_save(3, 5, amplitude_variation, phase_variation, noise_variation, exposure, r)
    #     train_and_save(5, 3, amplitude_variation, phase_variation, noise_variation, exposure, r)
    #     train_and_save(5, 5, amplitude_variation, phase_variation, noise_variation, exposure, r)

    # Loading saved model

    data = Generate_Data(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure)

    model = ML(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure, repeats)
    model.load()

    # for i in tqdm(data): model.compare(i, info=False, save=True)

    while True:
        sup = data.get_random()
        model.compare(sup)

    # print(tf.config.list_physical_devices())
    # with tf.device("gpu:0"):
    #     model = ML(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure, repeats)
    #     try:
    #         model.load()
    #     except:
    #         model.train()
    #         model.save

    #     # Generating test data for comparisons

    #     data = Generate_Data(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure)

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

    # data = Generate_Data(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure)

    # errs = get_model_error(model, data, 0.5)

    # print(errs[0])
    # print(errs[1])
    # plt.imshow(errs[2])
    # plt.show()