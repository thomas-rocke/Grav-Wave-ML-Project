##################################################
##########                              ##########
##########              ML              ##########
##########        CLASSIFICATION        ##########
##########                              ##########
##################################################

# TODO Header for file

# Imports
import sys
import os
import random

#from tensorflow.python.keras.applications.vgg16 import VGG16
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide Tensorflow info, warning and error messages

from Gaussian_Beam import Hermite, Superposition, Laguerre
from DataHandling import Generate_Data, Dataset
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import tensorflow as tf
import keras
from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, Convolution2D, ZeroPadding2D
from keras.constraints import maxnorm
import keras.backend as K
from keras.optimizers import SGD, Adadelta




##################################################
##########                              ##########
##########           CLASSES            ##########
##########                              ##########
##################################################


class ML:
    '''
    The class 'ML' that represents a Keras model using datasets from Gaussian modes.
    '''
    def __init__(self, max_order: int = 1, number_of_modes: int = 1, amplitude_variation: float = 0.0, phase_variation: float = 0.0, noise_variation: float = 0.0, exposure: tuple = (0.0, 1.0), repeats: int = 1):
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

        self.model = None
        self.solutions = None
        self.input_shape = None
        self.pixels = None
        self.epoch = 1
        self.history = {"loss": [], "binary_accuracy": [], "val_loss": [], "val_binary_accuracy": []}

        self.max_epochs = 200
        self.start_number = 2
        self.step_speed = 0.067
        self.batch_size = 128
        self.success_performance = 0.99
        self.optimizer = Adadelta()
        # self.optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

        # print("                    " + (len(str(self)) + 4) * "_")
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
        return self.__class__.__name__ + "(" + str(self.max_order) + ", " + str(self.number_of_modes) + ", " + str(self.amplitude_variation) + ", " + str(self.phase_variation) + ", " + str(self.noise_variation) + ", " + str(self.exposure) + ", " + str(self.repeats) + ")"

    def train(self):
        '''
        Train the model.
        '''
        start_time = perf_counter()

        # Initialisation

        self.model = self.create_model(summary=False) # Generate preliminary data to determine all solutions (classes) and create the model

        # Training

        for number_of_modes in range(self.start_number, self.number_of_modes + 1):
            (train_inputs, train_outputs), (val_inputs, val_outputs) = self.load_data(number_of_modes) # Load training and validation data

            etl = (((len(train_inputs) / self.batch_size) * self.step_speed) * self.max_epochs) / 60
            text("[TRAIN] Training model using " + str(self.repeats) + " datasets of " + str(len(train_inputs)) + " elements in batches of " + str(self.batch_size) + " to a maximum epoch of " + str(self.max_epochs * (number_of_modes + 1 - self.start_number)) + " or an accuracy of " + str(self.success_performance * 100) + "%... (ETL: " + str(int(round(etl / 60, 0))) + " hours " + str(int(round(etl % 60, 0))) + " minutes)")

            #data = []
            try:
                for i in range(self.epoch, (self.max_epochs * (number_of_modes + 1 - self.start_number)) + 1): #, "[TRAIN] Training with " + str(number_of_modes) + " modes"):
                    history_callback = self.model.fit(train_inputs, train_outputs, validation_data=(val_inputs, val_outputs), batch_size=self.batch_size, verbose=1)

                    for i in self.history: self.history[i].append(history_callback.history[i][0]) # Save performance of epoch
                    self.epoch += 1

                    if self.history["binary_accuracy"][-1] >= self.success_performance:
                        text("[TRAIN] " + str(self.success_performance * 100) + "% acccuracy achieved at epoch " + str(self.epoch - 1) + ".")
                        break

                    # if self.epoch == 2: 
                    #     data = self.plot(False)
                    #     plt.show(block=False)
                    #     plt.pause(0.01)
                    # self.plot(False)
                    # self.update(data)

            except KeyboardInterrupt:
                text("\n[WARN] Aborted!")
            
            plt.show()

            if self.epoch >= self.max_epochs * (number_of_modes + 1 - self.start_number): text("[WARN] Reached max epoch of " + str(self.max_epochs * (number_of_modes + 1 - self.start_number)) + "!")
            text("[TRAIN] Done!\n")

            self.evaluate(val_inputs, val_outputs) # Evaluation

        text("[INFO] Training complete after " + str(int((perf_counter() - start_time) // 60)) + " minutes " + str(int((perf_counter() - start_time) % 60)) + " seconds.\n")

    def generate_prelim(self):
        '''
        Gather preliminary data for the training of the model.
        '''
        text("[INIT] Generating preliminary data for model generation...")

        prelim_data = Generate_Data(self.max_order, self.number_of_modes, info=False)
        self.solutions = prelim_data.get_classes()
        self.input_shape = (prelim_data[0].pixels, prelim_data[0].pixels, 1)

        text("[INIT] Done!\n")

    def load_data(self, number_of_modes: int = 1):
        '''
        Load training and testing data.
        '''
        try:
            text("[DATA] Generating data for superpositions of " + str(number_of_modes) + " different modes...")
            text("[DATA] |")

            train_data = Generate_Data(self.max_order, number_of_modes, self.amplitude_variation, self.phase_variation, self.noise_variation, self.exposure, self.repeats, info=False)
            train_inputs = train_data.get_inputs(Colour.OKGREEN + "[DATA] " + Colour.ENDC + "|-> " + str(self.repeats) + " datasets of training data")
            train_outputs = train_data.get_outputs()

            text("[DATA] |")

            val_data = Generate_Data(self.max_order, number_of_modes, self.amplitude_variation, self.phase_variation, self.noise_variation, self.exposure, 1, info=False)
            val_inputs = val_data.get_inputs(Colour.OKGREEN + "[DATA] " + Colour.ENDC + "|-> 1 dataset of validation data")
            val_outputs = val_data.get_outputs()

            text("[DATA] V")
            text("[DATA] Done!\n")

        except MemoryError: # TODO Is not called when memory overflow occurs
            text("[DATA] V")
            text("[FATAL] Memory overflow!\n")
            sys.exit()

        # If our loss function was 'categorical_crossentropy':
        # train_outputs = np_utils.to_categorical(train_outputs)
        # val_outputs = np_utils.to_categorical(val_outputs)

        return (train_inputs, train_outputs), (val_inputs, val_outputs)

    def loss(self, y_true, y_pred):
        '''
        Loss function for assessing the performance of the neural network.
        '''
        K.print_tensor(y_true)
        K.print_tensor(y_pred)

        loss = K.square(y_true - y_pred)
        return K.sum(loss)

        # modes_true = [i.copy() for i in self.solutions]
        # K.print_tensor(y_true)
        # K.print_tensor(y_true[0])
        # for i in range(len(modes_true)): modes_true[i].amplitude = K.eval(y_true)[i]
        # sup_true = Superposition(*modes_true)

        # modes_pred = [i.copy() for i in self.solutions]
        # for i in range(len(modes_pred)): modes_pred[i].amplitude = K.eval(y_pred)[i]
        # sup_pred = Superposition(*modes_pred)

        # return K.mean(K.square(sup_true.superpose() - sup_pred.superpose()), axis=1)

    def create_model(self, summary: bool = False):
        '''
        Create the Keras model in preparation for training.
        '''
        if self.input_shape == None: self.generate_prelim()

        text("[MODEL] Generating model... (classes = " + str(len(self.solutions)) + ", shape = " + str(self.input_shape) + ")")

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
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=["binary_accuracy"])

        # We choose sigmoid and binary_crossentropy here because we have a multilabel neural network, which becomes K binary
        # classification problems. Using softmax would be wrong as it raises the probabiity on one class and lowers others.

        text("[MODEL] Done!\n")
        if summary: text(model.summary())
        # if summary: keras.utils.plot_model(model, str(self), show_shapes=True) # TODO Make work using packages

        return model

    def evaluate(self, val_inputs, val_outputs):
        '''
        Evaluate the model using some validation data.
        '''
        text("[EVAL] Evaluating...")

        scores = self.model.evaluate(val_inputs, val_outputs, verbose=0)

        text("[EVAL] Done! Accuracy: " + str(round(scores[1] * 100, 1)) + "%, loss: " + str(round(scores[0], 3)) + ".\n")

    def update(self, data):
        '''
        Update the history plot.
        '''
        t = np.arange(1, self.epoch)
        data[0].set_xdata(t)
        data[1].set_xdata(t)
        data[2].set_xdata(t)
        data[3].set_xdata(t)
        data[0].set_ydata(self.history["loss"])
        data[1].set_ydata(self.history["binary_accuracy"])
        data[2].set_ydata(self.history["val_loss"])
        data[3].set_ydata(self.history["val_binary_accuracy"])
        plt.pause(0.0001)

    def plot(self, info: bool = True):
        '''
        Plot the history of the model whilst training.
        '''
        if info: text("[PLOT] Plotting history...")

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
        fig.suptitle("Training and Validation History for " + str(self))

        t = np.arange(1, self.epoch)

        data1 = ax1.plot(t, self.history["loss"], label="Training Loss")[0]
        data2 = ax2.plot(t, self.history["binary_accuracy"], label="Training Accuracy")[0]
        data3 = ax1.plot(t, self.history["val_loss"], label="Validation Loss")[0]
        data4 = ax2.plot(t, self.history["val_binary_accuracy"], label="Validation Accuracy")[0]

        ax1.set_ylabel("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")

        plt.xlim(0, self.epoch)
        ax1.set_ylim(0, np.max(self.history["loss"]))
        ax2.set_ylim(0, 1)

        ax1.grid()
        ax2.grid()
        ax1.legend()
        ax2.legend()

        if info:
            plt.show()
            text("[PLOT] Done!\n")

        return (data1, data2, data3, data4)

    def save(self):
        '''
        Save the history of the training to text files.
        '''
        if not self.check_trained(): return

        text("[SAVE] Saving model...")

        self.model.save("Models/" + str(self) + "/" + str(self) + ".h5")

        np.savetxt("Models/" + str(self) + "/loss_history.txt", self.history["loss"], delimiter=",")
        np.savetxt("Models/" + str(self) + "/accuracy_history.txt", self.history["binary_accuracy"], delimiter=",")
        np.savetxt("Models/" + str(self) + "/val_loss_history.txt", self.history["val_loss"], delimiter=",")
        np.savetxt("Models/" + str(self) + "/val_accuracy_history.txt", self.history["val_binary_accuracy"], delimiter=",")

        np.savetxt("Models/" + str(self) + "/solutions.txt", self.solutions, fmt="%s", delimiter=",")

        self.plot(info=False)
        plt.savefig("Models/" + str(self) + "/history.png", bbox_inches='tight', pad_inches=0)

        text("[SAVE] Done!\n")

    def load(self):
        '''
        Load a saved model.
        '''
        text("[LOAD] Loading model...")

        self.model = keras.models.load_model("Models/" + str(self) + "/" + str(self) + ".h5")

        self.history["loss"] = np.loadtxt("Models/" + str(self) + "/loss_history.txt", delimiter=",")
        self.history["binary_accuracy"] = np.loadtxt("Models/" + str(self) + "/accuracy_history.txt", delimiter=",")
        self.history["val_loss"] = np.loadtxt("Models/" + str(self) + "/val_loss_history.txt", delimiter=",")
        self.history["val_binary_accuracy"] = np.loadtxt("Models/" + str(self) + "/val_accuracy_history.txt", delimiter=",")

        self.solutions = np.loadtxt("Models/" + str(self) + "/solutions.txt", dtype=str, delimiter="\n")
        self.solutions = [eval(i.replace("HG", "Hermite")) for i in self.solutions]

        text("[LOAD] Done!\n")

    def check_trained(self):
        '''
        Check if the model has been trained yet.
        '''
        if self.model == None:
            text("[FATAL] Model not yet trained!")
            return False
        else:
            os.makedirs("Models/" + str(self), exist_ok=True) # Create directory for model
            return True
    
    def predict(self, data, threshold: float = 0.5, info: bool = True):
        '''
        Predict the superposition based on a 2D numpy array of the unknown optical cavity.
        '''
        start_time = perf_counter()
        if info: text("[PRED] Predicting... (shape = " + str(data.shape) + ")")

        formatted_data = np.array([data[..., np.newaxis]]) # Convert to the correct format for our neural network
        prediction = self.model.predict(formatted_data)[0] # Make prediction using model (return index of superposition)

        modes = []
        for i in range(len(prediction)): # For all values of prediction
            if info: text("[PRED] " + str(self.solutions[i]) + ": " + str(round(prediction[i], 3)) + Colour.FAIL + (prediction[i] > threshold) * " ***" + Colour.ENDC)

            if prediction[i] > threshold: # If the prediction is above a certain threshold
                modes.append(self.solutions[i].copy()) # Copy the corresponding solution to modes
                modes[-1].amplitude = prediction[i] # Set that modes amplitude to the prediction value
                # modes[-1].phase = (i // 11) / 10

        # prediction = [(self.solutions[i], prediction[i]) for i in range(len(prediction))]
        # prediction = {i[0] : i[1] for i in prediction}
        # prediction = {k : v for k, v in sorted(prediction.items(), key=lambda item: item[1])} # Sort list

        # modes = list(prediction.keys())[-self.number_of_modes:]
        # amplitudes = list(prediction.values())[-self.number_of_modes:]

        # for i in range(len(modes)): modes[i].amplitude = amplitudes[i] # Set the amplitudes

        if len(modes) == 0:
            text("[FATAL] Prediction failed! A threshold of " + str(threshold) + " is likely too high.\n")
            sys.exit()

        answer = Superposition(*modes) # Normalise the amplitudes

        self.calculate_phase(data, answer)

        if info: text("[PRED] Done! Took " + str(round((perf_counter() - start_time) * 1000, 3)) + " milliseconds.")
        if info: text("[PRED] Reconstructed: " + str(answer) + "\n")

        return answer

    def compare(self, sup: Superposition, save: bool = False):
        '''
        Plot given superposition against predicted superposition for visual comparison.
        '''
        text("[PRED] Actual: " + str(sup))

        pred = self.predict(sup.superpose())

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(r"$\bf{" + str(self) + "}$")
        
        ax1.imshow(sup.superpose(), cmap='Greys_r')
        ax2.imshow(pred.superpose(), cmap='Greys_r')

        ax1.set_title(r"$\bf{Actual: }$" + repr(sup))
        ax2.set_title(r"$\bf{Reconstructed: }$" + repr(pred))
        plt.axis('off')

        if save: plt.savefig("Images/" + str(self) + ".png", bbox_inches='tight', pad_inches=0)
        else: plt.show()

    def calculate_phase(self, data, superposition: Superposition):
        '''
        Calculate the phases for modes in the superposition that minimises the MSE to the original data.
        '''
        for mode in superposition:
            sups = []
            for phase in range(11):
                mode.phase = round((phase / 10) * (2 * np.pi), 2)
                sups.append(np.mean(np.square(superposition.superpose() - data)))
            mode.phase = round((np.argmin(sups) / 10) * (2 * np.pi), 2)




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




##################################################
##########                              ##########
##########          FUNCTIONS           ##########
##########                              ##########
##################################################


def text(message):
    '''
    Print message in the format given.
    '''
    if "INFO" in message: print(Colour.OKBLUE + message[:7] + Colour.ENDC + message[7:])
    elif "WARN" in message: print(Colour.WARNING + message[:7] + Colour.ENDC + message[7:])
    elif "FATAL" in message: print(Colour.FAIL + message[:7] + Colour.ENDC + message[7:])
    else: print(Colour.OKGREEN + message[:7] + Colour.ENDC + message[7:])

def process(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure, repeats):
    '''
    Runs a process that creates a model, trains it and then saves it. Can be run on a separate thread to free GPU memory after training for multiple training runs.
    '''
    text("[INFO] Done!\n")

    model = ML(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure, repeats)
    model.train()
    model.save()

def train_and_save(max_order: int = 1, number_of_modes: int = 1, amplitude_variation: float = 0.0, phase_variation: float = 0.0, noise_variation: float = 0.0, exposure: tuple = (0.0, 1.0), repeats: int = 1):
    '''
    Starts a thread for training and saving of a model to ensure GPU memory is freed after training is complete.
    '''
    text("[INFO] Starting process to ensure GPU memory is freed after taining is complete...")

    p = multiprocessing.Process(target=process, args=(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure, repeats))
    p.start()
    p.join()

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


def get_model_error(model, data_object:Generate_Data, test_number:int=10, sup:Superposition=None):
    '''
    Tests the accuracy of the model from data contained within data_object
    
    Assumes a gaussian resultant error through the Central Limit Theorem
    
    Tests on test_percent of the data given in the data_object
    '''
    if sup == None:
        test_data = random.sample(data_object.combs, test_number) # Select superpositions randomly from combs
    else:
        test_data = np.array([sup.copy() for i in range(test_number)]) # Make several copies of the target superposition

    test_data = np.array([data_object.randomise_amp_and_phase(i) for i in test_data]) # Randomise all amps and phases
    
    model_predictions = np.array([model.predict(data.superpose()) for data in test_data]) # Predict superpositions through models
    
    
    test_amps = np.array([[mode.amplitude for mode in sup] for sup in test_data])
    model_amps = np.array([[mode.amplitude for mode in sup] for sup in model_predictions])
    
    amp_err = (np.sum([(model_amps[i] - test_amps[i])**2 for i in range(len(test_amps))])/(len(test_amps) - 1))**0.5 # Predicts amplitude error assuming error is constant throughout multivariate space
    
    test_phases = np.array([[mode.phase for mode in sup] for sup in test_data])
    model_phases = np.array([[mode.phase for mode in sup] for sup in model_predictions])
    
    phase_err = (np.sum([(model_phases[i] - test_phases[i])**2 for i in range(len(test_phases))])/(len(test_phases) - 1))**0.5 # Predicts phase error assuming error is constant throughout multivariate space
    
    test_imgs = np.array([sup.superpose() for sup in test_data])
    model_imgs = np.array([sup.superpose() for sup in model_predictions])
    
    img_err = (np.sum([(model_imgs[i] - test_imgs[i])**2 for i in range(len(test_imgs))])/(len(test_imgs) - 1))**0.5 # Predicts img error assuming error is constant throughout multivariate space
    

    return amp_err, phase_err, img_err



##################################################
##########                              ##########
##########            MAIN              ##########
##########                              ##########
##################################################


# This is supposed to automatically allocate memory to the GPU when it is needed instead of reserving the full space.
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

if __name__ == '__main__':
    max_order = 3
    number_of_modes = 5
    amplitude_variation = 0.2
    phase_variation = 0.0
    noise_variation = 0.0
    exposure = (0.0, 1.0)
    repeats = 50

    model = ML(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure, repeats)
    #sys.path.insert(1, '../System') # Move to directory containing simulation files
    model.load()

    data = Generate_Data(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure)

    errs = get_model_error(model, data, 0.5)

    print(errs[0])
    print(errs[1])
    plt.imshow(errs[2])
    plt.show()

    '''
    os.system('cls' if os.name == 'nt' else 'clear')

    print("█████▀█████████████████████████████████████████████████████████████████████████████\n"
          "█─▄▄▄▄██▀▄─██▄─██─▄█─▄▄▄▄█─▄▄▄▄█▄─▄██▀▄─██▄─▀█▄─▄███▄─▀█▀─▄█─▄▄─█▄─▄▄▀█▄─▄▄─█─▄▄▄▄█\n"
          "█─██▄─██─▀─███─██─██▄▄▄▄─█▄▄▄▄─██─███─▀─███─█▄▀─█████─█▄█─██─██─██─██─██─▄█▀█▄▄▄▄─█\n"
          "▀▄▄▄▄▄▀▄▄▀▄▄▀▀▄▄▄▄▀▀▄▄▄▄▄▀▄▄▄▄▄▀▄▄▄▀▄▄▀▄▄▀▄▄▄▀▀▄▄▀▀▀▄▄▄▀▄▄▄▀▄▄▄▄▀▄▄▄▄▀▀▄▄▄▄▄▀▄▄▄▄▄▀\n")

    max_order = 3
    number_of_modes = 3
    amplitude_variation = 0.4
    phase_variation = 0.0
    noise_variation = 0.0
    exposure = (0.0, 1.0)
    repeats = 100

    # Training and saving

    train_and_save(max_order, number_of_modes, amplitude_variation, 0.0, noise_variation, exposure, repeats)
    # train_and_save(5, 3, amplitude_variation, 0.0, noise_variation, exposure, 30)
    # train_and_save(3, 5, amplitude_variation, 0.0, noise_variation, exposure, 30)
    # train_and_save(5, 5, amplitude_variation, 0.0, noise_variation, exposure, 30)

    # Loading saved model

    model = ML(max_order, number_of_modes, amplitude_variation, 0.0, noise_variation, exposure, repeats)
    model.load()

    # Generating test data for comparisons

    data = Generate_Data(max_order, number_of_modes, amplitude_variation, phase_variation, noise_variation, exposure)
    while True:
        sup = data.get_random()
        model.compare(sup)

    # sup = Superposition(Hermite(1,2), Hermite(2,0), Hermite(0,1))
    # prediction = model.predict(sup.superpose())
    # sup.show()
    # prediction.show()'''