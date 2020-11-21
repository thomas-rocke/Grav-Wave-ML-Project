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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide Tensorflow info, warning and error messages
sys.path.insert(1, '../Simulation/Cavity Simulation') # Move to directory containing simulation files

from Gaussian_Beam import Hermite, Superposition, Laguerre, Generate_Data
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D




##################################################
##########                              ##########
##########           CLASSES            ##########
##########                              ##########
##################################################


class ML:
    '''
    The class 'ML' that represents a Keras model using datasets from Gaussian modes.
    '''
    def __init__(self, max_order: int = 1, number_of_modes: int = 1, amplitude_variation: float = 0.0, phase_variation: float = 0.0, repeats: int = 1):
        '''
        Initialise the class.
        '''
        self.max_order = max_order
        self.number_of_modes = number_of_modes
        self.amplitude_variation = amplitude_variation
        self.phase_variation = phase_variation
        self.repeats = repeats

        self.epochs = 1
        self.model = None
        self.history = {"loss": [], "binary_accuracy": [], "val_loss": [], "val_binary_accuracy": []}
        self.solutions = None

        self.max_epochs = 50
        self.start_number = 2
        self.step_speed = 0.06
        self.batch_size = 128
        self.success_performance = 0.95
        self.optimizer = "sgd"
        self.input_shape = (120, 120, 1)

        # print("                    " + (len(str(self)) + 4) * "_")
        print("____________________| " + str(self) + " |____________________\n")

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return repr(self)

    def __repr__(self):
        '''
        Magic method for the repr() function.
        '''
        return self.__class__.__name__ + "(" + str(self.max_order) + ", " + str(self.number_of_modes) + ", " + str(self.amplitude_variation) + ", " + str(self.phase_variation) + ", " + str(self.repeats) + ")"

    def train(self):
        '''
        Train the model.
        '''
        start_time = perf_counter()

        # Initialisation

        self.generate_prelim() # Generate preliminary data to determine all solutions (classes)
        self.model = self.create_model(summary=False) # Create the model

        # Training

        for number_of_modes in range(self.start_number, self.number_of_modes + 1):
            (train_inputs, train_outputs), (val_inputs, val_outputs) = self.load_data(number_of_modes) # Load training and validation data

            etl = (((len(train_inputs) / self.batch_size) * self.step_speed) * self.max_epochs) / 60
            print("[TRAIN] Training model using " + str(self.repeats) + " datasets of " + str(len(train_inputs)) + " elements in batches of " + str(self.batch_size) + " to a maximum epoch of " + str(self.max_epochs * (number_of_modes + 1 - self.start_number)) + " or an accuracy of " + str(int(self.success_performance * 100)) + "%... (ETL: " + str(int(round(etl / 60, 0))) + " hours " + str(int(round(etl % 60, 0))) + " minutes)")

            try:
                for i in range(self.epochs, (self.max_epochs * (number_of_modes + 1 - self.start_number)) + 1): #, "[TRAIN] Training with " + str(number_of_modes) + " modes"):
                    history_callback = self.model.fit(train_inputs, train_outputs, validation_data=(val_inputs, val_outputs), batch_size=self.batch_size, verbose=1)

                    for i in self.history: self.history[i].append(history_callback.history[i][0]) # Save performance of epoch
                    self.epochs += 1
    
                    if self.history["binary_accuracy"][-1] >= self.success_performance:
                        print("[TRAIN] " + str(int(self.success_performance * 100)) + "% acccuracy achieved at epoch " + str(self.epochs - 1) + "!")
                        break

            except KeyboardInterrupt:
                print("\n[WARN]  Aborted!")

            if self.epochs >= self.max_epochs * (number_of_modes + 1 - self.start_number): print("[WARN]  Reached max epoch of " + str(self.max_epochs * (number_of_modes + 1 - self.start_number)) + "!")
            print("[TRAIN] Done!\n")

            self.evaluate(val_inputs, val_outputs) # Evaluation

        print("[INFO] Training complete after " + str(int((perf_counter() - start_time) // 60)) + " minutes " + str(int((perf_counter() - start_time) % 60)) + " seconds.\n")

    def generate_prelim(self):
        '''
        Gather preliminary data for the training of the model.
        '''
        print("[INIT]  Generating preliminary data for model generation...")

        prelim_data = Generate_Data(self.max_order, self.number_of_modes, 0.0, 0.0, 1, info=False)
        self.solutions = prelim_data.get_classes()

        print("[INIT]  Done!\n")

    def load_data(self, number_of_modes: int = 1):
        '''
        Load training and testing data.
        '''
        try:
            print("[DATA]  Generating data for superpositions of " + str(number_of_modes) + " different modes...")
            print("[DATA]  |")

            train_data = Generate_Data(self.max_order, number_of_modes, self.amplitude_variation, self.phase_variation, self.repeats, info=False)
            train_inputs = train_data.get_inputs("[DATA]  |-> " + str(self.repeats) + " datasets of training data")
            train_outputs = train_data.get_outputs()

            print("[DATA]  |")

            val_data = Generate_Data(self.max_order, number_of_modes, self.amplitude_variation, self.phase_variation, 1, info=False)
            val_inputs = val_data.get_inputs("[DATA]  |-> 1 dataset of validation data")
            val_outputs = val_data.get_outputs()

            print("[DATA]  V")
            print("[DATA]  Done!\n")

        except MemoryError:
            print("[DATA] V")
            print("[FATAL] Memory overflow!\n")
            sys.exit()

        # If our loss function was 'categorical_crossentropy':
        # train_outputs = np_utils.to_categorical(train_outputs)
        # val_outputs = np_utils.to_categorical(val_outputs)

        return (train_inputs, train_outputs), (val_inputs, val_outputs)

    def create_model(self, summary: bool = False):
        '''
        Create the Keras model in preparation for training.
        '''
        print("[MODEL] Generating model... (classes = " + str(len(self.solutions)) + ", shape = " + str(self.input_shape) + ")")

        model = Sequential()

        # Conv2D: Matrix that traverses the image and blurs it
        # Dropout: Randomly sets input units to 0 to help prevent overfitting
        # MaxPooling2D: Downsamples the input representation

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

        model.add(Dense(256, kernel_constraint=maxnorm(3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(128, kernel_constraint=maxnorm(3)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(len(self.solutions)))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['binary_accuracy'])

        # We choose sigmoid and binary_crossentropy here because we have a multilabel neural network, which becomes K binary
        # classification problems. Using softmax would be wrong as it raises the probabiity on one class and lowers others.

        print("[MODEL] Done!\n")
        if summary: print(model.summary())
        if summary: keras.utils.plot_model(model, str(self), show_shapes=True)

        return model

    def evaluate(self, val_inputs, val_outputs):
        '''
        Evaluate the model using some validation data.
        '''
        print("[EVAL]  Evaluating...")
        scores = self.model.evaluate(val_inputs, val_outputs, verbose=0)
        print("[EVAL]  Done! Accuracy: " + str(round(scores[1] * 100, 1)) + "%, loss: " + str(round(scores[0], 3)) + ".\n")

    def plot(self):
        '''
        Plot the history of the model whilst training.
        '''
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
        fig.suptitle("Training and Validation History for " + str(self))

        t = np.arange(1, self.epochs)

        ax1.plot(t, self.history["loss"], label="Training Loss")
        ax2.plot(t, self.history["binary_accuracy"], label="Training Accuracy")
        ax1.plot(t, self.history["val_loss"], label="Validation Loss")
        ax2.plot(t, self.history["val_binary_accuracy"], label="Validation Accuracy")

        ax1.set_ylabel("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")

        plt.xlim(0, self.epochs)
        ax1.set_ylim(0, np.max(self.history["loss"]))
        ax2.set_ylim(0, 1)

        ax1.grid()
        ax2.grid()
        ax1.legend()
        ax2.legend()

    def show(self):
        '''
        Show the plot of the Gaussian mode.
        '''
        print("[PLOT]  Plotting history...")

        self.plot()
        plt.show()

        print("[PLOT]  Done!\n")

    def save(self):
        '''
        Save the history of the training to text files.
        '''
        if not self.check_trained(): return

        print("[SAVE]  Saving model...")

        self.model.save("Models/" + str(self) + "/" + str(self) + ".h5")

        np.savetxt("Models/" + str(self) + "/loss_history.txt", self.history["loss"], delimiter=",")
        np.savetxt("Models/" + str(self) + "/accuracy_history.txt", self.history["binary_accuracy"], delimiter=",")
        np.savetxt("Models/" + str(self) + "/val_loss_history.txt", self.history["val_loss"], delimiter=",")
        np.savetxt("Models/" + str(self) + "/val_accuracy_history.txt", self.history["val_binary_accuracy"], delimiter=",")

        np.savetxt("Models/" + str(self) + "/solutions.txt", self.solutions, fmt="%s", delimiter=",")

        self.plot()
        plt.savefig("Models/" + str(self) + "/history.png", bbox_inches='tight', pad_inches=0)

        print("[SAVE]  Done!\n")
    
    def load(self):
        '''
        Load a saved model.
        '''
        print("[LOAD]  Loading model...")

        self.model = keras.models.load_model("Models/" + str(self) + "/" + str(self) + ".h5")

        self.history["loss"] = np.loadtxt("Models/" + str(self) + "/loss_history.txt", delimiter=",")
        self.history["binary_accuracy"] = np.loadtxt("Models/" + str(self) + "/accuracy_history.txt", delimiter=",")
        self.history["val_loss"] = np.loadtxt("Models/" + str(self) + "/val_loss_history.txt", delimiter=",")
        self.history["val_binary_accuracy"] = np.loadtxt("Models/" + str(self) + "/val_accuracy_history.txt", delimiter=",")

        self.solutions = np.loadtxt("Models/" + str(self) + "/solutions.txt", dtype=str, delimiter="\n")
        self.solutions = [eval(i) for i in self.solutions]

        print("[LOAD]  Done!\n")
    
    def check_trained(self):
        '''
        Check if the model has been trained yet.
        '''
        if self.model == None:
            print("[ERROR] Model not yet trained!")
            return False
        else:
            os.makedirs("Models/" + str(self), exist_ok=True) # Create directory for model
            return True
    
    def predict(self, data, info: bool = True):
        '''
        Predict the superposition based on a 2D numpy array of the unknown optical cavity.
        '''
        if info: print("[PRED]  Predicting... (shape = " + str(data.shape) + ")")

        data = np.array([data[..., np.newaxis]]) # Convert to the correct format for our neural network

        prediction = self.model.predict(data)[0] # Make prediction using model (return index of superposition)
        prediction = [(self.solutions[i], prediction[i]) for i in range(len(prediction))]
        prediction = {i[0] : i[1] for i in prediction}
        prediction = {k : v for k, v in sorted(prediction.items(), key=lambda item: item[1])} # Sort list

        modes = list(prediction.keys())[-self.number_of_modes:]
        amplitudes = list(prediction.values())[-self.number_of_modes:]

        for i in range(len(modes)): modes[i].amplitude = amplitudes[i] # Set the amplitudes
        answer = Superposition(modes) # Normalise the amplitudes

        if info: print("[PRED]  Done! Prediction: " + str(answer) + "\n")
        return answer

    def compare(self, sup: Superposition):
        '''
        Plot given superposition against predicted superposition for visual comparison.
        '''
        pred = self.predict(sup.superpose())

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle(r"$\bf{" + str(self) + "}$")
        
        ax1.imshow(sup.superpose(), cmap='Greys_r')
        ax2.imshow(pred.superpose(), cmap='Greys_r')

        ax1.set_title(r"$\bf{Input: }$" + str(sup))
        ax2.set_title(r"$\bf{Prediction: }$" + str(pred))
        plt.axis('off')

        plt.show()




##################################################
##########                              ##########
##########          FUNCTIONS           ##########
##########                              ##########
##################################################


def process(max_order, number_of_modes, amplitude_variation, phase_variation, repeats):
    '''
    Runs a process that creates a model, trains it and then saves it. Can be run on a separate thread to free GPU memory after training for multiple training runs.
    '''
    print("[INFO]  Done!\n")

    model = ML(max_order, number_of_modes, amplitude_variation, phase_variation, repeats)
    model.train()
    model.save()

def train_and_save(max_order, number_of_modes, amplitude_variation, phase_variation, repeats):
    '''
    Starts a thread for training and saving of a model to ensure GPU memory is freed after training is complete.
    '''
    print("[INFO]  Starting process to ensure GPU memory is freed after taining is complete...")

    p = multiprocessing.Process(target=process, args=(max_order, number_of_modes, amplitude_variation, phase_variation, repeats))
    p.start()
    p.join()




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
    os.system('cls' if os.name == 'nt' else 'clear')

    print("█████▀█████████████████████████████████████████████████████████████████████████████\n"
          "█─▄▄▄▄██▀▄─██▄─██─▄█─▄▄▄▄█─▄▄▄▄█▄─▄██▀▄─██▄─▀█▄─▄███▄─▀█▀─▄█─▄▄─█▄─▄▄▀█▄─▄▄─█─▄▄▄▄█\n"
          "█─██▄─██─▀─███─██─██▄▄▄▄─█▄▄▄▄─██─███─▀─███─█▄▀─█████─█▄█─██─██─██─██─██─▄█▀█▄▄▄▄─█\n"
          "▀▄▄▄▄▄▀▄▄▀▄▄▀▀▄▄▄▄▀▀▄▄▄▄▄▀▄▄▄▄▄▀▄▄▄▀▄▄▀▄▄▀▄▄▄▀▀▄▄▀▀▀▄▄▄▀▄▄▄▀▄▄▄▄▀▄▄▄▄▀▀▄▄▄▄▄▀▄▄▄▄▄▀\n")

    train_and_save(3, 3, 0.2, 0.0, 50)
    train_and_save(4, 3, 0.2, 0.0, 50)
    train_and_save(5, 3, 0.2, 0.0, 20)

    model = ML(max_order = 4,
                  number_of_modes = 3,
                  amplitude_variation = 0.2,
                  phase_variation = 0.0,
                  repeats = 50
    )
    model.load()

    sup = Superposition([Hermite(3,1), Hermite(2,0), Hermite(1,0)], 0.0)
    model.compare(sup)
    # print("Test: " + str(sup))
    # prediction = model.predict(sup.superpose())
    # sup.show()
    # prediction.show()