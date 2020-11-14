import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide Tensorflow info, warning and error messages

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import cuda
import multiprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from Gaussian_Beam import Generate_Data, Superposition, Gaussian_Mode


class Model:
    '''
    The class 'Model' that represents a Keras model using datasets from Gaussian modes.
    '''
    def __init__(self, max_order: int = 1, number_of_modes: int = 1, amplitude_variation: float = 0, epochs: int = 30, repeats: int = 1):
        '''
        Initialise the class.
        '''
        self.max_order = max_order
        self.number_of_modes = number_of_modes
        self.amplitude_variation = amplitude_variation
        self.epochs = epochs
        self.repeats = repeats

        self.step_speed = 0.168
        self.batch_size = 30
        self.optimizer = "Adam"

        self.model = None
        self.loss_history, self.accuracy_history = None, None
        self.val_loss_history, self.val_accuracy_history = None, None
        self.solutions = None

        print("____________________| " + str(self) + " |____________________\n")

    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return self.__class__.__name__ + "(" + str(self.max_order) + ", " + str(self.number_of_modes) + ", " + str(self.amplitude_variation) + ", " + str(self.epochs) + ", " + str(self.repeats) + ")"

    def train(self):
        '''
        Train the model.
        '''
        # Initialisation

        (X_train, Y_train), (X_test, Y_test), num_classes, solutions = self.load_data(self.max_order, self.number_of_modes, self.amplitude_variation) # Load training and validation data
        self.model = self.create_model(num_classes, X_train.shape[1:]) # Create the model
        self.solutions = solutions

        # Training

        etl = (((len(X_train) / self.batch_size) * self.step_speed) * self.epochs) / 60

        print("Training model using " + str(self.repeats) + " datasets of " + str(len(X_train)) + " elements in batches of " + str(self.batch_size) + " for " + str(self.epochs) + " epochs... (ETL: " + str(int(round(etl / 60, 0))) + " hours " + str(int(round(etl % 60, 0))) + " minutes)")
        try:
            history_callback = self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=self.epochs, batch_size=self.batch_size)
        except KeyboardInterrupt:
            print("Aborted!")
        print("Done!\n")

        self.loss_history, self.accuracy_history = np.array(history_callback.history["loss"]), np.array(history_callback.history["accuracy"])
        self.val_loss_history, self.val_accuracy_history = np.array(history_callback.history["val_loss"]), np.array(history_callback.history["val_accuracy"])

        # Evaluation

        print("Evaluating...")
        scores = self.model.evaluate(X_test, Y_test, verbose=0)
        print("Done! Accuracy: %.2f%%\n" % (scores[1]*100))

    def load_data(self, max_order: int = 1, number_of_modes: int = 1, amplitude_variation: float = 0):
        '''
        Load training and testing data.
        '''
        print("Generating " + str(self.repeats) + " datasets of training data...")

        x_train = Generate_Data(max_order, number_of_modes, amplitude_variation, self.repeats, False)
        X_train = x_train.superpose()
        y_train, Y_train = x_train.get_outputs()

        print("Done!\n\nGenerating testing data...")

        x_test = Generate_Data(max_order, number_of_modes, amplitude_variation, 1, False)
        X_test = x_test.superpose()
        y_test, Y_test = x_test.get_outputs()

        print("Done!\n")

        Y_train = np_utils.to_categorical(Y_train)
        Y_test = np_utils.to_categorical(Y_test)

        solutions = np.array(y_train, dtype=object)

        return (X_train, Y_train), (X_test, Y_test), Y_train.shape[1], solutions

    def create_model(self, num_classes, shape, summary: bool = False):
        '''
        Create the Keras model in preparation for training.
        '''
        print("Generating model... (shape = " + str(shape) + ")")

        model = Sequential()

        model.add(Conv2D(32, (1, 1), input_shape=shape, padding='same'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Conv2D(64, (1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Conv2D(128, (1, 1), padding='same'))
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
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        print("Done!\n")
        if summary: print(model.summary())

        return model
    
    def plot(self):
        '''
        Plot the history of the model whilst training.
        '''
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
        fig.suptitle("Training and Validation History for " + str(self))

        t = np.arange(1, self.epochs + 1)
        
        ax1.plot(t, self.loss_history, label="Training Loss")
        ax2.plot(t, self.accuracy_history, label="Training Accuracy")
        ax1.plot(t, self.val_loss_history, label="Validation Loss")
        ax2.plot(t, self.val_accuracy_history, label="Validation Accuracy")

        # ax1.set_title("Loss")
        # ax2.set_title("Accuracy")

        plt.xlim(0, self.epochs)
        ax1.set_ylim(0, np.max(self.loss_history))
        ax2.set_ylim(0, 1)

        ax1.grid()
        ax2.grid()
        ax1.legend()
        ax2.legend()
    
    def show(self):
        '''
        Show the plot of the Gaussian mode.
        '''
        print("Plotting history...")

        self.plot()
        plt.show()

        print("Done!\n")
    
    def save(self):
        '''
        Save the history of the training to text files.
        '''
        if not self.check_trained(): return

        print("Saving model...")

        self.model.save("Models/" + str(self) + "/" + str(self) + ".h5")

        np.savetxt("Models/" + str(self) + "/loss_history.txt", self.loss_history, delimiter=",")
        np.savetxt("Models/" + str(self) + "/accuracy_history.txt", self.accuracy_history, delimiter=",")
        np.savetxt("Models/" + str(self) + "/val_loss_history.txt", self.val_loss_history, delimiter=",")
        np.savetxt("Models/" + str(self) + "/val_accuracy_history.txt", self.val_accuracy_history, delimiter=",")

        np.savetxt("Models/" + str(self) + "/solutions.txt", self.solutions, fmt="%s", delimiter=",")

        self.plot()
        plt.savefig("Models/" + str(self) + "/history.png", bbox_inches='tight', pad_inches=0)

        print("Done!\n")
    
    def load(self):
        '''
        Load a saved model.
        '''
        print("Loading model...")

        self.model = keras.models.load_model("Models/" + str(self) + "/" + str(self) + ".h5")

        self.loss_history = np.loadtxt("Models/" + str(self) + "/loss_history.txt", delimiter=",")
        self.accuracy_history = np.loadtxt("Models/" + str(self) + "/accuracy_history.txt", delimiter=",")
        self.val_loss_history = np.loadtxt("Models/" + str(self) + "/val_loss_history.txt", delimiter=",")
        self.val_accuracy_history = np.loadtxt("Models/" + str(self) + "/val_accuracy_history.txt", delimiter=",")

        self.solutions = np.loadtxt("Models/" + str(self) + "/solutions.txt", dtype=str, delimiter="\n")

        print("Done!\n")
    
    def check_trained(self):
        '''
        Check if the model has been trained yet.
        '''
        if self.model == None:
            print("Model not yet trained!")
            return False
        else:
            os.makedirs("Models/" + str(self), exist_ok=True) # Create directory for model
            return True
    
    def predict(self, data):
        '''
        Predict the superposition based on a 2D numpy array of the unknown optical cavity.
        '''
        data = np.array([data[..., np.newaxis]]) # Convert to the correct format for our neural network

        print("Predicting... (shape = " + str(data.shape) + ")")
        prediction = self.model.predict(data) # Make prediction using model (return index of superposition)
        print("Done!\n")

        answer = self.solutions[np.argmax(prediction, axis=1)]

        return list(eval(eval(str(answer))[0]))




##################################################
##########                              ##########
##########            MAIN              ##########
##########                              ##########
##################################################



def train_and_save(max_order, number_of_modes, amplitude_variation, epochs, repeats):
    model = Model(max_order, number_of_modes, amplitude_variation, epochs, repeats)
    model.train()
    model.save()

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')

    print("█████▀█████████████████████████████████████████████████████████████████████████████\n"
          "█─▄▄▄▄██▀▄─██▄─██─▄█─▄▄▄▄█─▄▄▄▄█▄─▄██▀▄─██▄─▀█▄─▄███▄─▀█▀─▄█─▄▄─█▄─▄▄▀█▄─▄▄─█─▄▄▄▄█\n"
          "█─██▄─██─▀─███─██─██▄▄▄▄─█▄▄▄▄─██─███─▀─███─█▄▀─█████─█▄█─██─██─██─██─██─▄█▀█▄▄▄▄─█\n"
          "▀▄▄▄▄▄▀▄▄▀▄▄▀▀▄▄▄▄▀▀▄▄▄▄▄▀▄▄▄▄▄▀▄▄▄▀▄▄▀▄▄▀▄▄▄▀▀▄▄▀▀▀▄▄▄▀▄▄▄▀▄▄▄▄▀▄▄▄▄▀▀▄▄▄▄▄▀▄▄▄▄▄▀\n")

    # model = Model(5, 2, 0.0, 30, 1)
    # model.train()
    # model.save()

    numbers = np.arange(1, 4)
    amplitude_variations = np.arange(0.0, 0.8, 0.2)
    repeats = np.arange(1, 6, 2)

    # for n in numbers:
    for r in repeats:
        for a in amplitude_variations:
            p = multiprocessing.Process(target=train_and_save, args=(5, 3, a, 30, r))
            p.start()
            p.join()
            # model = Model(5, 3, a, 30, r)
            # model.train()
            # model.save()
            # device = cuda.get_current_device()
            # device.reset()

    max_order = 5
    number_of_modes = 3
    amplitude_variation = 0.2
    epochs = 30
    repeats = 5

    # model = Model(max_order, number_of_modes, amplitude_variation, epochs, repeats)
    # model.train()
    # model.save()

    # model2 = Model(max_order, number_of_modes, amplitude_variation, epochs, repeats)
    # model2.load()
    # model2.show()
    # sup = Superposition([Gaussian_Mode(2,1), Gaussian_Mode(2,2), Gaussian_Mode(4,1)], amplitude_variation)
    # sup.show()
    # prediction = Superposition(model2.predict(sup.superpose()))
    # prediction.show()
    # print(prediction)