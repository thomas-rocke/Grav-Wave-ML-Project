import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide Tensorflow info, warning and error messages

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from Gaussian_Beam import Generate_Data


class Model:
    '''
    The class 'Model' that represents a Keras model using datasets from Gaussian modes.
    '''
    def __init__(self, max_order: int = 1, number_of_modes: int = 1, amplitude_variation: float = 0, epochs: int = 30):
        '''
        Initialise the class.
        '''
        self.max_order = max_order
        self.number_of_modes = number_of_modes
        self.amplitude_variation = amplitude_variation
        self.epochs = epochs
        self.optimizer = "Adam"

        self.model = None
        self.loss_history, self.accuracy_history = None, None
        self.val_loss_history, self.val_accuracy_history = None, None
    
    def __str__(self):
        '''
        Magic method for the str() function.
        '''
        return self.__class__.__name__ + "(" + str(self.max_order) + ", " + str(self.number_of_modes) + ", " + str(self.amplitude_variation) + ", " + str(self.epochs) + ")"

    def train(self):
        '''
        Train the model.
        '''
        # Initialisation

        (X_train, Y_train), (X_test, Y_test), num_classes = self.load_data(self.max_order, self.number_of_modes, self.amplitude_variation) # Load training and validation data
        self.model = self.create_model(num_classes, X_train.shape[1:]) # Create the model

        # Training

        print("Training...")
        history_callback = self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=self.epochs, batch_size=64)
        print("Done!\n")

        self.loss_history, self.accuracy_history = np.array(history_callback.history["loss"]), np.array(history_callback.history["accuracy"])
        self.val_loss_history, self.val_accuracy_history = np.array(history_callback.history["val_loss"]), np.array(history_callback.history["val_accuracy"])

        # Evaluation

        print("Evaluating...")
        scores = self.model.evaluate(X_test, Y_test, verbose=0)
        print("Done! Accuracy: %.2f%%\n" % (scores[1]*100))

        # # Saving

        # self.save_performance(history_callback)
        # self.save_model()

    def load_data(self, max_order: int = 1, number_of_modes: int = 1, amplitude_variation: float = 0):
        '''
        Load training and testing data.
        '''
        print("Generating training data...")

        x_train = Generate_Data(max_order, number_of_modes, amplitude_variation)
        X_train = np.array([i.superposition for i in x_train])[..., np.newaxis]
        y_train, Y_train = x_train.get_outputs()

        print("Done!\nGenerating testing data...")

        x_test = Generate_Data(max_order, number_of_modes, amplitude_variation)
        X_test = np.array([i.superposition for i in x_test])[..., np.newaxis]
        y_test, Y_test = x_test.get_outputs()

        print("Done!\n\nOne hot encoding outputs...")

        Y_train = np_utils.to_categorical(Y_train)
        Y_test = np_utils.to_categorical(Y_test)

        print("Done!\n")

        return (X_train, Y_train), (X_test, Y_test), Y_train.shape[1]

    def create_model(self, num_classes, shape, summary: bool = False):
        '''
        Create the Keras model in preparation for training.
        '''
        print("Generating model...")

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
    
    def plot_history(self):
        '''
        Plot the history of the model whilst training.
        '''
        print("Plotting history...")
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})

        fig.suptitle("Training and Validation History for " + str(self))

        t = np.arange(1, self.epochs + 1)

        # ax1.set_title("Loss")
        # ax2.set_title("Accuracy")
        
        ax1.plot(t, self.loss_history, label="Training Loss")
        ax2.plot(t, self.accuracy_history, label="Training Accuracy")
        ax1.plot(t, self.val_loss_history, label="Validation Loss")
        ax2.plot(t, self.val_accuracy_history, label="Validation Accuracy")

        plt.xlim(0, self.epochs)
        ax1.set_ylim(0, np.max(self.val_loss_history))
        ax2.set_ylim(0, 1)

        ax1.grid()
        ax1.legend()
        ax2.grid()
        ax2.legend()

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
        print("Done\n")
    
    def load(self, max_order: int, number_of_modes: int, amplitude_variation: float, epochs: int):
        '''
        Load a saved model.
        '''
        print("Loading model...")
        self.max_order = max_order
        self.number_of_modes = number_of_modes
        self.amplitude_variation = amplitude_variation
        self.epochs = epochs

        #self.model = keras.models.load_model("Models/" + str(self) + "/" + str(self) + ".h5")

        self.loss_history = np.loadtxt("Models/" + str(self) + "/loss_history.txt", delimiter=",")
        self.accuracy_history = np.loadtxt("Models/" + str(self) + "/accuracy_history.txt", delimiter=",")
        self.val_loss_history = np.loadtxt("Models/" + str(self) + "/val_loss_history.txt", delimiter=",")
        self.val_accuracy_history = np.loadtxt("Models/" + str(self) + "/val_accuracy_history.txt", delimiter=",")
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
        




##################################################
##########                              ##########
##########            MAIN              ##########
##########                              ##########
##################################################



if __name__ == '__main__':
    model = Model(max_order=5, number_of_modes=3, amplitude_variation=0.3, epochs=30)
    model.train()
    model.save()

    # model2 = Model()
    # model2.load(5, 3, 0.5, 50)
    # model2.plot_history()