import numpy as np
from itertools import combinations, chain, zip_longest, islice
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, Convolution2D, ZeroPadding2D
from keras.constraints import maxnorm
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from multiprocessing import cpu_count, Pool

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def grouper(iterable, n, fillvalue=None):
    '''
    Itertools grouper recipe.
    '''
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

class SinData(keras.utils.Sequence):
    def __init__(self, num_batches:int = 1000, batch_size:int = 128):
        self.num_batches = num_batches
        self.num_samples = num_batches * batch_size
        self.batch_size = batch_size

        self.input_data = np.linspace(0, 2*np.pi, self.num_samples)
        self.output_data = (np.sin(self.input_data) + 1)/2

        self.shuffle_data()
        self.batch_data()

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        '''
        Gets batch data at index
        '''
        input_batch = np.array(self.input_batches[index])[..., np.newaxis]
        output_batch = np.array(self.output_batches[index])[..., np.newaxis]
        return input_batch, output_batch
    
    def shuffle_data(self):
        '''
        Shuffles dataset order, maintaining the correspondance between input and output data
        '''
        zipped_data = list(zip(self.input_data, self.output_data))
        np.random.shuffle(zipped_data)
        self.shuffled_inputs, self.shuffled_outputs = zip(*zipped_data)
    
    def batch_data(self):
        '''
        Sorts shuffled data into correct sized batches
        '''
        self.input_batches = list(grouper(self.shuffled_inputs, self.batch_size))
        self.output_batches = list(grouper(self.shuffled_outputs, self.batch_size))

    def on_epoch_end(self):
        # Reshuffle and rebatch data every epoch
        self.shuffle_data()
        self.batch_data()



class SinModel():
    def __init__(self, data_generator:SinData, optimizer:str = "Adamax", learning_rate:float = 0.002, max_epochs:int=10):
        self.data_generator = data_generator
        sample_inputs, sample_outputs = self.data_generator[0]
        self.input_len = len(sample_inputs[0])
        self.output_len = len(sample_outputs[0])
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
    

    def make_model(self):
        model = Sequential()
        model.add(Dense(16, input_dim=self.input_len, activation="relu"))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(units=self.output_len))
        model.add(Activation('relu'))

        model.compile(loss="mse", optimizer=eval(self.optimizer + "(learning_rate=" + str(self.learning_rate) + ")"))

        return model
        
    def approx_sin(self):
        '''
        Use model predictions to approximate sin wave
        '''
        predictions = self.model.predict(np.array(self.data_generator.input_data)[..., np.newaxis], batch_size = self.data_generator.batch_size, workers=cpu_count(), use_multiprocessing=True)[:, 0]
        return predictions


    def train(self):
        self.model = self.make_model()
        fig, ax = plt.subplots(nrows=self.max_epochs)
        for n in range(self.max_epochs):
            self.model.fit(self.data_generator, validation_data=self.data_generator, validation_steps=1,
                            batch_size=self.data_generator.batch_size, use_multiprocessing=True, workers=cpu_count())
            
            ax[n].scatter(self.data_generator.input_data[::50], self.approx_sin()[::50], label='Epoch {}'.format(n))
        return ax
    
if __name__ == '__main__':
    fig, ax = plt.subplots(nrows=16)
    dat = SinData(batch_size=20, num_batches=300)
    sin_model = SinModel(dat, max_epochs=len(ax))
    ax = sin_model.train()
    x = np.linspace(0, 2*np.pi, 10000)
    y = (1 + np.sin(x))/2
    for i in range(len(ax)):
        ax[i].plot(x, y, label="Goal function")
    plt.legend()
    plt.show()