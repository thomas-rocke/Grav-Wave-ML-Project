import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide Tensorflow info, warning and error messages

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from Gaussian_Beam import Generate_Data

def load_data(max_order: int = 1, number_of_modes: int = 1, amplitude_variation: float = 0):
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

    print("Done!\nOne hot encoding outputs...")

    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)

    print("Done!\n")
    return (X_train, Y_train), (X_test, Y_test)

# Load data

(X_train, Y_train), (X_test, Y_test) = load_data(5, 3, 0.2)
num_classes = Y_train.shape[1]

# Create the model

print("Generating model...")
model = Sequential()

model.add(Conv2D(32, (1, 1), input_shape=X_train.shape[1:], padding='same'))
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

epochs = 50
optimizer = 'Adam'

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print("Done!\n")
# print(model.summary())

print("Training...")
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=64)
print("Done!\n")

# Final evaluation of the model

print("Evaluating...")
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Done! Accuracy: %.2f%%" % (scores[1]*100))