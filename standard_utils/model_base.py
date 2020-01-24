#!/usr/bin/env python

# Keras imports to create a convolutional neural network using tensorflow on the low level
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Dropout
#to save the model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
#popular optimization strategy that uses gradient descent
from keras.optimizers import Adam
#add regularizer to limit overfitting
from keras import regularizers
#Gotta have them graphs
#helper class to define input shape and generate training images given image paths & steering angles
import importlib
import numpy as np


class BaseSequentialModel:

    def __init__(self, utils_file="utils.py"):
        self.model = Sequential() #linear stack of layers
        self.utils = importlib.import_module(utils_file)


    def model(self):
        print("Building model...")

        #based off of Nvidia's Dave 2 system
        #raw image height = 480, width = 640
        #NN input image shape (crop and resize raw) = 66x200x3: INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

        #normalize the image  to avoid saturation and make the gradients work better
        self.model.add(Lambda(lambda x: x/127.5-1.0, input_shape=self.utils.INPUT_SHAPE)) #127.5-1.0 = experimental value from udacity self driving car course
        #24 5x5 convolution kernels with 2x2 stride and activation function Exponential Linear Unit (to avoid vanishing gradient problem)
        self.model.add(Conv2D(24, 5, activation="elu", strides=2, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1(self.regularizer)))
        self.model.add(Conv2D(36, 5, activation="elu", strides=2, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1(self.regularizer)))
        self.model.add(Conv2D(48, 5, activation="elu", strides=2, kernel_initializer='he_normal', kernel_regularizer=regularizers.l1(self.regularizer)))
        self.model.add(Conv2D(64, 3, activation="elu")) #stride = 1x1
        self.model.add(Conv2D(64, 3, activation="elu")) #stride = 1x1

        self.model.add(Dropout(0.5)) #magic number from udacity self driving car course
        #turn convolutional feature maps into a fully connected ANN
        self.model.add(Flatten())
        self.model.add(Dense(100, activation="elu", kernel_initializer='he_normal', kernel_regularizer=regularizers.l1(self.regularizer)))
        self.model.add(Dense(50, activation="elu", kernel_initializer='he_normal', kernel_regularizer=regularizers.l1(self.regularizer)))
        self.model.add(Dense(10, activation="elu", kernel_initializer='he_normal', kernel_regularizer=regularizers.l1(self.regularizer)))
        self.model.add(Dense(1)) #No need for activation function because this is the output and it is not a probability

        self.model.summary() #print a summary representation of model

    def compile_model(self, loss='mean_squared_error', optimizer=Adam(1.0e-4)):
        self.model.compile(loss=loss, optimizer=optimizer)

    def train_model(self, x=None, y=None, x_train=None, y_train=None, x_test=None, y_test=None, datasets=None, data_dir=None, tensorboard=None, batch_size=40, validation_steps=1000, nb_epochs=10, steps_per_epoch=1500, regularizer=0.01):
        # filepath for save = rosey.epoch-loss.h5 (rosey-{epoch:03d}.h5 is another option)
        #saves epoch with the minimum val_loss
        checkpoint = ModelCheckpoint('model.{epoch:03d}-{val_loss:.2f}.h5', # filepath = working directory/
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            mode='auto')

        #batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
        #generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None,
        # class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0
        if x_train is not None:
            self.model.fit_generator(self.batch_generator(data_dir, datasets, x_train, y_train, batch_size, True),
                steps_per_epoch, nb_epochs, max_queue_size=1,
                validation_data=self.batch_generator(data_dir, datasets, x_test, y_test ,batch_size, False),
                #validation_steps=len(x_test), #Takes too long
                validation_steps= validation_steps,
                callbacks=[checkpoint, tensorboard],
                verbose=1)

    def batch_generator(self, data_dir, image_paths, steering_angles, batch_size, is_training):
        """
        Generate training image give image paths and associated steering angles
        """
        images = np.empty([batch_size, self.utils.IMAGE_HEIGHT, self.utils.IMAGE_WIDTH, self.utils.IMAGE_CHANNELS])
        steers = np.empty(batch_size)
        while True:
            i = 0
            for index in np.random.permutation(len(image_paths)):
                img = image_paths[index]
                steering_angle = steering_angles[index]
                # argumentation
                if is_training and np.random.rand() < 0.6:
                    image, steering_angle = self.utils.augument(data_dir, img, steering_angle)
                else:
                    image = self.utils.load_image(data_dir, img)
                # add the image and steering angle to the batch
                images[i] = self.utils.preprocess(image)
                steers[i] = steering_angle
                i += 1
                if i == batch_size:
                    break
            yield images, steers
