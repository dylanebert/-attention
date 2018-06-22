import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Add, RepeatVector
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LSTM, LSTMCell
from keras.models import Model
from matplotlib import pyplot as plt
from config import Config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = Config()
data = np.load('/home/dylan/Documents/Visual-Attention-Model/MNIST/data/mnist_digit_sample_8dsistortions9x9.npz')

x_train = np.reshape(data['X_train'], (-1, 10000))
y_train = np.reshape(data['y_train'], (-1))
y_train = keras.utils.to_categorical(y_train, num_classes=config.num_classes)
x_va = np.reshape(data['X_valid'], (-1, 10000))
y_va = np.reshape(data['y_valid'], (-1))
y_va = keras.utils.to_categorical(y_va, num_classes=config.num_classes)
x_test = np.reshape(data['X_test'], (-1, 10000))
y_test = np.reshape(data['y_test'], (-1))
y_test = keras.utils.to_categorical(y_test, num_classes=config.num_classes)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_va.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')

input_shape = [config.original_size * config.original_size * config.num_channels]

#----------Begin glimpse network----------#
def get_glimpse(args):
    loc, images = args
    imgs = K.reshape(images, (-1, config.original_size, config.original_size, config.num_channels))
    glimpse_all_scales = []
    for scale in range(1, config.num_scales + 1):
        glimpse_imgs = tf.image.extract_glimpse(imgs, [config.win_size * scale, config.win_size * scale], loc)
        glimpse_imgs = tf.image.resize_bilinear(glimpse_imgs, (config.win_size, config.win_size))
        glimpse_imgs = K.reshape(glimpse_imgs, [-1, config.win_size * config.win_size * config.num_channels])
        glimpse_all_scales.append(glimpse_imgs)
    return K.stack(glimpse_all_scales, axis=1)

loc = Input(shape=(2,))
img = Input(shape=(input_shape))
glimpse = Lambda(get_glimpse)([loc, img])
hg1 = Dense(config.hg_size, activation='relu')(glimpse)
hg2 = Dense(config.g_size)(hg1)
hl1 = Dense(config.hl_size, activation='relu')(glimpse)
hl2 = Dense(config.g_size)(hl1)
g = Add()([hg2, hl2])
glimpse_net = Model(inputs=[loc, img], outputs=[g])
#----------End glimpse network----------#

def random_location(image):
    return K.random_uniform((K.shape(image)[0], 2), minval=-1., maxval=1.)

#----------Begin location network----------#
def randomize_loc(args):
    loc, mean = args
    mean = K.clip(mean, -1., 1.)
    mean = K.stop_gradient(mean)
    next_loc = mean + K.random_normal((K.shape(loc)[0], config.loc_dim), stddev=config.loc_std)
    next_loc = K.clip(next_loc, -1., 1.)
    next_loc = K.stop_gradient(next_loc)
    return next_loc

loc_in = Input(shape=(2,))
loc_mean = Dense(config.loc_dim)(loc_in)
next_loc = Lambda(randomize_loc)([loc_in, loc_mean])
loc_net = Model(inputs=[loc_in], outputs=[next_loc, loc_mean])
#----------End location network----------#

loc_mean_arr = []
sampled_loc_arr = []

def get_next_input(output, i):
    loc, loc_mean = loc_net(output)
    gl_next = glimpse_net(loc)
    loc_mean_arr.append(loc_mean)
    sampled_loc_arr.append(loc)
    return gl_next

x = Input(shape=input_shape)
y = Input(batch_shape=[None])

#x_expanded = K.tile(x, [config.M, 1])
#y_expanded = K.tile(y, [config.M])

init_loc = Lambda(random_location)(x)
glimpse = glimpse_net([init_loc, x])

lstm_cell = LSTMCell(config.cell_size)

predictions = Dense(config.num_classes, activation='softmax')(x)

model = Model(x, predictions)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='Adam', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=config.batch_size, epochs=1, verbose=1, validation_data=(x_va, y_va))
