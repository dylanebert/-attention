import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Add
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model

class GlimpseNet(object):
    def __init__(self, config, images):
        self.original_size = config.original_size
        self.num_channels = config.num_channels
        self.num_scales = config.num_scales
        self.win_size = config.win_size
        self.batch_size = config.batch_size
        self.hg_size = config.hg_size
        self.hl_size = config.hl_size
        self.g_size = config.g_size
        self.sensor_size = config.sensor_size
        self.batch_size = config.batch_size
        self.images = images

    def get_glimpse(self, loc):
        imgs = K.reshape(self.images, (self.batch_size, self.original_size, self.original_size, self.num_channels))
        glimpse_all_scales = []
        for scale in range(1, self.num_scales + 1):
            glimpse_imgs = tf.image.extract_glimpse(imgs, [self.win_size * scale, self.win_size * scale], loc)
            glimpse_imgs = tf.image.resize_bilinear(glimpse_imgs, (self.win_size, self.win_size))
            glimpse_imgs = K.reshape(glimpse_imgs, [self.batch_size, self.win_size * self.win_size * self.num_channels])
            glimpse_all_scales.append(glimpse_imgs)
        return K.stack(glimpse_all_scales, axis=1)

    def __call__(self, loc):
        x = self.get_glimpse(loc)
        hg1 = Dense(self.hg_size, activation='relu')(x)
        hg2 = Dense(self.g_size)(hg1)
        hl1 = Dense(self.hl_size, activation='relu')(x)
        hl2 = Dense(self.g_size)(hl1)
        g = Add()([hg2, hl2])
        return g

class LocNet(object):
    def __init__(self, config):
        self.loc_dim = config.loc_dim
        self.input_dim = config.cell_output_size
        self.loc_std = config.loc_std
        self._sampling = True

    def __call__(self, input):
        mean = Dense(self.loc_dim)(input)
        mean = K.clip(mean, -1., 1.)
        mean = K.stop_gradient(mean)
        if self._sampling:
            loc = mean + K.random_normal((input.shape[0], self.loc_dim), stddev=self.loc_std)
            loc = K.clip(loc, -1., 1.)
        else:
            loc = mean
        loc = K.stop_gradient(loc)
        return loc, mean

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self, sampling):
        self._sampling = sampling
