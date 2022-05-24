import os
import time
import tensorflow as tf

import tensorflow.keras
import numpy as np

from tcn import TCN

import keras_tuner as kt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, InputLayer, Flatten, Reshape, Layer, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, DepthwiseConv2D
from tensorflow.keras.layers import Activation, Dropout, Conv1D, UpSampling1D, MaxPooling1D, AveragePooling1D

from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Layer, InputSpec
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers, activations, initializers, constraints, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import UnitNorm, Constraint

import tensorflow.experimental.numpy as tnp
import tensorflow_probability as tfp

# from AE import Sampling
# from AE import VAE

from sklearn import preprocessing as p


from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import matplotlib.pyplot as plt

class DenseMax(Layer):
    """
        Custom kera slayer 
    """
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 max_n=100,
                 lambertian=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.max_n = max_n
        self.lambertian = lambertian
                
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]


        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)

        # Maybe useful to store filter in object to study evolution after
        sorted_output = tnp.sort(output)
        threshold_for_each_batch = sorted_output[:, -(self.max_n + 1)]
        filter_bool = tnp.transpose(tnp.greater(tnp.transpose(output), threshold_for_each_batch))
        output = tnp.multiply(output, filter_bool)

        if self.lambertian:
            output = output.reshape(-1, 10, 10)
            output = output.reshape(-1, 100)

        return output

    def get_config(self):
        base_config = super(DenseMax, self).get_config()
        base_config['max_n'] = self.max_n
 

        return base_config

class SmoothEncoder(keras.Model):

    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs)
        self.true_freq_corr = np.load(os.path.join('toeplitz', 'toeplitz_100.npy')).reshape(1, 100, 100)
        self.test_freq = np.load(os.path.join('toeplitz', 'toeplitz.pkl'), allow_pickle=True)[:, :, :112]
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)[0]  # Forward pass, get only final output
            
            tested_freq = self(self.test_freq, training=False)[1] # Get output of encoded layer


            corr = tf.reshape(tfp.stats.correlation(tf.transpose(tested_freq)), (1, 100, 100)) # Not the good correlation
            tf.print(corr)
            freq_loss = self.compiled_loss(tf.constant(self.true_freq_corr), corr, regularization_losses=self.losses)
            #tf.print(freq_loss.shape)


            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses) + 0.2*freq_loss


        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

class Autoencoder():
    # This class should return the required autoencoder architecture
    def __init__(self, model, input_shape, latent_dim, dataset_type='log', max_n=None):
        self.model = model
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.dataset_type = dataset_type
        self.max_n = max_n

    def get_model(self):
        if self.model == 'conv_simple':
            return self.__conv_simple(max_n=self.max_n)

    def __conv_simple(self, max_n=100):

        def freq_smoothing_loss(y_true, y_pred):
            true_freq_corr = np.load(os.path.join('toeplitz', 'toeplitz_100.npy')).reshape(10000)
            test_freq = np.load(os.path.join('toeplitz', 'toeplitz.pkl'), allow_pickle=True)[:, :, :112]
            print(y_pred.shape)  # this is of shpe (none, 100) 
            # corrzlation matrix will need to be of shape (None, 100, 100) and be compared to true_freq_corr
            # batch size is y_true.shape[0]

            corr = tfp.stats.correlation(y_pred, y_pred, event_axis=1)

            corr = tf.reshape(corr, (10000,))

                    #### Why loss is getting nan values ? ?????
            return tnp.mean(true_freq_corr)-tnp.mean(corr)
        

        opt = keras.optimizers.Adam(learning_rate=0.0001)

        kernel_size = 3

        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
        dst = np.sqrt(x*x + y*y)

        # Considering 1 px = 150um
        sigma = np.sqrt(np.log(2)/2)
        muu = 0.000

        kernel_weights = np.exp(-((dst-muu)**2 / (2.0 * sigma**2)))

        kernel_weights = np.expand_dims(kernel_weights, axis=-1)
        kernel_weights = np.repeat(kernel_weights, 1, axis=-1)
        kernel_weights = np.expand_dims(kernel_weights, axis=-1)

        def gaussian_blur_filter(shape, dtype=None):
            f = np.array(kernel_weights)

            assert f.shape == shape
            return K.variable(f, dtype='float32')

        gaussian_blur = Conv2D(1, kernel_size, use_bias=False, kernel_initializer=gaussian_blur_filter, padding='same', trainable=False, name='gaussian_blur')

        inputs = Input((*self.input_shape, 1))

        x = Conv2D(96, kernel_size=11, padding='same', activation='relu', name='E_conv_1')(inputs)
        x = MaxPooling2D((2, 2), padding="same", name='E_pool_1')(x)
        x = Conv2D(64, kernel_size=5, padding='same', activation='relu', name='E_conv_2')(x)
        x = MaxPooling2D((2, 2), padding="same", name='E_pool_2')(x)
        x = Conv2D(64, kernel_size=5, padding='same', activation='relu', name='E_conv_3')(x)
        x = MaxPooling2D((2, 2), padding="same", name='E_pool_3')(x)
        x = Conv2D(48, kernel_size=7, padding='same', activation='relu', name='E_conv_4')(x)
        x = Flatten()(x)
        encoded = DenseMax(self.latent_dim, max_n=max_n, lambertian=False, kernel_constraint=UnitNorm(), name='Dense_maxn')(x)
        

        x = Reshape((10, 10, 1))(encoded)
        x = gaussian_blur(x)
        x = Reshape((100,))(x)
        
        x = Dense(16*14*48)(x)
        x = Reshape((16, 14, 48))(x)
        x = Conv2DTranspose(48, 7, strides=1, activation="relu", padding="same", name='D_conv_1')(x)
        x = UpSampling2D((2, 2), name='D_upsamp_1')(x)
        x = Conv2DTranspose(48, 5, strides=1, activation="relu", padding="same", name='D_conv_2')(x)
        x = UpSampling2D((2, 2), name='D_upsamp_2')(x)
        x = Conv2DTranspose(64, 5, strides=1, activation="relu", padding="same", name='D_conv_3')(x)
        x = UpSampling2D((2, 2), name='D_upsamp_3')(x)
        x = Conv2DTranspose(96, 9, strides=1, activation="relu", padding="same", name='D_conv_4')(x)

        decoded = Conv2D(1, (1, 1), activation="relu", padding="same", name='output')(x)

        autoencoder = SmoothEncoder(inputs=inputs, outputs=[decoded, encoded])
        
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder


   