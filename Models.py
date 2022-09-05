import os
import time
import tensorflow as tf

import tensorflow.keras
import numpy as np

from tcn import TCN

import keras_tuner as kt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, InputLayer, Flatten, Reshape, Layer, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, DepthwiseConv2D, LeakyReLU
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

class Autoencoder():
    # This class should return the required autoencoder architecture
    def __init__(self, model, input_shape, latent_dim, dataset_type='log', max_n=None, toeplitz_spec=None, toeplitz_true=None):
        self.model = model
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.dataset_type = dataset_type
        self.max_n = max_n
        self.toeplitz_spec = toeplitz_spec
        self.toeplitz_true = toeplitz_true

        self.true_freq_corr = tf.convert_to_tensor(np.load(os.path.join('toeplitz', self.toeplitz_spec)))
        self.test_freq = tf.convert_to_tensor(np.load(os.path.join('toeplitz', self.toeplitz_true), allow_pickle=True)[:, :input_shape[0], :input_shape[1]])

    def get_model(self):
        if self.model == 'conv_simple':
            return self.__conv_simple(max_n=self.max_n)
        if self.model == 'conv_small':
            return self.__conv_small(max_n=self.max_n)
        if self.model == 'tcn_ae':
            return self.__tcn_ae()
    

    def __conv_simple(self, max_n=100):

        def fn_smoothing(y_true, y_pred):

            pred_freq_corr = autoencoder(self.test_freq)[1]

            def t(a): return tf.transpose(a)

            x = pred_freq_corr
            mean_t = tf.reduce_mean(x, axis=1, keepdims=True)
            #cov_t = x @ t(x)
            cov_t = ((x-mean_t) @ t(x-mean_t))/(pred_freq_corr.shape[1]-1)
            cov2_t = tf.linalg.diag(1/tf.sqrt(tf.linalg.diag_part(cov_t)))
            cor = cov2_t @ cov_t @ cov2_t

            # Fin how to compute autocorrelation matrix
            loss = tf.keras.losses.mean_squared_error(self.true_freq_corr, cor)
            # May need to return output with batch size 

            return loss

        def normalized_mse(y_true, y_pred):
            loss = tf.keras.losses.mean_squared_error(y_true/255.0, y_pred/255.0)

            return loss


                

        opt = keras.optimizers.RMSprop(learning_rate=0.001, epsilon=1e-8)
        optadam = keras.optimizers.Adam(learning_rate=0.0005)


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
        # x = Dropout(0.1)(x)
        x = MaxPooling2D((2, 2), padding="same", name='E_pool_2')(x)
        x = Conv2D(48, kernel_size=5, padding='same', activation='relu', name='E_conv_3')(x)
        # x = Dropout(0.1)(x)
        x = MaxPooling2D((2, 2), padding="same", name='E_pool_3')(x)
        x = Conv2D(48, kernel_size=7, padding='same', activation='relu', name='E_conv_4')(x)
        # x = Dropout(0.1)(x)
        x = Flatten()(x)
        x = DenseMax(self.latent_dim, max_n=max_n, lambertian=False, kernel_constraint=UnitNorm(), name='Dense_maxn')(x)
        

        x = Reshape((10, 10, 1))(x)
        x = gaussian_blur(x)
        encoded = Reshape((100,))(x)
        
        x = Dense(int(self.input_shape[0]/8)*int(self.input_shape[1]/8)*48)(encoded)
        x = Reshape((int(self.input_shape[0]/8), int(self.input_shape[1]/8), 48))(x)
        x = Conv2DTranspose(48, 7, strides=1, activation='relu', padding="same", name='D_conv_1')(x)
        #x = LeakyReLU(alpha=0.3)(x)
        x = UpSampling2D((2, 2), name='D_upsamp_1')(x)
        # x = Dropout(0.1)(x)
        x = Conv2DTranspose(48, 5, strides=1, activation='relu', padding="same", name='D_conv_2')(x)
        #x = LeakyReLU(alpha=0.3)(x)
        x = UpSampling2D((2, 2), name='D_upsamp_2')(x)
        # x = Dropout(0.1)(x)
        x = Conv2DTranspose(64, 5, strides=1, activation='relu', padding="same", name='D_conv_3')(x)
        #x = LeakyReLU(alpha=0.3)(x)
        x = UpSampling2D((2, 2), name='D_upsamp_3')(x)
        # x = Dropout(0.1)(x)
        x = Conv2DTranspose(96, 11, strides=1, activation='relu', padding="same", name='D_conv_4')(x)
        #x = LeakyReLU(alpha=0.3)(x)

        decoded = Conv2D(1, (1, 1), activation='relu', padding="same", name='output')(x)
        #decoded = LeakyReLU(alpha=0.3)(x)

        autoencoder = Model(inputs=inputs, outputs=[decoded, encoded])
        print(autoencoder.summary())
        
        autoencoder.compile(optimizer=optadam, loss=[normalized_mse, fn_smoothing], loss_weights=[0.95, 0.05])
        
        return autoencoder

    def __conv_small(self, max_n=100):

        def fn_smoothing(y_true, y_pred):

            pred_freq_corr = autoencoder(self.test_freq)[1]

            def t(a): return tf.transpose(a)

            x = pred_freq_corr
            mean_t = tf.reduce_mean(x, axis=1, keepdims=True)
            #cov_t = x @ t(x)
            cov_t = ((x-mean_t) @ t(x-mean_t))/(pred_freq_corr.shape[1]-1)
            cov2_t = tf.linalg.diag(1/tf.sqrt(tf.linalg.diag_part(cov_t)))
            cor = cov2_t @ cov_t @ cov2_t

            # Fin how to compute autocorrelation matrix
            loss = tf.keras.losses.mean_squared_error(self.true_freq_corr, cor)
            # May need to return output with batch size 

            return loss

        def normalized_mse(y_true, y_pred):
            loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

            return loss


                

        opt = keras.optimizers.RMSprop(learning_rate=0.001, epsilon=1e-8)
        optadam = keras.optimizers.Adam(learning_rate=0.001)


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
            return K.variable(f)

        gaussian_blur = Conv2D(1, kernel_size, use_bias=False, kernel_initializer=gaussian_blur_filter, padding='same', trainable=False, name='gaussian_blur')

        inputs = Input((*self.input_shape, 1))

        x = Conv2D(64, kernel_size=13, padding='same', activation='relu', name='E_conv_1')(inputs)
        x = MaxPooling2D((2, 2), padding="same", name='E_pool_1')(x)
        x = Conv2D(48, kernel_size=5, padding='same', activation='relu', name='E_conv_2')(x)
        # x = Dropout(0.1)(x)
        x = MaxPooling2D((2, 2), padding="same", name='E_pool_2')(x)
        x = Conv2D(32, kernel_size=5, padding='same', activation='relu', name='E_conv_3')(x)
        # x = Dropout(0.1)(x)
        x = MaxPooling2D((2, 2), padding="same", name='E_pool_3')(x)
        x = Conv2D(16, kernel_size=7, padding='same', activation='relu', name='E_conv_4')(x)
        x = MaxPooling2D((2, 2), padding="same", name='E_pool_4')(x)


        # x = Dropout(0.1)(x)
        x = Flatten()(x)
        dnmax = DenseMax(self.latent_dim, max_n=max_n, lambertian=False, kernel_constraint=UnitNorm(), name='Dense_maxn')(x)
        

        x = Reshape((10, 10, 1))(dnmax)
        x = gaussian_blur(x)
        encoded = Reshape((100,), dtype=tf.float32)(x)
        
        x = Dense(int(self.input_shape[0]/16)*int(self.input_shape[1]/16)*16)(encoded)
        x = Reshape((int(self.input_shape[0]/16), int(self.input_shape[1]/16), -1))(x)
        x = UpSampling2D((2, 2), name='D_upsamp_0')(x)
        x = Conv2DTranspose(16, 7, strides=1, activation='relu', padding="same", name='D_conv_1')(x)
        #x = LeakyReLU(alpha=0.3)(x)
        x = UpSampling2D((2, 2), name='D_upsamp_1')(x)
        # x = Dropout(0.1)(x)
        x = Conv2DTranspose(32, 5, strides=1, activation='relu', padding="same", name='D_conv_2')(x)
        #x = LeakyReLU(alpha=0.3)(x)
        x = UpSampling2D((2, 2), name='D_upsamp_2')(x)
        # x = Dropout(0.1)(x)
        x = Conv2DTranspose(48, 5, strides=1, activation='relu', padding="same", name='D_conv_3')(x)
        #x = LeakyReLU(alpha=0.3)(x)
        x = UpSampling2D((2, 2), name='D_upsamp_3')(x)
        # x = Dropout(0.1)(x)
        x = Conv2DTranspose(64, 13, strides=1, activation='relu', padding="same", name='D_conv_4')(x)
        #x = LeakyReLU(alpha=0.3)(x)

        decoded = Conv2D(1, (1, 1), activation='relu', padding="same", name='output', dtype=tf.float32)(x)
        #decoded = LeakyReLU(alpha=0.3)(x)

        autoencoder = Model(inputs=inputs, outputs=[decoded, encoded])
        
        print(autoencoder.summary())
        autoencoder.compile(optimizer=optadam, loss=['mse', fn_smoothing], loss_weights=[0.95, 0.05])
        
        return autoencoder


    def __tcn_ae(self):

        ts_dimension = 1
        dilations = (4, 8, 16, 64, 96)
        nb_filters = 20
        kernel_size = 20
        nb_stacks = 1
        padding = 'same'
        dropout_rate = 0.00
        filters_conv1d = 32
        activation_conv1d = 'linear'
        latent_sample_rate = 8
        pooler = AveragePooling1D
        lr = 0.001
        conv_kernel_init = 'glorot_normal'
        loss = 'logcosh'
        
                        
        tensorflow.keras.backend.clear_session()
        sampling_factor = latent_sample_rate
        i = Input(batch_shape=(None, 31920, ts_dimension))

        # Put signal through TCN. Output-shape: (batch,sequence length, nb_filters)
        tcn_enc = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks, dilations=dilations, 
                      padding=padding, use_skip_connections=True, dropout_rate=dropout_rate, return_sequences=True,
                      kernel_initializer=conv_kernel_init, name='tcn-enc')(i)

        # Now, adjust the number of channels...
        enc_flat = Conv1D(filters=filters_conv1d, kernel_size=1, activation=activation_conv1d, padding=padding)(tcn_enc)

        ## Do some average (max) pooling to get a compressed representation of the time series (e.g. a sequence of length 8)
        enc_pooled = pooler(pool_size=sampling_factor, strides=None, padding='valid', data_format='channels_last')(enc_flat)
        
        # If you want, maybe put the pooled values through a non-linear Activation
        enc_out = Activation("relu")(enc_pooled)



        encoder = Model(inputs=[i], outputs=[enc_out])

        # Now we should have a short sequence, which we will upsample again and then try to reconstruct the original series
        j = Input(batch_shape=(None, None, filters_conv1d)) 
        
        dec_upsample = UpSampling1D(size=sampling_factor)(j)

        dec_reconstructed = TCN(nb_filters=nb_filters, kernel_size=kernel_size, nb_stacks=nb_stacks, dilations=dilations, 
                                padding=padding, use_skip_connections=True, dropout_rate=dropout_rate, return_sequences=True,
                                kernel_initializer=conv_kernel_init, name='tcn-dec')(dec_upsample)

        # Put the filter-outputs through a dense layer finally, to get the reconstructed signal
        o = Dense(ts_dimension, activation='linear')(dec_reconstructed)

        decoder = Model(inputs=[j], outputs=[o])
        

        code = encoder(i)
        reconstruction = decoder(code)

        autoencoder = Model(i, reconstruction)

        adam = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True)
        
        autoencoder.compile(loss=loss, optimizer=adam, metrics=[loss])
        
        print(encoder.summary())
        print(decoder.summary())
        print(autoencoder.summary())
        
        return encoder, decoder, autoencoder 
   