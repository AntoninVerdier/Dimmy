import tensorflow as tf

import tensorflow.keras
import numpy as np


from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, InputLayer, Flatten, Reshape, Layer, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Discretization, LSTM
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Layer, InputSpec
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers, activations, initializers, constraints, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import UnitNorm, Constraint

import tensorflow.experimental.numpy as tnp

# from AE import Sampling
# from AE import VAE


from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


# Could be useful to implement talos library for gidsearch to run on the weekend

class DenseTied(Layer):
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
                 tied_to=None,
                 **kwargs):
        self.tied_to = tied_to
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
                
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = K.transpose(self.tied_to.kernel)
            self._non_trainable_weights.append(self.kernel)
        else:
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
        return output

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
                 max_n=None,
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
        threshold_for_each_batch = sorted_output[:, -self.max_n]
        filter_bool = tnp.transpose(tnp.greater(tnp.transpose(output), threshold_for_each_batch))
        output = tnp.multiply(output, filter_bool)

        return output

    def get_config(self):
        base_config = super(DenseMax, self).get_config()
        base_config['max_n'] = self.max_n
 

        return base_config

class Autoencoder():
    # This class should return the required autoencoder architecture
    def __init__(self, model, input_shape, latent_dim, dataset_type='log'):
        self.model = model
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.dataset_type = dataset_type

    def get_model(self):
        if self.model == 'dense':
            return self.__dense()
        elif self.model == 'dense_tied':
            return self.__dense_tied()
        elif self.model == 'densebin':
            return self.__dense_with_constraints()
        elif self.model == 'conv_simple':
            return self.__conv_simple()
        elif self.model == 'conv_vae':
            return self.__conv_vae()
        elif self.model == 'conv_simple_test':
            return self.__conv_simple_test()
    
    def get_data(self):
        if self.dataset_type == 'log':
            self.X_train = np.load(open('Data/dataset_train_log.pkl', 'rb'), allow_pickle=True)
        if self.dataset_type == 'linear':
            self.X_train = np.load(open('Data/dataset_train_linear.pkl', 'rb'), allow_pickle=True)

        print('Data loaded.')

    def __dense(self, max_n=10):

        encoder = Sequential()
        encoder.add(InputLayer(self.input_shape))
        encoder.add(Flatten())
        encoder.add(Dense(1024, activation='relu'))
        encoder.add(Dense(512, activation='relu'))
        encoder.add(Dense(256, activation='relu'))
        encoder.add(Dense(128, activation='relu'))
        encoder.add(Dense(128, activation='relu'))
        encoder.add(DenseMax(self.latent_dim, name='latent_dim', max_n=max_n, kernel_constraint=UnitNorm()))

        decoder = Sequential()
        decoder.add(InputLayer((self.latent_dim,)))
        decoder.add(Dense(128, activation='relu'))
        decoder.add(Dense(128, activation='relu'))
        decoder.add(Dense(256, activation='relu'))
        decoder.add(Dense(512, activation='relu'))
        decoder.add(Dense(1024, activation='relu'))

        decoder.add(Dense(np.prod(self.input_shape)))
        decoder.add(Reshape(self.input_shape))

        opt = keras.optimizers.Adam(learning_rate=0.001)


        encoder.compile(optimizer=opt, loss='mse')
        decoder.compile(optimizer=opt, loss='mse')

        inp = Input(self.input_shape)
        code = encoder(inp)
        reconstruction = decoder(code)

        autoencoder = Model(inp, reconstruction, name='dense')
        autoencoder.compile(optimizer=opt, loss='mse')
        return encoder, decoder, autoencoder

    def __dense_tied(self):

        inputs = Input((1, 128))
        
        dense_0 = Dense(512, activation='relu', name='dense_0')
        dense_1 = Dense(256, activation='relu', name='dense_1')
        dense_2 = Dense(128, activation='relu', name='dense_2')
        latent_dim = Dense(self.latent_dim, name='latent_dim')

        x = Flatten()(inputs)
        x = dense_0(x)
        x = dense_1(x)
        output = dense_2(x)
        # output = latent_dim(x)

        encoder = keras.Model(inputs=inputs, outputs=output)

        print(encoder.summary())
        
        inputs_dec = Input((self.latent_dim,))
        dense_dec_0 = DenseTied(128, activation='relu', tied_to=dense_2)(inputs_dec)
        dense_dec_1 = DenseTied(256, activation='relu', tied_to=dense_1)(dense_dec_0)
        dense_dec_2 = DenseTied(512, activation='relu', tied_to=dense_0)(dense_dec_1)
        dense_format = Dense(np.prod(self.input_shape))(dense_dec_2)
        reconstruction = Reshape(self.input_shape)(dense_format)

        decoder = keras.Model(inputs=inputs_dec, outputs=reconstruction)

        autoencoder = Model(inputs, reconstruction, name='dense')
        autoencoder.compile(optimizer='adam', loss='mse')
        return encoder, decoder, autoencoder


    def __conv_simple(self):

        opt = keras.optimizers.Adam(learning_rate=0.001)

        encoder = Sequential()
        encoder.add(InputLayer((*self.input_shape, 1)))

        encoder.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        encoder.add(MaxPooling2D((2, 2), padding="same"))
        encoder.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        encoder.add(MaxPooling2D((2, 2), padding="same"))
        encoder.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
        encoder.add(MaxPooling2D((2, 2), padding="same"))
        encoder.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
        encoder.add(Flatten())
        encoder.add(DenseMax(self.latent_dim, max_n=100, kernel_constraint=UnitNorm()))

        encoder.compile(optimizer=opt, loss='mse')


        for l in encoder.layers :
            print(l.output_shape)


        decoder = Sequential()
        decoder.add(InputLayer((100)))
        #decoder.add(Discretization(num_bins=10, epsilon=0.01)) # Need to check if binning is good, i.e what is the range of input data
        decoder.add(Dense(64*70*16))
        decoder.add(Reshape((64, 70, 16)))
        decoder.add(Conv2DTranspose(16, (3, 3), strides=1, activation="relu", padding="same"))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2DTranspose(16, (3, 3), strides=1, activation="relu", padding="same"))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2DTranspose(32, (3, 3), strides=1, activation="relu", padding="same"))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2DTranspose(64, (3, 3), strides=1, activation="relu", padding="same"))

        decoder.add(Conv2D(1, (1, 1), activation="relu", padding="same"))


        print(encoder.summary())
        decoder.compile(optimizer=opt, loss='mse')

        print(decoder.summary())

        inp = Input((*self.input_shape, 1))
        code = encoder(inp)
        reconstruction = decoder(code)

        autoencoder = Model(inp, reconstruction, name='dense')
        autoencoder.compile(optimizer=opt, loss='mse')
        return encoder, decoder, autoencoder


    def __dense_auditory(self):
        pass

    def __conv_vae(self):
        latent_dim = 2

        encoder_inputs = Input(shape=(*self.input_shape, 1))
        x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Flatten()(x)
        x = Dense(16, activation="relu")(x)
        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        latent_inputs = keras.Input(shape=(latent_dim,))
        x = Dense(7 * 7 * 64, activation="relu")(latent_inputs)
        x = Reshape((7, 7, 64))(x)
        x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()


        # mnist_digits = np.concatenate([x_train, x_test], axis=0)

        mnist_digits = x_train
        mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
        vae = VAE(encoder, decoder)
        vae.compile(optimizer=keras.optimizers.Adam())
        print(mnist_digits.shape)
        vae.fit(mnist_digits, epochs=30, batch_size=128)


    def __conv_simple_test(self):

        opt = keras.optimizers.Adam(learning_rate=0.001)

        encoder = Sequential()
        encoder.add(InputLayer((*self.input_shape, 1)))

        encoder.add(Conv2D(64, kernel_size=(7, 7), padding='same', activation='relu'))
        encoder.add(MaxPooling2D((2, 2), padding="same"))
        encoder.add(Conv2D(32, kernel_size=(7, 7), padding='same', activation='relu'))
        encoder.add(MaxPooling2D((2, 2), padding="same"))
        encoder.add(Conv2D(16, kernel_size=(7, 7), padding='same', activation='relu'))
        encoder.add(MaxPooling2D((2, 2), padding="same"))
        encoder.add(Conv2D(16, kernel_size=(7, 7), padding='same', activation='relu'))
        encoder.add(Flatten())
        encoder.add(DenseMax(self.latent_dim, max_n=100, kernel_constraint=UnitNorm()))

        encoder.compile(optimizer=opt, loss='mse')


        for l in encoder.layers :
            print(l.output_shape)


        decoder = Sequential()
        decoder.add(InputLayer((100)))
        #decoder.add(Discretization(num_bins=10, epsilon=0.01)) # Need to check if binning is good, i.e what is the range of input data
        decoder.add(Dense(64*70*16))
        decoder.add(Reshape((64, 70, 16)))
        decoder.add(Conv2DTranspose(16, (7, 7), strides=1, activation="relu", padding="same"))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2DTranspose(16, (7, 7), strides=1, activation="relu", padding="same"))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2DTranspose(32, (7, 7), strides=1, activation="relu", padding="same"))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2DTranspose(64, (7, 7), strides=1, activation="relu", padding="same"))

        decoder.add(Conv2D(1, (1, 1), activation="relu", padding="same"))


        print(encoder.summary())
        decoder.compile(optimizer=opt, loss='mse')

        print(decoder.summary())

        inp = Input((*self.input_shape, 1))
        code = encoder(inp)
        reconstruction = decoder(code)

        autoencoder = Model(inp, reconstruction, name='dense')
        autoencoder.compile(optimizer=opt, loss='mse')
        return encoder, decoder, autoencoder