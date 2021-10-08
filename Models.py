import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import tensorflow.keras
import numpy as np


from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, InputLayer, Flatten, Reshape, Layer, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras import backend as K

# from AE import Sampling
# from AE import VAE


from tensorflow import keras
from tensorflow.keras import layers
# class BinningLayer(Layer):
# 	self.output_dim = output_dim
# 	super(BinningLayer, self).__init__(**kwargs)

# 	def build(self, input_shape):
# 		self.kernel = self.add_weight(name='binning',
# 			shape=(input_shape[1], self.output_dim),
# 			initializer='normal', trainable=False)
# 		super(BinningLayer, self).build(input_shape)
	
# 	def call(self, input_data):
# 		return K.dot(input_data, self.kernel)

# 	def compute_output_shape(self, input_shape): 
# 		return (input_shape[0], self.output_dim)




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
		elif self.model == 'densebin':
			return self.__dense_with_constraints()
		elif self.model == 'conv_simple':
			return self.__conv_simple()
		elif self.model == 'conv_vae':
			return self.__conv_vae()
	
	def get_data(self):
		if self.dataset_type == 'log':
			self.X_train = np.load(open('Data/dataset_train_log.pkl', 'rb'), allow_pickle=True)
		if self.dataset_type == 'linear':
			self.X_train = np.load(open('Data/dataset_train_linear.pkl', 'rb'), allow_pickle=True)

		print('Data loaded.')

	def __dense(self):

		encoder = Sequential()
		encoder.add(InputLayer(self.input_shape))
		encoder.add(Flatten())
		encoder.add(Dense(512, activation='relu'))
		encoder.add(Dense(256, activation='relu'))
		encoder.add(Dense(128, activation='relu'))
		encoder.add(Dense(self.latent_dim, name='latent_dim'))

		decoder = Sequential()
		decoder.add(InputLayer((self.latent_dim,)))
		decoder.add(Dense(128, activation='relu'))
		decoder.add(Dense(256, activation='relu'))
		decoder.add(Dense(512, activation='relu'))
		decoder.add(Dense(np.prod(self.input_shape)))
		decoder.add(Reshape(self.input_shape))

		inp = Input(self.input_shape)
		code = encoder(inp)
		reconstruction = decoder(code)

		autoencoder = Model(inp, reconstruction, name='dense')
		autoencoder.compile(optimizer='adam', loss='mse')
		return encoder, decoder, autoencoder

	def __dense_with_constraints(self):

		encoder = Sequential()
		encoder.add(InputLayer(self.input_shape))
		encoder.add(Flatten())
		encoder.add(Dense(513, activation='tanh'))
		encoder.add(Dense(256, activation='tanh'))
		encoder.add(Dense(self.latent_dim))
		encoder.add(keras.layers.experimental.preprocessing.Discretization(num_bins=10))

		decoder = Sequential()
		decoder.add(InputLayer((self.latent_dim,)))
		decoder.add(Dense(256, activation='tanh'))
		decoder.add(Dense(513, activation='tanh'))
		decoder.add(Dense(np.prod(self.input_shape)))
		decoder.add(Reshape(self.input_shape))

		inp = Input(self.input_shape)
		code = encoder(inp)
		reconstruction = decoder(code)

		autoencoder = Model(inp, reconstruction)
		autoencoder.compile(optimizer='sgd', loss='mse')

		return encoder, decoder, autoencoder


	def __conv_simple(self):

		input_img = Input(shape=(*self.input_shape, 1))

		#encoder.add(InputLayer((*self.input_shape, 1)))
		encoder = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(input_img)
		encoder = MaxPooling2D((2, 2), padding="same")(encoder)
		encoder = Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(encoder)
		encoder = MaxPooling2D((2, 2), padding="same")(encoder)

		decoder = Conv2DTranspose(32, (5, 5), strides=1, activation="relu", padding="same")(encoder)
		decoder = UpSampling2D((2, 2))(decoder)
		decoder = Conv2DTranspose(64, (3, 3), strides=1, activation="relu", padding="same")(decoder)
		decoder = UpSampling2D((2, 2))(decoder)
		decoder = Conv2D(1, (1, 1), activation="sigmoid", padding="same")(decoder)


		autoencoder = Model(input_img, decoder)
		autoencoder.compile(optimizer='adam', loss='mse')

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

