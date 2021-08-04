import keras
import numpy as np

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, InputLayer, Flatten, Reshape, Layer, Conv2D
from keras import backend as K

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
		encoder.add(Dense(1024, activation='tanh'))
		encoder.add(Dense(513, activation='tanh'))
		encoder.add(Dense(256, activation='tanh'))
		encoder.add(Dense(self.latent_dim))

		decoder = Sequential()
		decoder.add(InputLayer((self.latent_dim,)))
		decoder.add(Dense(256, activation='tanh'))
		decoder.add(Dense(513, activation='tanh'))
		decoder.add(Dense(1024, activation='tanh'))
		decoder.add(Dense(np.prod(self.input_shape)))
		decoder.add(Reshape(self.input_shape))

		inp = Input(self.input_shape)
		code = encoder(inp)
		reconstruction = decoder(code)

		autoencoder = Model(inp, reconstruction)
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
		autoencoder.compile(optimizer='adam', loss='mse')

		return encoder, decoder, autoencoder


	def __conv_simple(self):

		encoder = Sequential()
		encoder.add(InputLayer((*self.input_shape, 1)))
		encoder.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
		encoder.add(Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu'))
		encoder.add(Flatten())
		encoder.add(Dense(self.latent_dim))

		print(encoder.summary())

		decoder = Sequential()
		decoder.add(InputLayer((self.latent_dim)))
		encoder.add(Dense(self.latent_dim, activation='softmax'))
		encoder.add(Dense(513*126*64))
		encoder.add(Reshape((513, 126, 64)))
		encoder.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
		encoder.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
		decoder.add(Dense(np.prod(self.input_shape)))
		decoder.add(Reshape(self.input_shape))

		inp = Input(self.input_shape)
		code = encoder(inp)
		reconstruction = decoder(code)

		autoencoder = Model(inp, reconstruction)
		autoencoder.compile(optimizer='adam', loss='mse')

		return encoder, decoder, autoencoder


	def __dense_auditory(self):
		pass

	def __conv_vae(self):

		encoder = Sequential()
		encoder.add(InputLayer((*self.input_shape, 1)))
		encoder.add(Conv2D(32, kernel_size=3, strides=(2, 2), activation='relu'))
		encoder.add(Conv2D(64, kernel_size=3, strides=(2, 2), activation='relu'))
		encoder.add(Flatten())
		encoder.add(Dense(self.input_shape + self.input_shape))

		print(encoder.summary())

		decoder = Sequential()
		decoder.add(InputLayer)




