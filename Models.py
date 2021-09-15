import tensorflow.keras
import numpy as np

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, InputLayer, Flatten, Reshape, Layer, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras import backend as K

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
		encoder.add(Dense(1024, activation='tanh'))
		encoder.add(Dense(513, activation='tanh'))
		encoder.add(Dense(256, activation='tanh'))
		encoder.add(Dense(self.latent_dim, name='latent_dim'))

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

		autoencoder = Model(inp, reconstruction, name='conv_simple')
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

		print(autoencoder.summary())

		return encoder, decoder, autoencoder


	def __dense_auditory(self):
		pass

	# def __conv_vae(self):
	# 	print(self.input_shape)

	# 	encoder = Sequential()
	# 	encoder.add(InputLayer((*self.input_shape, 1)))
	# 	encoder.add(Conv2D(32, kernel_size=3, strides=(2, 2), activation='relu'))
	# 	encoder.add(Conv2D(64, kernel_size=3, strides=(2, 2), activation='relu'))
	# 	encoder.add(Flatten())
	# 	encoder.add(Dense(self.latent_dim + self.latent_dim))

	# 	print(encoder.summary())

	# 	decoder = Sequential()
	# 	decoder.add(InputLayer(input_shape=(self.latent_dim,)))
	# 	decoder.add(Dense(units=7*7*32, activation='relu'))
	# 	decoder.add(Reshape(target_shape=(7, 7, 32)))
	# 	decoder.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'))
	# 	decoder.add(Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu'))
	# 	decoder.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))


	# 	@tf.function
	# 	def sample(self, eps=None):
	# 		if eps is None:
	# 			eps = tf.random.normal(shape=(100, self.latent_dim))
	# 			return self.decode(eps, apply_sigmoid=True)

	# 	def encode(self, x):
	# 		mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
	# 		return mean, logvar

	# 	def reparameterize(self, mean, logvar):
	# 		eps = tf.random.normal(shape=mean.shape)
	# 		return eps * tf.exp(logvar * .5) + mean

	# 	def decode(self, z, apply_sigmoid=False):
	# 		logits = self.decoder(z)
	# 		if apply_sigmoid:
	# 			probs = tf.sigmoid(logits)
	# 			return probs
	# 		return logits

	# 	optimizer = tf.keras.optimizers.Adam(1e-4)


	# 	def log_normal_pdf(sample, mean, logvar, raxis=1):
	# 	  log2pi = tf.math.log(2. * np.pi)
	# 	  return tf.reduce_sum(
	# 	      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
	# 	      axis=raxis)


	# 	def compute_loss(model, x):
	# 	  mean, logvar = model.encode(x)
	# 	  z = model.reparameterize(mean, logvar)
	# 	  x_logit = model.decode(z)
	# 	  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
	# 	  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
	# 	  logpz = log_normal_pdf(z, 0., 0.)
	# 	  logqz_x = log_normal_pdf(z, mean, logvar)
	# 	  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


	# 	@tf.function
	# 	def train_step(model, x, optimizer):
	# 	  """Executes one training step and returns the loss.

	# 	  This function computes the loss and gradients, and uses the latter to
	# 	  update the model's parameters.
	# 	  """
	# 	  with tf.GradientTape() as tape:
	# 	    loss = compute_loss(model, x)
	# 	  gradients = tape.gradient(loss, model.trainable_variables)
	# 	  optimizer.apply_gradients(zip(gradients, model.trainable_variables))


