import os
import pickle as pkl
import keras
import numpy as np

from scipy import signal
import matplotlib.pyplot as plt

from scipy.io import wavfile
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, InputLayer, Flatten, Reshape

import settings as s
from data_gen import DataGenerator

paths = s.paths()
params = s.params()
# Could be a good idea to store compute this as a generator because of the ram it needs.
# Thhis way dataset will not be stored each time

def build_autoencoder(sound_shape, latent_dim):

	encoder = Sequential()
	encoder.add(InputLayer(sound_shape))
	encoder.add(Flatten())
	encoder.add(Dense(513, activation='tanh'))
	encoder.add(Dense(256, activation='tanh'))
	encoder.add(Dense(latent_dim))

	decoder = Sequential()
	decoder.add(InputLayer((latent_dim,)))
	decoder.add(Dense(256, activation='tanh'))
	decoder.add(Dense(513, activation='tanh'))
	decoder.add(Dense(np.prod(sound_shape)))
	decoder.add(Reshape(sound_shape))

	inp = Input(sound_shape)
	code = encoder(inp)
	reconstruction = decoder(code)

	autoencoder = Model(inp, reconstruction)
	autoencoder.compile(optimizer='adam', loss='mse')

	return encoder, decoder, autoencoder

def get_generators(dim, batch_size, shuffle=True):
	print(dim, batch_size, shuffle)
	partition = {}
	partition['train'] = os.listdir(paths.path2train)
	partition['validation'] = os.listdir(paths.path2valid)
	partition['test'] = os.listdir(paths.path2test)

	training_generator = DataGenerator(partition['train'], dim, batch_size, shuffle)
	validation_generator = DataGenerator(partition['validation'], dim, batch_size, shuffle)
	test_generator = DataGenerator(partition['test'], dim, batch_size, shuffle)

	if not os.path.exists('Output/generator/'):
		os.makedirs('Output/generator/')

	pkl.dump(training_generator.indexes, open('Output/generator/train_indexes.pkl', 'wb'))
	pkl.dump(validation_generator.indexes, open('Output/generator/valid_indexes.pkl', 'wb'))
	pkl.dump(test_generator.indexes, open('Output/generator/test_indexes.pkl', 'wb'))


	return training_generator, validation_generator, test_generator

training_generator, validation_generator, test_generator = get_generators(**params.gen_params)
encoder, decoder, autoencoder = build_autoencoder(params.specshape, params.latent_size)

history = autoencoder.fit(validation_generator,
						  validation_data=test_generator,
						  epochs=params.epochs)

autoencoder.save('Autoencoder_model')
encoder.save('Encoder_model')
decoder.save('Decoder_model')

pkl.dump(history.history, open('Output/model_history.pkl', 'wb'))

autoencoder = load_model('Autoencoder_model')
encoder = load_model('Encoder_model')
decoder = load_model('Decoder_model')

params = {'dim': (513,126),
          'batch_size': 1,
          'shuffle': False,}

magnitudes = encoder.predict(test_generator)
prediction = decoder.predict(magnitudes)
phases = np.array([p for p in phases]).reshape(4096, 513, 126)

null = np.zeros(shape=(4096, 513, 126))
Zxx = prediction * np.exp(null*1j)

reconstructed_sounds = []
for i, k in enumerate(Zxx):
	t, sound = signal.istft(k, fs=16000, window='hamming', nperseg=1024, noverlap=512)
	wavfile.write('Sounds/Sound_{}'.format(partition['test'][i]), 16000, sound)


fig, axs = plt.subplots(10, 10, figsize=(20, 20))

for i in range(100):
	axs[i//10, i%10].imshow(prediction[i].reshape(5, 5))

plt.show()







