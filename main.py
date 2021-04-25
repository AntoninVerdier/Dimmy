import os
import pickle as pkl
import cmath
import keras
import librosa
import numpy as np
from tqdm import tqdm

from scipy import signal
import matplotlib.pyplot as plt

from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Dense, InputLayer, Flatten, Reshape

from data_gen import DataGenerator
#import keras


# Could be a good idea to store compute this as a generator because of the ram it needs.
# Thhis way dataset will not be stored each time

# Load nsynth audio data randomly
def load_data(folder, cap=None):
	files = os.listdir(folder)
	if cap:
		files = np.random.choice(files, cap)

	for j, fs in enumerate(np.array_split(np.array(files), 10)):
		specs = []
		for i, file in enumerate(tqdm(fs)):
			sample, samplerate = librosa.load(os.path.join(folder, file), sr=None)

			f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

			specs.append(Zxx)

		pkl.dump([specs, fs, samplerate], open('ata_{}.pkl'.format(j), 'wb'), )


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

	return encoder, decoder

# specs, files, samplerate = load_data('/home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-train/audio')

partition = {}
params = {'dim': (513,126),
          'batch_size': 1024,
          'shuffle': True}

partition['train'] = os.listdir('/home/user/Documents/Antonin/Code/Dimmy/Data/mags')
partition['validation'] = os.listdir('/home/user/Documents/Antonin/Code/Dimmy/Data/mags')

training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)

# for i in training_generator:
# 	pass

encoder, decoder = build_autoencoder((513, 126), 25)

inp = Input((513, 126))
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(training_generator,
						  validation_data=validation_generator,
						  epochs=1)

autoencoder.save('Autoencoder_model')
pkl.dump(history, open('model_history.pkl', 'wb'))

# autoencoder = keras.models.load_model('Autoencoder_model')

# mag_t, phases_t, samplerate = load_data('home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-valid/audio')
# decoded_mag = autoencoder.predict(mag_t)

# Zxx = decoded_mag * np.exp(phases_t*1j)

# reconstructed_sounds = []
# for i, k in enumerate(Zxx):
# 	t, sound = signal.istft(k, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)
# 	wavfile.write('Sounds/Sound_{}'.format(i), samplerate, sound)

