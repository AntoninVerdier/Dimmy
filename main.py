import os
import cmath
import librosa
import numpy as np
from tqdm import tqdm

from scipy import signal
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Input, Dense, InputLayer, Flatten, Reshape
#import keras

# Load nsynth audio data randomly
def load_data(folder):
	files = os.listdir(folder)
	selected_files = np.random.choice(files, 10000)

	sounds = {}

	for i, file in tqdm(enumerate(files)):
		sample, samplerate = librosa.load(os.path.join(folder, file),
								  sr=None, mono=True, offset=0.0, duration=None)
		sounds['{}'.format(file[:-4])] = sample

	return sounds, samplerate
# Perform STFT
# Actual shallow AE, LSTM-AE,

# Testing and extraction of latent dimension


def get_mag_phases(sounds):
	""" Could be better to use map function and apply the function along the list
	"""
	magnitudes, phases = [], []
	for i, k in enumerate(sounds):
		f, t, Zxx = signal.stft(sounds[k], fs=samplerate, window='hamming', nperseg=1024, noverlap=512)
		mag = np.abs(Zxx)
		p = np.angle(Zxx)
		magnitudes.append(mag)
		phases.append(p)

	return np.array(magnitudes), np.array(phases)




sounds, samplerate = load_data('/home/anverdie/Downloads/nsynth-test/audio/')

mag, phases = get_mag_phases(sounds)

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

encoder, decoder = build_autoencoder(mag.shape[1:], 20)

inp = Input(mag.shape[1:])
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)
autoencoder.compile(optimizer='adam', loss='mse')

print(autoencoder.summary())

history = autoencoder.fit(mag, mag, epochs=1)
decoded_mag = autoencoder.predict(mag)

Zxx = decoded_mag * np.exp(phases*1j)

reconstructed_sounds = []
for i, k in enumerate(Zxx):
	sound = signal.istft(k, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)
	reconstructed_sounds.append(sound)


# Aitoencoder get prediction (therefore mag values)
# Smush them with phases 
# do an ISTFT 
# get sound vibin'
