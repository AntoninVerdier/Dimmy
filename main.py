import os
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
#import keras

# Load nsynth audio data randomly
def load_data(folder):
	files = os.listdir(folder)
	files = np.random.choice(files, 30000)

	sounds = {}

	for i, file in tqdm(enumerate(files), total=len(files)):
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
	for i, k in tqdm(enumerate(sounds), total=len(sounds)):
		f, t, Zxx = signal.stft(sounds[k], fs=samplerate, window='hamming', nperseg=1024, noverlap=512)
		mag = np.abs(Zxx)
		p = np.angle(Zxx)
		magnitudes.append(mag)
		phases.append(p)

	return np.array(magnitudes), np.array(phases)


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

# sounds, samplerate = load_data('/home/pouple/PhD/Code/Dimmy/Data/nsynth-train/audio')
# mag, phases = get_mag_phases(sounds)

# encoder, decoder = build_autoencoder(mag.shape[1:], 20)

# inp = Input(mag.shape[1:])
# code = encoder(inp)
# reconstruction = decoder(code)

# autoencoder = Model(inp, reconstruction)
# autoencoder.compile(optimizer='adam', loss='mse')


# history = autoencoder.fit(mag, mag, epochs=100)
# autoencoder.save('Autoencoder_model')
autoencoder = keras.models.load_model('Autoencoder_model')

sounds_test, samplerate = load_data('/home/pouple/PhD/Code/Dimmy/Data/nsynth-valid/audio')
mag_t, phases_t = get_mag_phases(sounds_test)



decoded_mag = autoencoder.evaluate(mag_t, mag_t)
print(decoded_mag)

Zxx = decoded_mag * np.exp(phases_t*1j)

reconstructed_sounds = []
for i, k in enumerate(Zxx):
	t, sound = signal.istft(k, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)
	wavfile.write('Sounds/Sound_{}'.format(i), samplerate, sound)


# Aitoencoder get prediction (therefore mag values)
# Smush them with phases
# do an ISTFT
# get sound vibin'
