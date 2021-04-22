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
#import keras

# Load nsynth audio data randomly
def load_data(folder, cap=None):
	magnitudes, phases = [], []
	files = os.listdir(folder)
	if cap:
		files = np.random.choice(files, cap)

	for i, file in tqdm(enumerate(files), total=len(files)):
		sample, samplerate = librosa.load(os.path.join(folder, file), sr=None)

		f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

		magnitudes.append(np.abs(Zxx))
		phases.append(np.angle(Zxx))

	return np.array(magnitudes), np.array(phases), samplerate


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

mag, phases, samplerate = load_data('/home/pouple/PhD/Code/Dimmy/Data/nsynth-train/audio', cap=30000)

encoder, decoder = build_autoencoder(mag.shape[1:], 40)

inp = Input(mag.shape[1:])
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)
autoencoder.compile(optimizer='adam', loss='mse')


history = autoencoder.fit(mag, mag, epochs=100)
autoencoder.save('Autoencoder_model')

pickle.dump(open('model_history.pkl', 'w'), history)

autoencoder = keras.models.load_model('Autoencoder_model')

mag_t, phases_t, samplerate = load_data('/home/pouple/PhD/Code/Dimmy/Data/nsynth-valid/audio')
decoded_mag = autoencoder.predict(mag_t)

Zxx = decoded_mag * np.exp(phases_t*1j)

reconstructed_sounds = []
for i, k in enumerate(Zxx):
	t, sound = signal.istft(k, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)
	wavfile.write('Sounds/Sound_{}'.format(i), samplerate, sound)


# Aitoencoder get prediction (therefore mag values)
# Smush them with phases
# do an ISTFT
# get sound vibin'
