import os
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



sounds, samplerate = load_data('/home/anverdie/Downloads/nsynth-valid/audio/')

phases = []
magnitudes = []
for i, k in tqdm(enumerate(sounds)):
	f, t, Zxx = signal.stft(sounds[k], fs=samplerate, window='hamming', nperseg=1024, noverlap=512)
	mag = np.abs(Zxx)
	p = np.angle(Zxx)
	magnitudes.append(mag)
	phases.append(p)

latent_dim = 20
inp_shape = ffts.shape[1:]
print(inp_shape)
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

encoder, decoder = build_autoencoder(inp_shape, 20)

inp = Input(inp_shape)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)
autoencoder.compile(optimizer='adam', loss='mse')

print(autoencoder.summary())

history = autoencoder.fit(ffts, ffts, epochs=30)
