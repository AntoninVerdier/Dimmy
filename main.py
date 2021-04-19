import os 
import librosa
import numpy as np 
from tqdm import tqdm 

from scipy import signal 
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Input, Dense
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



sounds, samplerate = load_data('/home/anverdie/Downloads/nsynth-test/audio/')
ffts = []
for i, k in tqdm(enumerate(sounds)):
	f, t, Zxx = signal.stft(sounds[k], fs=samplerate, window='hamming', nperseg=1024, noverlap=512)
	ffts.append(Zxx)


	# Need to convert to log scale and normalize by frequency
latent_dim = 20

model = Sequential()

model.add(Input(shape=513))
model.add(Dense(513, activation='tanh'))
model.add(Dense(256, activation='tanh'))

model.add(Dense(latent_dim))

model.add(Dense(256, activation='tanh'))
model.add(Dense(513, activation='tanh'))

model.compile(optimizer='adam')

model.fit(ffts, ffts, batch_size=2, shuffle=True)
           