import os
import librosa
import numpy as np

from tqdm import tqdm

from scipy import signal



def load_data(folder, cap=None):
	files = os.listdir(folder)

	ids = [f[:-4] for f in files]

	for i, file in enumerate(tqdm(ids)):
		sample, samplerate = librosa.load(os.path.join(folder, file + '.wav'), sr=None)

		f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

		mag = np.abs(Zxx)
		phase = np.angle(Zxx)

		np.save(open('Data/mags/{}.npy'.format(file), 'wb'), mag)
		np.save(open('Data/phases/{}.npy'.format(file), 'wb'), phase)

load_data('/home/pouple/PhD/Code/Dimmy/Data/nsynth-test/audio')
