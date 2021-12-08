import os
import pywt
import pickle as pkl
import librosa
import numpy as np
from rich.progress import track

from tqdm import tqdm

from scipy import signal

from multiprocessing import Pool

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from scipy.io import wavfile
from skimage.transform import resize
from scipy.spatial.distance import cosine

# Goal is to clarify this file and make a large function that can generate the dataset with the preprocessing steps required (mel,log-scaled, etc.)

def load_data_array(folder, mod=None):
	files = os.listdir(folder)

	ids = [f[:-4] for f in files]

	dataset = np.empty((len(ids), 513, 189), dtype=np.float16)

	for i, file in enumerate(track(ids)):
		sample, samplerate = librosa.load(os.path.join(folder, file + '.wav'), sr=400000)
		sample = librosa.resample(sample, samplerate, 192000)
		samplerate = 192e3

		f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

		if mod == 'log':
			mag = log(1 + np.abs(Zxx))
		else:
			mag = np.abs(Zxx)

		dataset[i, :, :] = np.array(((mag - np.min(mag))/np.max(mag))*255, dtype=np.uint8)


	pkl.dump(dataset, open('dataset_train_cnn_log.pkl', 'wb'))

def load_file(file, mod=None):
	sample, samplerate = librosa.load(file, sr=192000)
	sample = librosa.resample(sample, 192000, 16000)
	samplerate = 16e3

	f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)
	
	if mod == 'log':
		mag = log(1 + np.abs(Zxx))
	else:
		mag = np.abs(Zxx)

	mag = np.array(((mag - np.min(mag))/np.max(mag))*255, dtype=np.uint8)

	return mag

def load_data_multi(file):

	sample, samplerate = librosa.load(os.path.join(folder, file + '.wav'), sr=16000)

	f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

	mag = np.log(1 + np.abs(Zxx))

	dataset[i, :, :] = mag

def correlation_matrix(projections):
	""" Projections must be of the form : (n_proj, x, y)"""
	correlation_matrix = np.corrcoef(np.array([np.matrix.flatten(p) for p in projections]))

	return correlation_matrix

def eucidian_distance(arr, brr):
	return np.linalg.norm(arr.flatten()-brr.flatten()) # Should reeturn Euclidian distance between matrices

def corrleation(arr, brr):
	return np.correlatee(arr.flatten(), brr.flatten)

def cosine_distance(arr, brr):
	return cosine(arr.flatten(), brr.flatten())

if __name__ == '__main__':
	load_data_array('/home/anverdie/Documents/Code/Dimmy/Data/audio_wav', mod='log')


# if __name__ == '__main__':
# 	folder = '/home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-train/audio'
# 	files = os.listdir(folder)
# 	ids = [f[:-4] for f in files]
# 	dataset = np.empty((len(ids), 513, 126), dtype=np.float16)
# 	pool = Pool(processes=24)
# 	pool.map(load_data_multi, ids)

		
