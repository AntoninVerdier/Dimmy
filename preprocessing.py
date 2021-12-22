import os
import pywt
import pickle as pkl
import librosa
import numpy as np
from rich.progress import track

from tqdm import tqdm

from scipy import signal

from multiprocessing import Pool, Manager

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
			mag = np.log(1 + np.abs(Zxx))
		else:
			mag = np.abs(Zxx)

		#dataset[i, :, :] = np.array(((mag - np.min(mag))/np.max(mag))*255, dtype=np.uint8)
		dataset[i, :, :] = np.array((mag - np.min(mag))/np.max(mag))


	pkl.dump(dataset, open('dataset_train_cnn_log.pkl', 'wb'))

def load_file(file, mod=None):
	sample, samplerate = librosa.load(file, sr=192000)
	#sample = librosa.resample(sample, 192000, 16000)
	#samplerate = 16e3
	f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)


	if mod == 'log':
		mag = np.log(1 + np.abs(Zxx))
	else:
		mag = np.abs(Zxx)

	mag = np.array(((mag - np.min(mag))/np.max(mag))*255, dtype=np.uint8)

	return mag

def load_unique_file(arg, mod=None):
	path, mod = arg
	sample, samplerate = librosa.load(os.path.join(path), sr=192000)
	# sample = librosa.resample(sample, samplerate, 192000)
	# samplerate = 192e3

	f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

	if mod == 'log':
		mag = np.log(1 + np.abs(Zxx))

	else:
		mag = np.abs(Zxx)
		mag = mag * r

	#dataset[i, :, :] = np.array(((mag - np.min(mag))/np.max(mag))*255, dtype=np.uint8)
	return np.array((mag - np.min(mag))/np.max(mag)*255, dtype=np.uint8)

def load_data_array_multi(folder, filename='dataset', mod=None):
	files = os.listdir(folder)
	ids = [f[:-4] for f in files]

	dataset = []

	paths = [(os.path.join(folder, file), mod) for file in files]

	with Pool() as p:
		results = [p.apply_async(load_unique_file, args=(path, )) for path in paths]
		dataset = [p.get() for p in track(results, description='Computing spectrograms ...')]

	dataset = np.array(dataset, dtype=np.uint8)
	pkl.dump(dataset, open(filename, 'wb'))

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
	load_data_array_multi('/home/user/share/gaia/Data/Data_agmentation_CNN_Deepen/AugmentationFinal/Validate/raw_audio', mod='log', filename='validate_cnn_log.pkl')


		
