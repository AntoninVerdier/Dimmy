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

def load_raw_file(file):
	return librosa.load(file, sr=64000)[0]

def load_file(file, mod=None):
	sample, samplerate = librosa.load(file, sr=192000)
	sample = librosa.resample(sample, 192000, 64000)
	samplerate = 64000

	f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

	if mod == 'log':
		mag = np.log(1 + np.abs(Zxx))
	else:
		mag = np.abs(Zxx)

	spec = np.array(((mag - np.min(mag))/np.max(mag))*255, dtype=np.uint8)
	spec = spec[:int(len(spec)/2), :]

	return spec

def load_unique_file(arg, mod=None, cropmid=True):
	""" cropmid allwos to keep only 60% of the spectgram since NYQUIST frequency doesn't go akll the way 
	"""
	path = arg
	sample, samplerate = librosa.load(os.path.join(path), sr=64000)
	#sample = librosa.resample(sample, samplerate, 64000)
	#samplerate = 64e3

	f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=512, noverlap=256)

	if mod == 'log':
		mag = np.log(1 + np.abs(Zxx))

	else:
		mag = np.abs(Zxx)
		mag = mag

	spec = np.array((mag - np.min(mag))/np.max(mag)*255, dtype=np.uint8)
	spec = spec[:, :]
	#dataset[i, :, :] = np.array(((mag - np.min(mag))/np.max(mag))*255, dtype=np.uint8)
	return spec

def load_unique_file_cqt(arg, y, mod=None, cropmid=True):
	""" cropmid allwos to keep only 60% of the spectgram since NYQUIST frequency doesn't go akll the way 
	"""
	path = arg
	sample, samplerate = librosa.load(os.path.join(path), sr=64000)

	C = np.abs(librosa.cqt(sample, sr=64000, hop_length=256, fmin=500, n_bins=128, bins_per_octave=22, filter_scale=1, res_type='fft'))
	C = np.multiply(C, y)


	spec = np.array((C - np.min(C))/np.max(C)*255, dtype=np.uint8)
	
	return spec

def load_data_array_multi(file_list, filename='dataset', mod=None):
	ids = [os.path.basename(f) for f in file_list]

	x = np.arange(1, 129)
	y = 0.5*np.exp(-0.022*x)
	y = 1/(np.repeat(y, 126).reshape(128, 126))

	dataset = []

	paths = file_list#[(file, mod) for file in file_list]

	with Pool() as p:
		results = [p.apply_async(load_unique_file_cqt, args=(path, y)) for path in paths]
		dataset = [p.get() for p in track(results, description='Computing spectrograms ...')]

	dataset = np.array(dataset)
	pkl.dump(dataset, open(filename, 'wb'))

def correlation_matrix(projections):
	""" Projections must be of the form : (n_proj, x, y)"""
	correlation_matrix = np.corrcoef(np.array([np.matrix.flatten(p) for p in projections]))

	return correlation_matrix

def euclidian_distance(arr, brr):
	return np.linalg.norm(arr.flatten()-brr.flatten()) # Should reeturn Euclidian distance between matrices

def corrleation(arr, brr):
	return np.correlatee(arr.flatten(), brr.flatten)

def cosine_distance(arr, brr):
	return cosine(arr.flatten(), brr.flatten())

if __name__ == '__main__':

	import natsort as n

	pc = '/home/user/Documents/Antonin/Dimmy/Clean_sounds_datasetv2_60_28k'
	pn = '/home/user/Documents/Antonin/Dimmy/Noise_sounds_datasetv2_60_28k'
	toep = '/home/user/Documents/Antonin/Dimmy/toeplitz/toeplitz'


	files_noise = np.array([f for f in os.listdir(pn)])
	files_clean = np.array([f for f in os.listdir(pc)])

	files = np.intersect1d(files_noise, files_clean)
	paths_clean = [os.path.join(pc, f) for f in n.natsorted(files)]
	paths_noise = [os.path.join(pn, f) for f in n.natsorted(files)]

	paths = [os.path.join(toep, f) for f in os.listdir(toep)]

	
	load_data_array_multi(paths, mod='log', filename='/home/user/Documents/Antonin/Dimmy/toeplitz/toeplitz_offset_cqt_128_28k.pkl')
	load_data_array_multi(paths_clean, mod='log', filename='heardat_clean_datasetv2_60_cqt_128_28k.pkl')
	load_data_array_multi(paths_noise, mod='log', filename='heardat_noise_datasetv2_60_cqt_128_28k.pkl')
# 
	# sample, samplerate = librosa.load('/home/user/Documents/Antonin/Dimmy/Clean_sounds_datasetv2_60/PT_2047_500ms_70dB_noise6.wav', sr=64000)
	# f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=512, noverlap=2)
	# print(f)
	# print(np.geomspace(500, 32e3, 32))

	# plt.imshow(np.abs(Zxx))
	# plt.show()
	# import pywt

	# c = librosa.cqt(sample, sr=64000, n_bins=50, hop_length=256, fmin=100)
	# plt.matshow(np.abs(c)) 
	# plt.show()




		
