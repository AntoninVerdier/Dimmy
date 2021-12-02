import os
import pywt
import pickle as pkl
import librosa
import numpy as np

from tqdm import tqdm

from scipy import signal

from multiprocessing import Pool

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from scipy.io import wavfile
from skimage.transform import resize
from scipy.spatial.distance import cosine

# Goal is to clarify this file and make a large function that can generate the dataset with the preprocessing steps required (mel,log-scaled, etc.)


def load_data(folder, cap=None):
	files = os.listdir(folder)

	ids = [f[:-4] for f in files]

	dataset = {}

	for i, file in enumerate(tqdm(ids)):
		sample, samplerate = librosa.load(os.path.join(folder, file + '.wav'), sr=16000)

		f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

		mag = np.abs(Zxx)
		dataset[file] = mag

	all_max = 0
	for k in dataset:
		curr_max = np.max(dataset[k])
		if curr_max > all_max:
			all_max = curr_max

	for k in dataset:
		dataset[k] = np.array((dataset[k]/all_max)*(2**16), dtype=np.uint16)

	pkl.dump(dataset, open('dataset1.pkl', 'wb'))

def load_file(file):
	sample, samplerate = librosa.load(file, sr=16000)

	f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

	mag = np.log(1 + np.abs(Zxx)).astype(np.float16)
	mag = (mag / np.max(mag))*255
	mag = np.array(mag, dtype=np.uint8)

	return mag

def load_data_array(folder, cap=None):
	files = os.listdir(folder)

	ids = [f[:-4] for f in files]

	dataset = np.empty((len(ids), 513, 189), dtype=np.float16)

	for i, file in enumerate(tqdm(ids)):
		sample, samplerate = librosa.load(os.path.join(folder, file + '.wav'), sr=192000)

		f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)
		print(np.abs(Zxx).shape)

		mag = np.log(1 + np.abs(Zxx))

		plt.imshow(mag)
		plt.show()

		dataset[i, :, :] = mag.astype(np.float16)
		#dataset[i, :, :] = np.angle(Zxx).astype(np.float16)


	print(np.max(dataset), np.min(dataset))
	dataset = np.array(dataset/np.max(dataset), dtype=np.float16)
	dataset = dataset * 255
	dataset = np.array(dataset, dtype=np.uint8)

	print(np.max(dataset), np.min(dataset))

	pkl.dump(dataset, open('dataset_test_pupcages.pkl', 'wb'))


		


#load_data_array('/home/pouple/PhD/Data/splitted')

def test_load_data_array(folder, cap=None):
	files = os.listdir(folder)

	ids = [f[:-4] for f in files]

	dataset = np.empty((len(ids), 513, 126), dtype=np.float64)

	for i, file in enumerate(tqdm(ids)):
		sample, samplerate = librosa.load(os.path.join(folder, file + '.wav'), sr=16000)

		f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

		#mag = np.log(1 + np.abs(Zxx))
		mag = np.abs(Zxx)
		

		#mag = np.exp(mag) - 1

		Zxx = mag * np.angle(Zxx)*1j

		#dataset[i, :, :] = np.angle(Zxx).astype(np.float16)
	  	
		t, sound = signal.istft(Zxx, fs=16000, window='hamming', nperseg=1024, noverlap=512)
		wavfile.write('Sounds_testing/Sound_36_256_{}'.format(ids[i]), 16000, sound)

	

	# print(np.max(dataset), np.min(dataset))
	# dataset = np.array(dataset/np.max(dataset), dtype=np.float16)
	# dataset = dataset * 255
	# dataset = np.array(dataset, dtype=np.uint8)

	# print(np.max(dataset), np.min(dataset))

	# pkl.dump(dataset, open('dataset_test.pkl', 'wb'))


def load_data_multi(file):

	sample, samplerate = librosa.load(os.path.join(folder, file + '.wav'), sr=16000)

	f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

	mag = np.log(1 + np.abs(Zxx))

	dataset[i, :, :] = mag


def load_data_to_wl(folder):
	files = os.listdir(folder)

	ids = [f[:-4] for f in files]

	dataset = np.empty((len(ids), 513, 126), dtype=np.float64)

	for i, file in enumerate(tqdm(ids)):
		sample, samplerate = librosa.load(os.path.join(folder, file + '.wav'), sr=16000)
		coef, freqs=pywt.cwt(sample ,np.arange(1,129),'mexh')
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.matshow(coef) # doctest: +SKIP
		plt.axis('auto')
		plt.show()

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



def convert_to_dlp(projection):
	""" DLP has a resolution of 480 by 300px"""
	projection = projection.reshape(10, 10)
	fig, ax = plt.subplots(1, 2)
	ax[0].imshow(projection)


	projection = resize(projection, (300, 300), anti_aliasing=False)
	projection = np.array([90*[0] + list(p) + 90*[0] for p in projection]).reshape(300, 480)
	ax[1].imshow(projection, )
	plt.show()
	plt.close()



#test_load_data_array('/home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-test/audio')
#load_data_to_wl('/home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-test/audio')

# if __name__ == '__main__':
# 	folder = '/home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-train/audio'
# 	files = os.listdir(folder)
# 	ids = [f[:-4] for f in files]
# 	dataset = np.empty((len(ids), 513, 126), dtype=np.float16)
# 	pool = Pool(processes=24)
# 	pool.map(load_data_multi, ids)

		
