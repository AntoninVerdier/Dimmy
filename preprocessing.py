import os
import pickle as pkl
import librosa
import numpy as np

from tqdm import tqdm

from scipy import signal

from multiprocessing import Pool

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt



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

def load_data_array(folder, cap=None):
	files = os.listdir(folder)

	ids = [f[:-4] for f in files]

	dataset = np.empty((len(ids), 513, 126), dtype=np.float16)

	for i, file in enumerate(tqdm(ids)):
		sample, samplerate = librosa.load(os.path.join(folder, file + '.wav'), sr=16000)

		f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

		mag = np.log(1 + np.abs(Zxx))

		dataset[i, :, :] = mag

	dataset = dataset/np.max(dataset)

	pkl.dump(dataset, open('dataset_train.pkl', 'wb'))


################" TEHRE IS A PB WITH THSI PREPRO"
		


		# np.save(open('Data/mags-test/{}.npy'.format(file), 'wb'), mag)
		# np.save(open('Data/phases-test/{}.npy'.format(file), 'wb'), phase)

# load_data('/home/pouple/PhD/Code/Dimmy/Data/nsynth-train/audio')


def load_data_multi(file):

	sample, samplerate = librosa.load(os.path.join(folder, file + '.wav'), sr=16000)

	f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

	mag = np.log(1 + np.abs(Zxx))

	dataset[i, :, :] = mag


load_data_array('/home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-train/audio')

# if __name__ == '__main__':
# 	folder = '/home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-train/audio'
# 	files = os.listdir(folder)
# 	ids = [f[:-4] for f in files]
# 	dataset = np.empty((len(ids), 513, 126), dtype=np.float16)
# 	pool = Pool(processes=24)
# 	pool.map(load_data_multi, ids)

		
