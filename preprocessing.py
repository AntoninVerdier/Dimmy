import os
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

	dataset = np.empty((len(ids), 513, 126), dtype=np.int16)

	for i, file in enumerate(tqdm(ids)):
		sample, samplerate = librosa.load(os.path.join(folder, file + '.wav'), sr=16000)


		f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)


		#normag = np.floor((np.abs(Zxx) - np.min(np.abs(Zxx)))/np.max(np.abs(Zxx))*(2**16))
		mag = np.abs(Zxx)
		dataset[i, :, :] = mag
		# maglog = np.log(mag)
		# plt.imshow(maglog)
		# plt.show()
		# plt.close()
		# plt.imshow(mag)
		# plt.show()
		#dataset[i, 1, :, :] = np.angle(Zxx)

	dmax = np.max(dataset)
	mmax = np.min(dataset)

	for i, row in enumerate(dataset):
		dataset[i] = (dataset[i] - np.min(dataset))/np.max(dataset)

	np.save(open('dataset1.npy', 'wb'), dataset)
	print('ok')


		


		# np.save(open('Data/mags-test/{}.npy'.format(file), 'wb'), mag)
		# np.save(open('Data/phases-test/{}.npy'.format(file), 'wb'), phase)

# load_data('/home/pouple/PhD/Code/Dimmy/Data/nsynth-train/audio')


def load_data_multi(file):
	sample, samplerate = librosa.load(os.path.join(folder, file + '.wav'), sr=16000)

	f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

	mag = np.abs(Zxx)
	phase = np.angle(Zxx)

	np.save(open('Data/mags/{}.npy'.format(file), 'wb'), mag)
	np.save(open('Data/phases/{}.npy'.format(file), 'wb'), phase)

#load_data('/home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-train/audio')

# if __name__ == '__main__':
# 	folder = '/home/user/Documents/Antonin/Code/Dimmy/Data/DeepenSounds/AutoencoderSounds_Longer/WavFilesLong'
# 	files = os.listdir(folder)
# 	ids = [f[:-4] for f in files]
# 	pool = Pool(processes=24)
# 	pool.map(load_data_multi, ids)

		
