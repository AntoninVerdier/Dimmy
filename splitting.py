import os
import librosa
from rich.progress import track 
import numpy as np 
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

for file in os.listdir('Data/samples_nature'):
	sample, samplerate = librosa.load(open(os.path.join('Data', 'samples_nature', file), 'rb'), sr=96000)
	
	s_duration = sample.shape[0] / samplerate

	# Take 2 seconds samples 
	subsamples = np.array(np.array_split(sample, list(range(0, len(sample), 2*samplerate)))[1:-1])

	for i, s in enumerate(subsamples):
		wavfile.write(os.path.join('Data/samples_nature', file + '_{}.wav'.format(i)), samplerate, s)

	# f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=4096, noverlap=512)

	# mags = np.log(1 + np.abs(Zxx))
	# mags = (mags - np.min(mags))/np.max(mags)



	


