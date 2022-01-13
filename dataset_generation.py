import os
from QuickSound.Sound import Sound

import numpy as np

import os
import QuickSound.Sound as Sound
import numpy as np 
import matplotlib.pyplot as plt

from rich.progress import track

from multiprocessing import Pool, Manager


samplerate = 64000
amplitude = 70
duration = 500
path = 'Raw_sounds_dataset'

# Do we care after several amplitude ?

if not os.path.exists(path):
	os.makedirs(path)

np.random.seed(432754448)
# Pure tone frequencies
pts= np.random.randint(low=500, high=32000, size=3000, dtype=int)

carryam_frequencies = np.random.randint(low=6000, high=16000, size=100, dtype=int)
am_frequencies = np.random.randint(low=10, high=300, size=30, dtype=int)
ams = [(cam, fam) for cam in carryam_frequencies for fam in am_frequencies]


chirps= np.random.randint(low=500, high=20000, size=(3000, 2), dtype=int)

step_ranges = np.random.randint(low=500, high=20000, size=(3000, 2), dtype=int)
step_size = np.random.randint(low=2, high=60, size=(3000, 1), dtype=int)
steps = np.concatenate([step_ranges, step_size], axis=1)

# Generate pure_tones
def gen_pt(f):
	pure = Sound(amplitude=amplitude, samplerate=samplerate)
	pure.pure_tone(f, duration=duration)
	pure.save_wav(name='PT_{}_{}ms_{}dB'.format(f, duration, amplitude), path=path)

# Can generate two sets of independent variable
def gen_am(ams):
	am = Sound(amplitude=amplitude, samplerate=samplerate)
	am.amplitude_modulation(ams[0], ams[1], duration=duration)
	am.save_wav(name='AM_{}_{}_{}ms_{}dB'.format(ams[0], ams[1], duration, amplitude), path=path)

# Chirp can be generated by controlled ints
def gen_chirps(c):
	chirp = Sound(amplitude=amplitude, samplerate=samplerate)
	chirp.freq_modulation(c[0], c[1], duration=duration)
	chirp.save_wav(name='Chirp_{}_{}_{}ms_{}dB'.format(c[0], c[1], duration, amplitude), path=path)

def gen_steps(s):
	step = Sound(amplitude=amplitude, samplerate=samplerate)
	try:
		step.steps(s[0], s[1], s[2], spacing='Log', duration=duration)
		step.save_wav(name='Step_{}_{}_{}_{}ms_{}dB'.format(s[0], s[1], s[2], duration, amplitude), path=path)
	except Exception as e:
		print(e)

def gen_multi(function, args):
	with Pool() as p:
		results = [p.apply_async(function, args=(a, )) for a in args]
		dataset = [p.get() for p in track(results, description='Generating sounds ...')]

if __name__ == '__main__':
	# gen_multi(gen_pt, pts)
	# gen_multi(gen_am, ams)
	gen_multi(gen_chirps, chirps)
	gen_multi(gen_steps, steps)

# Generate sounds for frequency discrimnation




# for f in np.geomspace(0.5e3, 16e3, 16):
# 	pure = Sound(samplerate=samplerate, amplitude=amplitude)
# 	pure.pure_tone(f, duration=duration)
# 	pure.save_wav(name='PT_{0}Hz_{1}ms_{2}dB'.format(int(f), duration, amplitude), path=path)

# # Generate sounds for amplitude modulation
# for a in np.geomspace(20, 200, 16):
# 	pure = Sound(samplerate=samplerate, amplitude=amplitude)
# 	pure.amplitude_modulation(10e3, a, duration=duration)
# 	pure.save_wav(name='AM_{0}Hz_{1}ms_{2}dB'.format(int(a), duration, amplitude), path=path)

# # Generates sounds for amplitude modulation and noise
# for na in [45, 50, 55, 60]:
# 	for a in np.geomspace(20, 200, 6):
# 		pure = Sound(samplerate=samplerate, amplitude=amplitude)
# 		pure.amplitude_modulation(10e3, a, duration=duration)
# 		noise = Sound(samplerate=samplerate, amplitude=na)
# 		noise.noise(duration=duration)
# 		final = pure * noise
# 		final.save_wav(name='AMN_{0}Hz_{1}ms_{2}dB_noise_{3}dB'.format(int(a), duration, amplitude, na), path=path)

# # Generates a chirp
# chirp = Sound(samplerate=samplerate, amplitude=amplitude)
# chirp.freq_modulation(6e3, 16e3, duration=duration)
# chirp.save_wav(name='Chirp_6000Hz_16000Hz_{}ms_70dB'.format(duration), path=path)

# # Generates steps
# for s in [2, 4, 6, 8, 10]:
# 	step = Sound(samplerate=samplerate, amplitude=amplitude)
# 	step.steps(6e3, 16e3, s, duration=duration)
# 	step.save_wav(name='Steps_{}_6000Hz_16000Hz_{}ms_70dB'format(s, duration), path=path)

# # Generate harmonic 
# struc = [(4e3,), (4e3, 20e3), (4e3, 12e3, 20e3), (4e3, 8e3, 12e3, 20e3), (4e3, 8e3, 12e3, 16e3, 20e3)]
# for h in struct:
# 	for f in h:
# 		pure = Sound(samplerate=samplerate, amplitude=amplitude)
# 		pure.pure_tone(f, duration=500)
