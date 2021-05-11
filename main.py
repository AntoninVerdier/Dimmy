import os
import pickle as pkl
import cmath
import keras
import librosa
import numpy as np
from tqdm import tqdm

from scipy import signal
import matplotlib.pyplot as plt

from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, InputLayer, Flatten, Reshape
from keras import backend as K

from data_gen import DataGenerator

# Could be a good idea to store compute this as a generator because of the ram it needs.
# Thhis way dataset will not be stored each time

# Load nsynth audio data randomly
def load_data(folder, cap=None):
	files = os.listdir(folder)
	if cap:
		files = np.random.choice(files, cap)

	for j, fs in enumerate(np.array_split(np.array(files), 10)):
		specs = []
		for i, file in enumerate(tqdm(fs)):
			sample, samplerate = librosa.load(os.path.join(folder, file), sr=None)

			f, t, Zxx = signal.stft(sample, fs=samplerate, window='hamming', nperseg=1024, noverlap=512)

			specs.append(Zxx)

		pkl.dump([specs, fs, samplerate], open('ata_{}.pkl'.format(j), 'wb'), )


def build_autoencoder(sound_shape, latent_dim):

	encoder = Sequential()
	encoder.add(InputLayer(sound_shape))
	encoder.add(Flatten())
	encoder.add(Dense(513, activation='tanh'))
	encoder.add(Dense(256, activation='tanh'))
	encoder.add(Dense(latent_dim))

	decoder = Sequential()
	decoder.add(InputLayer((latent_dim,)))
	decoder.add(Dense(256, activation='tanh'))
	decoder.add(Dense(513, activation='tanh'))
	decoder.add(Dense(np.prod(sound_shape)))
	decoder.add(Reshape(sound_shape))

	return encoder, decoder


def reconstruct_sound(latent):
	decoded_mag = decoder.predict(latent)
	k = decoded_mag * np.exp(0j)
	t, sound = signal.istft(k, fs=16000, window='hamming', nperseg=1024, noverlap=512)
	print(sound)
	return sound

# specs, files, samplerate = load_data('/home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-train/audio')
partition = {}
params = {'dim': (513,126),
          'batch_size': 256,
          'shuffle': False,}

# partition['train'] = os.listdir('/home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-train/audio')
# partition['validation'] = os.listdir('/home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-valid/audio')
partition['test'] = os.listdir('/home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-test/audio')

# training_generator = DataGenerator(partition['train'], **params)
# validation_generator = DataGenerator(partition['validation'], **params)

# pkl.dump(training_generator.indexes, open('Output/train_indexes.pkl', 'wb'))
# pkl.dump(validation_generator.indexes, open('Output/valid_indexes.pkl', 'wb'))


# encoder, decoder = build_autoencoder((513, 126), 25)

# inp = Input((513, 126))
# code = encoder(inp)
# reconstruction = decoder(code)

# autoencoder = Model(inp, reconstruction)
# autoencoder.compile(optimizer='adam', loss='mse')

# history = autoencoder.fit(training_generator,
# 						  validation_data=validation_generator,
# 						  epochs=120)

# autoencoder.save('Autoencoder_model')
# encoder.save('Encoder_model')
# decoder.save('Decoder_model')

# pkl.dump(history.history, open('Output/model_history.pkl', 'wb'))

autoencoder = load_model('Autoencoder_model')
encoder = load_model('Encoder_model')
decoder = load_model('Decoder_model')


# for i in range(25):
# 	test = np.zeros(shape=(1, 25))
# 	test[0, i] = 1
# 	sound = reconstruct_sound(test)
# 	wavfile.write('Sounds_unique/Sound_{}'.format(i), 16000, sound[0])



test_generator = DataGenerator(partition['test'], test=True, **params)
print(test_generator.list_IDs)

prediction = encoder.predict(test_generator)

fig, axs = plt.subplots(10, 10, figsize=(20, 20))

kb_elec = [i for i, f in enumerate(test_generator.list_IDs) if 'keyboard_electronic' in f]

for i, j in enumerate(kb_elec[:100]):
	axs[i//10, i%10].imshow(prediction[j].reshape(5, 5))

plt.show()




history = pkl.load(open('Output/model_history.pkl', 'rb'))

plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.legend()
plt.show()

decoded_mag = autoencoder.predict(test_generator)
phases = DataGenerator(partition['test'], test=True, phase=True, **params)
phases = np.array([p for p in phases]).reshape(4096, 513, 126)
plt.imshow(phases[0])
plt.show()

Zxx = decoded_mag * np.exp(0j)

reconstructed_sounds = []
for i, k in enumerate(Zxx):
	t, sound = signal.istft(k, fs=16000, window='hamming', nperseg=1024, noverlap=512)
	wavfile.write('Sounds/Sound_{}'.format(partition['test'][i]), 16000, sound)


