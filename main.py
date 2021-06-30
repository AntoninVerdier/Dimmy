import os
import pickle as pkl
import numpy as np

from keras.models import load_model, Model

from scipy import signal
import matplotlib.pyplot as plt

from scipy.io import wavfile

from tqdm import tqdm 


import settings as s
from Models import Autoencoder

from data_gen import DataGenerator, DataGenerator_both, get_generators

paths = s.paths()
params = s.params()

# Could be a good idea to store compute this as a generator because of the ram it needs.
# Thhis way dataset will not be stored each time


# training_generator, validation_generator, test_generator = get_generators(**params.gen_params, channels=False, dataset='dataset1.pkl')
X_train = np.load(open('dataset_train.pkl', 'rb'), allow_pickle=True)
np.random.shuffle(X_train)

#training_generator = ((X, X) for X in np.array_split(X_train, 256))

auto = Autoencoder('densebin', (513, 126), params.latent_size)
encoder, decoder, autoencoder = auto.get_model()
print(autoencoder.summary())

history = autoencoder.fit(X_train, X_train,
                          epochs=params.epochs, 
                          batch_size=256)

autoencoder.save('Autoencoder_model')
encoder.save('Encoder_model')
decoder.save('Decoder_model')

pkl.dump(history.history, open('Output/model_history.pkl', 'wb'))

# autoencoder = load_model('Autoencoder_model')
# encoder = load_model('Encoder_model')
# decoder = load_model('Decoder_model')

# history = pkl.load(open('Output/model_history.pkl', 'rb'))
# print(history)
# plt.plot(history['loss'])
# plt.show()
# test_generator, phases = get_generators(**params.test_params, test=True)

# X_test = pkl.load(open('dataset_test.pkl', 'rb'))
# #test_data = np.array([t for t in test_generator]).reshape(len(test_generator), 513, 126)
# #phases = np.array([p for p in phases]).reshape(len(test_data), 513, 126)

# layer_1_encoder = Model(encoder.inputs, encoder.layers[1].output)
# layer_2_encoder = Model(encoder.inputs, encoder.layers[2].output)

# layer_1_decoder = Model(decoder.inputs, decoder.layers[0].output)
# layer_2_decoder = Model(decoder.inputs, decoder.layers[1].output)
# layer_3_decoder = Model(decoder.inputs, decoder.layers[2].output)

# names = os.listdir(paths.path2test)
# names = [t[:-4] for t in names]

# for i, t in enumerate(X_test):
#   x = t.reshape(1, 513, 126)

#   l1_out = layer_1_encoder(x, training=False)
#   l2_out = layer_2_encoder(x, training=False)
#   out = encoder(x, training=False).numpy()

#   l3_out = layer_1_decoder(out, training=False)
#   l4_out = layer_2_decoder(out, training=False)
#   l5_out = layer_3_decoder(out, training=False)

#   fig = plt.figure(figsize=(10, 10), constrained_layout=True)
#   gs = fig.add_gridspec(4, 4)

#   lay1 = fig.add_subplot(gs[0, 0])
#   lay1.imshow(l1_out.numpy().reshape(19, 27))

#   lay2 = fig.add_subplot(gs[0, 1])
#   lay2.imshow(l2_out.numpy().reshape(16, 16))

#   latent = fig.add_subplot(gs[0, 2])
#   latent.imshow(out.reshape(6, 6))

#   lay3 = fig.add_subplot(gs[1, 0])
#   lay3.imshow(l3_out.numpy().reshape(16, 16))

#   lay4 = fig.add_subplot(gs[1, 1])
#   lay4.imshow(l4_out.numpy().reshape(19, 27))

#   truespec = fig.add_subplot(gs[2, :])
#   truespec.imshow(x.T)

#   reconstruct = fig.add_subplot(gs[3, :])
#   reconstruct.imshow(l5_out.numpy().reshape(513, 126).T)

#   plt.title(names[i])
#   plt.savefig('Output/{}.svg'.format(names[i]))
#   plt.close()

# folder = '/home/user/Documents/Antonin/Code/Dimmy/Data/nsynth-test/audio'
# files = os.listdir(folder)
# names = [f[:-4] for f in files]

# X_test = pkl.load(open('dataset_test.pkl', 'rb'))
# phases = pkl.load(open('dataset_test_phases.pkl', 'rb'))



# #mags = autoencoder(X_test)
# mags = np.exp(X_test/255) -1

# Zxx = mags * np.exp(phases/255*1j)

# reconstructed_sounds = []
# for i, k in tqdm(enumerate(Zxx)):
#   t, sound = signal.istft(k, fs=16000, window='hamming', nperseg=1024, noverlap=512)
#   wavfile.write('Sounds/Sound_36_256_{}'.format(names[i]), 16000, sound)









