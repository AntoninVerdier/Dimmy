import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import natsort as n

import argparse
import pickle as pkl
import numpy as np

import tensorflow.keras as keras

from tensorflow.keras.models import load_model, Model

from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm 
import settings as s

from Models import Autoencoder

from data_gen import DataGenerator, DataGenerator_both, get_generators
from tensorflow.keras.callbacks import TensorBoard

import preprocessing as proc

# from AE import Sampling
# from AE import VAE

paths = s.paths()
params = s.params()

parser = argparse.ArgumentParser(description='Flags for model training')

parser.add_argument('--train', '-t', action='store_true',
                    help='Train the network')
parser.add_argument('--data_size', '-d', type=int, default=1,
                    help='Percentage of selected training data')
parser.add_argument('--predict', '-p', action='store_true',
                    help='')
parser.add_argument('--network', '-n', type=str,
                    help='Choose network type')
args = parser.parse_args()

# Tensorboard for weight and traingin evaluation
tensorboard = TensorBoard(
  log_dir='./logs',
  histogram_freq=1,
  write_images=True,
  update_freq=5
)

keras_callbacks = [
  tensorboard
]



if args.train:
    X_train = np.load(open('dataset_train.pkl', 'rb'), allow_pickle=True)
    
    # Select the desired portion of the data and shuffle it
    shuffle_mask = np.random.choice(X_train.shape[0], int(args.data_size/100 * X_train.shape[0]), replace=False)
    X_train = X_train[shuffle_mask]

    if args.network: # This to enable fair splitting for convolution
      X_train = X_train[:, :512, :124]
      input_shape = (512, 124)
      print(X_train.shape)

    else:
      input_shape = (513, 124)

    print(input_shape)

    auto = Autoencoder('{net}'.format(net=args.network if args.network else 'dense'), input_shape, params.latent_size)
    encoder, decoder, autoencoder = auto.get_model()

    X_train.reshape(*X_train.shape, 1)
    
    history = autoencoder.fit(X_train, X_train,
                              epochs=params.epochs, 
                              batch_size=8,)
                              #callbacks=keras_callbacks)

    autoencoder.save(os.path.join(paths.path2Models, 'Autoencoder_model_{}'.format(args.network)))
    encoder.save(os.path.join(paths.path2Models, 'Encoder_model_{}'.format(args.network)))
    decoder.save(os.path.join(paths.path2Models, 'Decoder_model_{}'.format(args.network)))

    pkl.dump(history.history, open(os.path.join(paths.path2Models, 'model_history.pkl'), 'wb'))

if args.predict:

  autoencoder = load_model(os.path.join(paths.path2Models,'Autoencoder_model_{}'.format(args.network)))
  encoder = load_model(os.path.join(paths.path2Models,'Encoder_model_{}'.format(args.network)))
  decoder = load_model(os.path.join(paths.path2Models,'Decoder_model_{}'.format(args.network)))

  #fig, axs = plt.subplots(10, 10, figsize=(20, 20))

  sounds_to_encode = '/home/user/Documents/Antonin/Code/Dimmy/Sounds_beh/4_sec'
  print(n.natsorted(os.listdir(sounds_to_encode)))

  all_latent = []
  for i, f in enumerate(n.natsorted(os.listdir(sounds_to_encode))):
    X_test = proc.load_file(os.path.join(sounds_to_encode, f)).reshape(1, 513, 126)
    X_test = X_test[:, :512, :124]

    latent_repre = encoder(X_test)

    plt.imshow(latent_repre.reshape(10, 10))
    plt.tight_layout()
    plt.savefig(os.path.join(paths.path2Output, '{}png'.format(f[:-3])))
    plt.close()
    all_latent.append(latent_repre)

  #   axs[i//10, i%10].imshow(latent_repre.reshape(10, 10))
  #   axs[i//10, i%10].axes.get_xaxis().set_visible(False)
  #   axs[i//10, i%10].axes.get_yaxis().set_visible(False)

  # plt.tight_layout()
  # plt.show()

  all_latent = np.array(all_latent)

  corr_matrix = proc.correlation_matrix(all_latent)
  plt.imshow(corr_matrix)
  plt.savefig(os.path.join(paths.path2Output, 'corr_matrix.png'))

  plt.show()


history = pkl.load(open(os.path.join(paths.path2Models, 'model_history.pkl'), 'rb'))
plt.plot(history['loss'])
plt.savefig('Output/model_history.png')



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









