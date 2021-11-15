import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import natsort as n
import pickle as pkl

from scipy import signal
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model, Model

from rich import print, traceback
traceback.install()

# from AE import Sampling
# from AE import VAE
import preprocessing as proc
from Models import Autoencoder
from data_gen import DataGenerator, DataGenerator_both, get_generators
import settings as s
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
parser.add_argument('--batch_size', '-b', type=int,
                    help='Choose batch size')
parser.add_argument('--callbacks', '-c', action='store_true',
                    help='Choose if there is a tensorboard callback')
args = parser.parse_args()

# Tensorboard for weight and traingin evaluation

if args.callbacks:
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
      X_train = X_train[:, :512, :112]
      input_shape = (512, 112)

    else:
      input_shape = (513, 124)

    auto = Autoencoder('{net}'.format(net=args.network if args.network else 'dense'), input_shape, params.latent_size)
    encoder, decoder, autoencoder = auto.get_model()

    X_train = np.expand_dims(X_train, 3)
    print(X_train.shape)

    if args.callbacks:
      history = autoencoder.fit(X_train, X_train,
                                epochs=params.epochs, 
                                batch_size=args.batch_size if args.batch_size else 32,
                                callbacks=keras_callbacks)
    else:
      history = autoencoder.fit(X_train, X_train,
                              epochs=params.epochs, 
                              batch_size=args.batch_size if args.batch_size else 32)

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
    proc.convert_to_dlp(latent_repre)

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


