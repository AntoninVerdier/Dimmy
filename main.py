import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import datetime
import matplotlib
import argparse
import numpy as np
import natsort as n
import pickle as pkl
import librosa
import scipy.io as sio
from sklearn import preprocessing as p

import pandas as pd
import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model, Model

from sklearn.model_selection import train_test_split

from rich import print, traceback
traceback.install()

from rich.progress import track

from sklearn.manifold import TSNE

import visualkeras
from PIL import ImageFont
import matplotlib


# from AE import Sampling
# from AE import VAE
import preproc as proc
from Models import Autoencoder
from data_gen import DataGenerator, DataGenerator_both, get_generators
import settings as s

# TCN utilities
import utils_tcn as utcn


from collections import defaultdict



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
parser.add_argument('--max_n', '-mn', type=int, default=100,
                    help='Number of led to be lit up')
parser.add_argument('--visualize', '-v', action='store_true', 
                    help='flag to visualize a network')
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
    X_train = np.load(open('heardat_noise_datasetv2_60.pkl', 'rb'), allow_pickle=True)
    X_train_c = np.load(open('heardat_clean_datasetv2_60.pkl', 'rb'), allow_pickle=True)

    print(X_train.shape)

    # Select the desired portion of the data and shuffle it
    shuffle_mask = np.random.choice(X_train.shape[0], int(args.data_size/100 * X_train.shape[0]), replace=False)
    X_train = X_train[shuffle_mask]
    X_train_c = X_train_c[shuffle_mask]
    print(X_train.shape)


    if args.network: # This to enable fair splitting for convolution
      X_train = X_train[:, :, :112]
      X_train_c = X_train_c[:, :, :112]

      # print(X_train.shape)
      # X_train = X_train[:, :31920]
      # X_train = X_train.reshape(X_train.shape[0], 31920, 1)

      # X_train_c = X_train_c[:, :31920]
      # X_train_c = X_train_c.reshape(X_train_c.shape[0], 31920, 1)
      # print(X_train.shape)



      input_shape = (128, 112)


    X_train, X_valid, X_train_c, X_valid_c = train_test_split(X_train, X_train_c, test_size=0.2, shuffle=True)

    auto = Autoencoder('{net}'.format(net=args.network if args.network else 'dense'), input_shape, params.latent_size, max_n=args.max_n)

    if "tune" in args.network:
      tuner = auto.get_model()
      tuner.search(X_train, X_train_c,
              validation_data=(X_valid, X_valid_c),
              epochs=params.epochs,
              batch_size=args.batch_size)
      best_model = tuner.get_best_models()[0]


    encoder, decoder, autoencoder = auto.get_model()

    #X_train = np.expand_dims(X_train, 2)


    print(X_train.shape)

    if args.callbacks:
      history = autoencoder.fit(X_train, X_train_c,
                                validation_data=(X_valid, X_valid_c),
                                epochs=params.epochs, 
                                batch_size=args.batch_size if args.batch_size else 32,
                                callbacks=keras_callbacks)
    else:
      history = autoencoder.fit(X_train, X_train_c,
                              validation_data=(X_valid, X_valid_c),
                              epochs=params.epochs, 
                              batch_size=args.batch_size if args.batch_size else 32)

    ts = str(int(datetime.datetime.now().timestamp())) + '_' + str(int(args.max_n))
    autoencoder.save(os.path.join(paths.path2Models, 'Autoencoder_model_{}_{}'.format(args.network, ts)))
    encoder.save(os.path.join(paths.path2Models, 'Encoder_model_{}_{}'.format(args.network, ts)))
    decoder.save(os.path.join(paths.path2Models, 'Decoder_model_{}_{}'.format(args.network, ts)))

    pkl.dump(history.history, open(os.path.join(paths.path2Models, 'model_history_{}.pkl'.format(ts)), 'wb'))

if args.predict:

  autoencoder = load_model(os.path.join(paths.path2Models,'Autoencoder_model_{}'.format(args.network)))
  encoder = load_model(os.path.join(paths.path2Models,'Encoder_model_{}'.format(args.network)))
  decoder = load_model(os.path.join(paths.path2Models,'Decoder_model_{}'.format(args.network)))  
  print(encoder.summary())
  print(decoder.summary())
  
  sounds_to_encode = '/home/user/Documents/Antonin/Dimmy/Data/SoundsHearlight'

  for i, f in track(enumerate(n.natsorted(os.listdir(sounds_to_encode))), total=len(os.listdir(sounds_to_encode))):
    print(f)
    X_test = proc.load_unique_file(os.path.join(sounds_to_encode, f), mod='log', cropmid=True).reshape(1, 128, 126)
    X_test = X_test[:, :, :112]
    X_test = np.expand_dims(X_test, 3)
    print(X_test.shape)
    latent_repre = encoder(X_test)

    plt.imshow(p.normalize(latent_repre.reshape(10, 10)), cmap='Blues')
    plt.colorbar()
    plt.savefig('latent_repre.svg')
    plt.close()
    blurred_output = Model(inputs=decoder.input, outputs=decoder.get_layer('gaussian_blur').output)
    

    blurred = blurred_output(latent_repre)    

    plt.imshow(p.normalize(blurred.reshape(10, 10)), cmap='Blues')
    plt.savefig('blurred.svg')
    plt.close()
    break;

if args.visualize:
  autoencoder = load_model(os.path.join(paths.path2Models,'Autoencoder_model_{}'.format(args.network)))
  encoder = load_model(os.path.join(paths.path2Models,'Encoder_model_{}'.format(args.network)))
  decoder = load_model(os.path.join(paths.path2Models,'Decoder_model_{}'.format(args.network)))

  from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, MaxPooling2D, UpSampling2D, Conv2DTranspose
  from Models import DenseMax


  font = ImageFont.truetype("Arial.ttf", 26)
  visualkeras.layered_view(encoder, 'encoder.png', legend=True, font=font)
  visualkeras.layered_view(decoder, 'decoder.png', legend=True, font=font)




##### DEEPEN TESTING #########
  # X_test = np.load('deepen_test.pkl', allow_pickle=True)
  # X_test = X_test[:, :, :376]

  # X_test = np.expand_dims(X_test, 3)
  # latent_repre = encoder.predict(X_test)
  # np.save('latents_sophie', latent_repre, allow_pickle=True)

  # plt.imshow(latent_repre[100].reshape(10, 10))
  # plt.show()
##############################




  #   latent_repre = encoder(X_test)
  #   decoded_spec = autoencoder(X_test)



  #   all_latent.append(latent_repre)

  # np.save('latents_sophie.pkl', all_latent, allow_pickle=True)

  
  # names = [f[:-4] for f in n.natsorted(os.listdir(sounds_to_encode))]

  # X_rec =  autoencoder.predict(test_X)
  # for i, sound in enumerate(X_rec):
  #   sio.wavfile.write('/home/user/Documents/Antonin/Dimmy/Output/Sounds/{}.wav'.format(names[i]), 64000, sound)
  
  # # do some padding in the end, since not necessarily the whole time series is reconstructed
  # X_rec = np.pad(X_rec, ((0,0),(0, test_X.shape[1] - X_rec.shape[1] ), (0,0)), 'constant').reshape(62, 32000, 1) 
  # E_rec = (X_rec - test_X.reshape(62, 32000, 1)).squeeze()
  # Err = utcn.slide_window(pd.DataFrame(E_rec), 128, verbose = 0)
  # Err = Err.reshape(-1, Err.shape[-1]*Err.shape[-2])
  # sel = np.random.choice(range(Err.shape[0]),int(Err.shape[0]*0.98))
  # mu = np.mean(Err[sel], axis=0)
  # cov = np.cov(Err[sel], rowvar = False)
  # sq_mahalanobis = utcn.mahalanobis_distance(X=Err[:], cov=cov, mu=mu)
  # # moving average over mahalanobis distance. Only slightly smooths the signal
  # anomaly_score = np.convolve(sq_mahalanobis, np.ones((50,))/50, mode='same')
  # anomaly_score = np.sqrt(anomaly_score)
  

  # print(anomaly_score)

#   #fig, axs = plt.subplots(10, 10, figsize=(20, 20))
#   cmap = matplotlib.cm.get_cmap('hsv')
#   sounds_to_encode = '/home/user/Documents/Antonin/Dimmy/Data/SoundsHearlight'
#   all_latent = []
#   colors = [cmap(0.1)]*6 + [cmap(0.3)]*24 + [cmap(1)] + [cmap(0.5)]*16 + [cmap(0.7)]*16

#   # X_test = [proc.load_file(os.path.join(sounds_to_encode, f), mod='log').reshape(1, 128, 126, 1) for i, f in enumerate(n.natsorted(os.listdir(sounds_to_encode)))]
#   # X_test = np.array(X_test).reshape(len(X_test), 256, 64, 1)
#   # print(autoencoder.evaluate(X_test, X_test))

  # for i, f in enumerate(n.natsorted(os.listdir(sounds_to_encode))):
  #   print(f)
  #   X_test = proc.load_unique_file((os.path.join(sounds_to_encode, f), 'log')).reshape(1, 128, 126, 1)
  #   X_test = X_test[:, :, :112, :]



  #   latent_repre = encoder(X_test)
  #   decoded_spec = autoencoder(X_test)



  #   all_latent.append(latent_repre)

    # if 'AM_' in f:
    #   colors.append(cmap(0.1)) # Orange
    # elif 'AMN_' in f:
    #   colors.append(cmap(0.3)) # Vert
    # elif 'PT_' in f:
    #   colors.append(cmap(0.7))
    # elif 'Steps_' in f:
    #   colors.append(cmap(0.5))
    # elif 'Chirp_' in f:
    #   colors.append(cmap(0.5))
    
  #   fig, axs = plt.subplots(2, 1)
    
  #   axs[0].imshow(X_test[0].T[0])
  #   axs[1].imshow(decoded_spec[0].T[0])
    
  #   axs[0].set_title('Sound input')
  #   axs[1].set_title('Retrieved spectrogram')

  #   plt.savefig(os.path.join(paths.path2Output, 'Specs', '{}.png'.format(f[:-4])), dpi=300)
  #   plt.close()

  #   np.save(os.path.join(paths.path2OutputD, '{}.npy'.format(f[:-4])), latent_repre.numpy())

  #   # proc.convert_to_dlp(latent_repre)

  #   plt.imshow(latent_repre.reshape(10, 10))
  #   plt.tight_layout()
  #   plt.savefig(os.path.join(paths.path2Output, '{}png'.format(f[:-3])))
  #   #np.save(latent_repre, os.path.join(paths.path2OutputD, '{}.npy'.format(f[:-3])))
  #   plt.close()
  #   # all_latent.append(latent_repre)

  # #   axs[i//10, i%10].imshow(latent_repre.reshape(10, 10))
  # #   axs[i//10, i%10].axes.get_xaxis().set_visible(False)
  # #   axs[i//10, i%10].axes.get_yaxis().set_visible(False)

  # # plt.tight_layout()
  # # plt.show()


  # all_latent = np.array(all_latent).reshape(-1, 100)
  # print(all_latent.shape)

  # clf = TSNE()
  # Y = clf.fit_transform(all_latent)
  # plt.scatter(Y[:, 0], Y[:, 1], c=colors)
  # plt.savefig('Output/TSNE/tsne.png')
  # plt.show()




  # corr_matrix = proc.correlation_matrix(all_latent)
  # plt.imshow(corr_matrix)
  # plt.savefig(os.path.join(paths.path2Output, 'corr_matrix.png'))

  # plt.show()


# history = pkl.load(open(os.path.join(paths.path2Models, 'model_history.pkl'), 'rb'))
# plt.plot(history['loss'])
# plt.savefig('Output/model_history.png')


