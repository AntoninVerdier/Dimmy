import os
import json
import datetime
import argparse
import numpy as np
import natsort as n
import pickle as pkl
from sklearn import preprocessing as p

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, MaxPooling2D, UpSampling2D, Conv2DTranspose
from sklearn.model_selection import train_test_split

import visualkeras
import matplotlib
from PIL import ImageFont
import matplotlib.pyplot as plt
from rich import print, traceback
traceback.install()
from rich.progress import track

import settings as s
import preproc as proc
from Models import Autoencoder, DenseMax


# Define arguments for inline parsing
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
parser.add_argument('--batch_size', '-b', type=int, default=32,
                    help='Choose batch size')
parser.add_argument('--callbacks', '-c', action='store_true',
                    help='Choose if there is a tensorboard callback')
parser.add_argument('--max_n', '-mn', type=int, default=100,
                    help='Number of led to be lit up')
parser.add_argument('--visualize', '-v', action='store_true', 
                    help='flag to visualize a network')
parser.add_argument('--quicktest', '-qt', type=str, default=None,
                    help='Placeholder for name and description')
args = parser.parse_args()




# Execute training if inline argument is passed
if args.train:
  # Tensorboard for weight and training evaluation - maye move to W&B

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
  
  # Get time and date for record when saving
  today = datetime.date.today()
  time = datetime.datetime.now()

  # Quick infos on the network for record
  if not args.quicktest:
    net_name = input('Name of the network > ')
    description = input('Small description of the current network training, for record > ')
  else:
    net_name, description = args.quicktest, args.quicktest
  
  # Datasets
  input_dataset_file = 'heardat_noise_datasetv2_60.pkl'
  output_dataset_file = 'heardat_clean_datasetv2_60.pkl'

  # Distinguish between noisy input and clean reconstruction target
  X_train = np.load(open(input_dataset_file, 'rb'), allow_pickle=True)
  X_train_c = np.load(open(output_dataset_file, 'rb'), allow_pickle=True)


  # Select the desired portion of the data and shuffle it
  shuffle_mask = np.random.choice(X_train.shape[0], int(args.data_size/100 * X_train.shape[0]), replace=False)
  X_train = X_train[shuffle_mask]
  X_train_c = X_train_c[shuffle_mask]
  
  # This to enable fair splitting for convolution. Configured for spectrogram training
  if args.network: 
    input_shape = (128, 112)
    X_train = X_train[:, :, :input_shape[1]]
    X_train_c = X_train_c[:, :, :input_shape[1]]

  # Create a validation set
  X_train, X_valid, X_train_c, X_valid_c = train_test_split(X_train, X_train_c, test_size=0.2, shuffle=True)

  # Create network class
  auto = Autoencoder('{net}'.format(net=args.network if args.network else 'dense'), input_shape, params.latent_size, max_n=args.max_n)

  # Use if model is created with tune tag, performing hyperparameter search 
  if "tune" in args.network:
    tuner = auto.get_model()
    tuner.search(X_train, X_train_c,
            validation_data=(X_valid, X_valid_c),
            epochs=params.epochs,
            batch_size=args.batch_size)
    best_model = tuner.get_best_models()[0]

  # Retrive compiled model from network class
  encoder, decoder, autoencoder = auto.get_model()

  # Launch training with callbacks to tensorboard if specified in inline command
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


  ################" SAVING MODEL INFO #########################

  # Add a timestamp to log files and model name so file is unique. 
  # Add ID to load it faster for further exp - WIP
  
  td = datetime.datetime.now() - time
  training_time = '{}h {}m {}s'.format(td.seconds//3600, (td.seconds//60)%60, td.seconds%60)

  today = today.strftime("%d%m%Y")
  time = time.strftime("%H%M%S")


  save_model_path = os.path.join(paths.path2Models, '{}_{}_{}'.format(today, time, net_name))
  if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
    os.makedirs(os.path.join(save_model_path, 'viz'))
    os.makedirs(os.path.join(save_model_path, 'predict', 'latent', 'data', 'sharp'))
    os.makedirs(os.path.join(save_model_path, 'predict', 'latent', 'data', 'blurred'))
    os.makedirs(os.path.join(save_model_path, 'predict', 'latent', 'img', 'sharp'))
    os.makedirs(os.path.join(save_model_path, 'predict', 'latent', 'img', 'blurred'))
    os.makedirs(os.path.join(save_model_path, 'predict', 'spec', 'data'))
    os.makedirs(os.path.join(save_model_path, 'predict', 'spec', 'img', 'both'))
    os.makedirs(os.path.join(save_model_path, 'predict', 'spec', 'img', 'indiv'))






  autoencoder.save(os.path.join(save_model_path, 'Autoencoder_model_{}_{}'.format(args.network, net_name)))
  encoder.save(os.path.join(save_model_path, 'Encoder_model_{}_{}'.format(args.network, net_name)))
  decoder.save(os.path.join(save_model_path, 'Decoder_model_{}_{}'.format(args.network, net_name)))

  pkl.dump(history.history, open(os.path.join(save_model_path, 'model_history.pkl'), 'wb'))

  print(decoder.get_layer('gaussian_blur').weights[0].shape)

  # Save arguments for record
  args_dict = vars(args)

  args_dict['name'] = net_name
  args_dict['desc'] = description
  args_dict['input_dataset_file'] = input_dataset_file
  args_dict['output_dataset_file'] = output_dataset_file
  args_dict['creation_date'] = today
  args_dict['creation time'] = time
  args_dict['training_time'] = training_time
  args_dict['epochs'] = params.epochs
  args_dict['blurring_kernel_size'] = decoder.get_layer('gaussian_blur').weights[0].shape[0]

  # Add training time

  with open(os.path.join(save_model_path, 'metadata.json'), 'w') as f:
    json.dump(args_dict, f, indent=4)

  ## Run prediction
  sounds_to_encode = '/home/user/Documents/Antonin/Dimmy/Data/SoundsHearlight'

  # Loop trough each sound and output the latent representation
  for i, f in track(enumerate(n.natsorted(os.listdir(sounds_to_encode))), total=len(os.listdir(sounds_to_encode))):
    # Load soundfile and compute spectrogram
    X_test = proc.load_unique_file(os.path.join(sounds_to_encode, f), mod='log', cropmid=True).reshape(1, 128, 126)
    X_test = X_test[:, :, :112]
    X_test = np.expand_dims(X_test, 3)
    
    # Get prediction
    latent_repre = encoder(X_test)

    blurred_output = Model(inputs=decoder.input, outputs=decoder.get_layer('gaussian_blur').output)
    blurred = blurred_output(latent_repre) 

    #Save intensity pattern and their representations. Blurred and non blurred
    np.save(os.path.join(save_model_path, 'predict', 'latent', 'data', 'sharp', '{}.npy'.format(f[:-4])), latent_repre.reshape(100))
    np.save(os.path.join(save_model_path, 'predict', 'latent', 'data', 'blurred', '{}.npy'.format(f[:-4])), blurred.reshape(100))

    plt.imshow(p.normalize(latent_repre.reshape(10, 10)), cmap='Blues')
    plt.savefig(os.path.join(save_model_path, 'predict', 'latent', 'img', 'sharp', '{}.svg'.format(f[:-4])))
    plt.close()

    plt.imshow(p.normalize(blurred.reshape(10, 10)), cmap='Blues')
    plt.savefig(os.path.join(save_model_path, 'predict', 'latent', 'img', 'blurred', '{}.svg'.format(f[:-4])))
    plt.close()

    final_spec = decoder(latent_repre)


    # Make figure of comparison side by side
    fig, axs = plt.subplots(1, 2)
    np.save(os.path.join(os.path.join(save_model_path, 'predict', 'spec', 'data' '{}.npy'.format(f[:-4]))), final_spec)
    
    axs[0].imshow(X_test.reshape(128, 112), cmap='inferno')
    axs[1].imshow(final_spec.reshape(128, 112), cmap='inferno')
    plt.tight_layout()
    plt.savefig(os.path.join(save_model_path, 'predict', 'spec', 'img', 'both', '{}.svg').format(f[:-4]))
    plt.close()

    plt.imshow(final_spec.reshape(128, 112), cmap='inferno')
    plt.savefig(os.path.join(save_model_path, 'predict', 'spec', 'img', 'indiv', '{}.svg').format(f[:-4]))
    plt.close()

  ## Visualization 

  # Generate figures and change fontsize to see legend
  font = ImageFont.truetype("Arial.ttf", 26)
  visualkeras.layered_view(encoder, os.path.join(save_model_path, 'viz', 'encoder.png'), legend=True, font=font)
  visualkeras.layered_view(decoder, os.path.join(save_model_path, 'viz', 'decoder.png'), legend=True, font=font)


# Enter prediction routine if specified in the inline command
# Kept for experimentation and retrocompatibility with old model saving system
if args.predict:
  # Load model when provided with timstamp in inline command
  autoencoder = load_model(os.path.join(paths.path2Models,'Autoencoder_model_{}'.format(args.network)))
  encoder = load_model(os.path.join(paths.path2Models,'Encoder_model_{}'.format(args.network)))
  decoder = load_model(os.path.join(paths.path2Models,'Decoder_model_{}'.format(args.network)))
  
  print(encoder.summary())
  print(decoder.summary())
  
  # Load sounds from behvaioural tasks - need to supply sounds from task 5
  sounds_to_encode = '/home/user/Documents/Antonin/Dimmy/Data/SoundsHearlight'

  # Loop trough each sound and output the latent representation
  for i, f in track(enumerate(n.natsorted(os.listdir(sounds_to_encode))), total=len(os.listdir(sounds_to_encode))):
    print(f)
    # Load soundfile and compute spectrogram
    X_test = proc.load_unique_file(os.path.join(sounds_to_encode, f), mod='log', cropmid=True).reshape(1, 128, 126)
    X_test = X_test[:, :, :112]
    X_test = np.expand_dims(X_test, 3)
    
    # Get prediction
    latent_repre = encoder(X_test)
    np.save(os.path.join('Latent', 'Stims', '{}_latent.npy'.format(f[:-4])), latent_repre.reshape(100))

    final_spec = decoder(latent_repre)



    # Make figure of comparison side by side
    fig, axs = plt.subplots(1, 2)
    np.save(os.path.join('Latent', 'Specs', '{}_spec.npy'.format(f[:-4])), final_spec)
    
    axs[0].imshow(X_test.reshape(128, 112), cmap='inferno')
    axs[1].imshow(final_spec.reshape(128, 112), cmap='inferno')
    plt.tight_layout()
    plt.savefig(os.path.join('Latent', 'Specs', '{}_spec.svg').format(f[:-4]))
    plt.close()
    #plt.show()


    # Plot latent representation as an intensity pattern
    # plt.imshow(p.normalize(latent_repre.reshape(10, 10)), cmap='Blues')
    # plt.colorbar()
    # plt.savefig('latent_repre_{}.svg'.format(f))
    # plt.close()
    
    # Extract blurred representation from early intermediate layer in decoder
    blurred_output = Model(inputs=decoder.input, outputs=decoder.get_layer('gaussian_blur').output)
    blurred = blurred_output(latent_repre) 
    np.save(os.path.join('Latent', 'Blurred', '{}_latent_blurred.npy'.format(f[:-4])), blurred[0, :, :, 0].reshape(100))


    # plt.imshow(p.normalize(blurred.reshape(10, 10)), cmap='Blues')
    # plt.savefig('blurred.svg')
    # plt.close()

    # # Visualize projection pattern side by side
    # fig, axs = plt.subplots(2)
    # plt.title(f[:-4])
    # axs[0].imshow(p.normalize(latent_repre.reshape(10, 10)), cmap='Blues')
    # axs[1].imshow(p.normalize(blurred.reshape(10, 10)), cmap='Blues')
    # #plt.show()
    # plt.close()

if args.visualize:
  # Use visual keras to have a quick view of the model architecture
  autoencoder = load_model(os.path.join(paths.path2Models,'Autoencoder_model_{}'.format(args.network)))
  encoder = load_model(os.path.join(paths.path2Models,'Encoder_model_{}'.format(args.network)))
  decoder = load_model(os.path.join(paths.path2Models,'Decoder_model_{}'.format(args.network)))


  # Generate figures and change fontsize to see legend
  font = ImageFont.truetype("Arial.ttf", 26)
  visualkeras.layered_view(encoder, 'encoder.png', legend=True, font=font)
  visualkeras.layered_view(decoder, 'decoder.png', legend=True, font=font)





