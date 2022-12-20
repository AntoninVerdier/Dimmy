import os
import json
import datetime
import argparse
import visualkeras
import numpy as np
import natsort as n
import pickle as pkl
import settings as s
import preproc as proc
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import ImageFont
from Models import Autoencoder
from rich.progress import track
from rich import print, traceback
from sklearn import preprocessing as p
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import TensorBoard, CallBack
from tensorflow.keras.models import load_model, Model
from sklearn.model_selection import train_test_split

traceback.install()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mixed_precision.set_global_policy('mixed_float16')

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
parser.add_argument('--epochs', '-e', type=int, default=150,
                    help='Number of epochs')
parser.add_argument('--tuner', '-tu', action='store_true',
                    help='Hyperparameters search')
args = parser.parse_args()

   
# Execute training if inline argument --train is passed
if args.train:
    # Get time and date for record when saving
    today = datetime.date.today()
    time = datetime.datetime.now()

    # Create saving folder now so we can write callbacks in it
    today = today.strftime("%d%m%Y")
    time_str = time.strftime("%H%M%S")

    # If inline argument is passed, create callback instance
    if args.callbacks:
      tensorboard = TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        write_images=True,
        update_freq=5
      )
      keras_callbacks = [tensorboard]

    # Allow user to give a name and quick description of the network for future record
    if not args.quicktest:
      net_name = input('Name of the network > ')
      description = input('Small description of the current network training, for record > ')
    else:
      net_name, description = args.quicktest, args.quicktest

#############################################################################
    # Create folder for logs 
    logs = "new_logs/" + '{}_{}_{}'.format(today, time_str, net_name)
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                     histogram_freq =1,
                                                     profile_batch =(2, 5))
    
#############################################################################  
    # Datasets
    input_dataset_file = 'heardat_noise_datasetv2_60_cqt_128_28k.pkl'
    output_dataset_file = 'heardat_clean_datasetv2_60_cqt_128_28k.pkl'
    toeplitz_true = 'ordered_96_28k_toeplitz.npy'
    toeplitz_spec = 'topelitz_gaussian_cxe_28k.npy'

    # Distinguish between noisy input and clean reconstruction target
    X_train = np.load(open(input_dataset_file, 'rb'), allow_pickle=True).astype('float32')/255.0
    X_train_c = np.load(open(output_dataset_file, 'rb'), allow_pickle=True).astype('float32')/255.0

    # Select the desired portion of the data and shuffle it
    shuffle_mask = np.random.choice(X_train.shape[0], int(args.data_size/100 * X_train.shape[0]), replace=False)
    X_train = X_train[shuffle_mask]
    X_train_c = X_train_c[shuffle_mask]
    
    # This to enable fair splitting for convolution. Configured for spectrogram training
    input_shape = (X_train.shape[1] - X_train.shape[1]%16, X_train.shape[2] - X_train.shape[2]%16)
    X_train = X_train[:, :input_shape[0], :input_shape[1]]
    X_train_c = X_train_c[:, :input_shape[0], :input_shape[1]]

    # Create a validation set
    X_train, X_valid, X_train_c, X_valid_c = train_test_split(X_train, X_train_c, test_size=0.2, shuffle=True)

    # Create network class
    auto = Autoencoder('{net}'.format(net=args.network if args.network else 'dense'), input_shape, params.latent_size, max_n=args.max_n, toeplitz_true=toeplitz_true, toeplitz_spec=toeplitz_spec)

    # Retrive compiled model from network class
    autoencoder = auto.get_model()

    save_model_path = os.path.join(paths.path2Models, '{}_{}_{}'.format(today, time_str, net_name))
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
      os.makedirs(os.path.join(save_model_path, 'Callbacks', 'Dat'))
      os.makedirs(os.path.join(save_model_path, 'Callbacks', 'Img'))
      os.makedirs(os.path.join(save_model_path, 'Performances', 'Img'))
      os.makedirs(os.path.join(save_model_path, 'Performances', 'Data'))

    # Launch training with callbacks to tensorboard if specified in inline command
    class ToeplitzLogger(Callback):
      """ Logger function called at each epoch end to generate a correlation matrix to see evolution 

      Attributes
      ----------
      test_freq : array
        Array of spectrograms of pure tones to be predicted
      
      Methods
      -------
      on_epoch_end
        Generate a array and a figure of the autocorrelation of predicted pure tones
      """      
      def __init__(self):
        """Constructor
        
        Load array of spectrograms of pure tones to be predicted and format and normalize them
        to the required shape.
        """
        self.test_freq = np.load(os.path.join('toeplitz', toeplitz_true), allow_pickle=True)[:, :input_shape[0], :input_shape[1]]/255.0
        

      def on_epoch_end(self, epoch):
        """ Method called at the end of the epoch. 

        Save both an array as a .npy file and a heatmap of the array as .svg. The
        array is an autocorrelation matrix of the latent representation of pure tone 
        predicted by the still-training autoencoder.

        Parameters
        ----------
        epoch : int
          Current epoch number
        """
        pred_freq_corr = autoencoder(self.test_freq)[1]

        def t(a): return tf.transpose(a)

        x = pred_freq_corr
        mean_t = tf.reduce_mean(x, axis=1, keepdims=True)
        #cov_t = x @ t(x)
        cov_t = ((x-mean_t) @ t(x-mean_t))/(pred_freq_corr.shape[1]-1)
        cov2_t = tf.linalg.diag(1/tf.sqrt(tf.linalg.diag_part(cov_t)))
        cor = cov2_t @ cov_t @ cov2_t

        np.save(os.path.join(save_model_path, 'Callbacks', 'Dat', 'log_top_{}.npy'.format(epoch)), cor)
        plt.imshow(cor, vmin=0, vmax=1)
        plt.savefig(os.path.join(save_model_path, 'Callbacks', 'Img','log_top_{}.svg'.format(epoch)))
        plt.close()
    
    class ToeplitzLogger_am(Callback):
      """ Logger function called at each epoch end to generate a correlation matrix to see evolution 

      Attributes
      ----------
      test_freq : array
        Array of spectrograms of amplitude modulated tones to be predicted
      
      Methods
      -------
      on_epoch_end
        Generate a array and a figure of the autocorrelation of predicted amplitude modulated tones
      """      
 
      def __init__(self):
        """Constructor
        
        Load array of spectrograms of amplitude-modulated tones to be predicted and format and normalize them
        to the required shape.
        """
        self.test_freq_am = np.load(os.path.join('toeplitz', 'toep_am_10k.npy'), allow_pickle=True)[:, :input_shape[0], :input_shape[1]]/255.0
        

      def on_epoch_end(self, epoch, logs=None):
        """ Method called at the end of the epoch. 

        Save both an array as a .npy file and a heatmap of the array as .svg. The
        array is an autocorrelation matrix of the latent representation of amplitude-modulated tone 
        predicted by the still-training autoencoder.

        Parameters
        ----------
        epoch : int
          Current epoch number
        """
          pred_freq_corr = autoencoder(self.test_freq_am)[2]

          def t(a): return tf.transpose(a)

          x = pred_freq_corr
          mean_t = tf.reduce_mean(x, axis=1, keepdims=True)
          #cov_t = x @ t(x)
          cov_t = ((x-mean_t) @ t(x-mean_t))/(pred_freq_corr.shape[1]-1)
          cov2_t = tf.linalg.diag(1/tf.sqrt(tf.linalg.diag_part(cov_t)))
          cor = cov2_t @ cov_t @ cov2_t

          np.save(os.path.join(save_model_path, 'Callbacks', 'Dat', 'logam_top_{}.npy'.format(epoch)), cor)
          plt.imshow(cor, vmin=0, vmax=1)
          plt.savefig(os.path.join(save_model_path, 'Callbacks', 'Img','logam_top_{}.svg'.format(epoch)))
          plt.close()

    # Finally train the model
    history = autoencoder.fit(X_train, X_train_c,
                            validation_data=(X_valid, X_valid_c),
                            epochs=args.epochs, 
                            use_multiprocessing=True,
                            batch_size=args.batch_size if args.batch_size else 32,
                            callbacks=[ToeplitzLogger(), ToeplitzLogger_am(), tboard_callback])



    ################" SAVING MODEL INFO #########################

    # Add a timestamp to log files and model name so file is unique. 
    # Add ID to load it faster for further exp - WIP
    td = datetime.datetime.now() - time
    training_time = '{}h {}m {}s'.format(td.seconds//3600, (td.seconds//60)%60, td.seconds%60)

    autoencoder.save(os.path.join(save_model_path, 'Autoencoder_model_{}_{}'.format(args.network, net_name)))

    pkl.dump(history.history, open(os.path.join(save_model_path, 'model_history.pkl'), 'wb'))

    # Save arguments for record
    args_dict = vars(args)

    args_dict['name'] = net_name
    args_dict['desc'] = description
    args_dict['input_dataset_file'] = input_dataset_file
    args_dict['output_dataset_file'] = output_dataset_file
    args_dict['creation_date'] = today
    args_dict['creation time'] = time_str
    args_dict['training_time'] = training_time
    args_dict['epochs'] = params.epochs
    args_dict['blurring_kernel_size'] = autoencoder.get_layer('gaussian_blur').weights[0].shape[0]
    args_dict['best_loss'] = np.min(history.history['loss'])
    args_dict['end_loss'] = history.history['loss'][-1]

    # Output keras model summary in txt file
    with open(os.path.join(save_model_path, 'summary.txt'), 'w') as f:
      autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))

    # Save all configurations parameters and training stats in jdon file
    with open(os.path.join(save_model_path, 'metadata.json'), 'w') as f:
      json.dump(args_dict, f, indent=4)

    ## Run prediction
    sounds_to_encode = '/home/user/Documents/Antonin/Dimmy/Data/SoundsHearlight'

    # Loop trough each sound and output the latent representation
    for i, f in track(enumerate(n.natsorted(os.listdir(sounds_to_encode))), total=len(os.listdir(sounds_to_encode))):
      # Normalize with exponential function to avoid under-representation of high frequencies
      x = np.arange(1, 129)
      y = 0.5*np.exp(-0.022*x)
      y = 1/(np.repeat(y, 126).reshape(128, 126))

      # Load soundfile and compute spectrogram
      X_test = np.expand_dims(proc.load_unique_file_cqt(os.path.join(sounds_to_encode, f), y), 0)
      X_test = X_test.astype('float32')/255.0
      X_test = X_test[:, :input_shape[0], :input_shape[1]]
      X_test = np.expand_dims(X_test, 3)

      # Construct encoder with and without gaussian blurring from entire model
      encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('Dense_maxn').output)
      blurred_output = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('gaussian_blur').output)

      # Get sharp and blurred latent representation 
      latent_repre = encoder(X_test)
      blurred = blurred_output(X_test) 

      #Save intensity pattern and their representations. Blurred and non blurred
      np.save(os.path.join(save_model_path, 'predict', 'latent', 'data', 'sharp', '{}.npy'.format(f[:-4])), latent_repre.reshape(100))
      np.save(os.path.join(save_model_path, 'predict', 'latent', 'data', 'blurred', '{}.npy'.format(f[:-4])), blurred.reshape(100))

      plt.imshow(p.normalize(latent_repre.reshape(10, 10)), cmap='Blues')
      plt.savefig(os.path.join(save_model_path, 'predict', 'latent', 'img', 'sharp', '{}.svg'.format(f[:-4])))
      plt.close()

      plt.imshow(p.normalize(blurred.reshape(10, 10)), cmap='Blues')
      plt.savefig(os.path.join(save_model_path, 'predict', 'latent', 'img', 'blurred', '{}.svg'.format(f[:-4])))
      plt.close()

      final_spec = autoencoder(X_test)[0]

      # Make figure of comparison side by side
      fig, axs = plt.subplots(1, 2)
      np.save(os.path.join(os.path.join(save_model_path, 'predict', 'spec', 'data' '{}.npy'.format(f[:-4]))), final_spec)
      
      axs[0].imshow(X_test.reshape(input_shape[0], input_shape[1]), cmap='inferno')
      axs[1].imshow(final_spec.reshape(input_shape[0], input_shape[1]), cmap='inferno')
      plt.tight_layout()
      plt.savefig(os.path.join(save_model_path, 'predict', 'spec', 'img', 'both', '{}.svg').format(f[:-4]))
      plt.close()

      plt.imshow(final_spec.reshape(input_shape[0], input_shape[1]), cmap='inferno')
      plt.savefig(os.path.join(save_model_path, 'predict', 'spec', 'img', 'indiv', '{}.svg').format(f[:-4]))
      plt.close()

    ## Visualization 

    # Generate figures and change fontsize to see legend
    font = ImageFont.truetype("Arial.ttf", 26)
    visualkeras.layered_view(autoencoder, os.path.join(save_model_path, 'viz', 'autoencoder.png'), legend=True, font=font)
    # visualkeras.layered_view(decoder, os.path.join(save_model_path, 'viz', 'decoder.png'), legend=True, font=font)

    # Correlation matrix
    path = os.path.join(save_model_path, 'predict', 'latent', 'data', 'sharp')
    filenames = n.natsorted(os.listdir(path))
    np.save(os.path.join(save_model_path, 'Performances', 'Data', 'filenames.npy'), filenames)
    all_latent = np.array([np.load(os.path.join(path, s)) for s in n.natsorted(os.listdir(path))]).reshape(len(os.listdir(path)), 100)

    corr_matrix = proc.correlation_matrix(all_latent)
    np.save(os.path.join(save_model_path, 'Performances', 'Data','corr_matrix.npy'), corr_matrix)

    plt.figure(figsize=(8, 8), dpi=100)
    plt.imshow(corr_matrix, cmap='viridis')
    plt.savefig(os.path.join(save_model_path, 'Performances', 'Img','corr_matrix.svg'))
    plt.close()
    
    path = os.path.join(save_model_path, 'predict', 'latent', 'data', 'blurred')
    filenames = n.natsorted(os.listdir(path))
    np.save(os.path.join(save_model_path, 'Performances', 'Data', 'filenames_blurred.npy'), filenames)
    corr_matrix_blurred = proc.correlation_matrix(all_latent)
    np.save(os.path.join(save_model_path, 'Performances', 'Data','corr_matrix_blurred.npy'), corr_matrix_blurred)

    plt.figure(figsize=(8, 8), dpi=100)
    plt.imshow(corr_matrix, cmap='viridis')
    plt.savefig(os.path.join(save_model_path, 'Performances', 'Img','corr_matrix_blurred.svg'))
    plt.close()

    plt.figure(figsize=(6, 8), dpi=100)
    plt.plot(history.history['loss'], color='blue', label='loss')
    plt.plot(history.history['val_loss'], color='orange', label='val_loss')
    plt.legend()
    plt.savefig(os.path.join(save_model_path, 'Performances', 'Img','loss.svg'))
    plt.close()

# Enter prediction routine if specified in the inline command
# Kept for experimentation and retrocompatibility with old model saving system
if args.predict:
  # Load model when provided with timstamp in inline command
  autoencoder = load_model(os.path.join(paths.path2Models,'Autoencoder_model_{}'.format(args.network)))
  encoder = load_model(os.path.join(paths.path2Models,'Encoder_model_{}'.format(args.network)))
  decoder = load_model(os.path.join(paths.path2Models,'Decoder_model_{}'.format(args.network)))

  
  # Load sounds from behvaioural tasks - need to supply sounds from task 5
  sounds_to_encode = '/home/user/Documents/Antonin/Dimmy/Data/SoundsHearlight'

  # Loop trough each sound and output the latent representation
  for i, f in track(enumerate(n.natsorted(os.listdir(sounds_to_encode))), total=len(os.listdir(sounds_to_encode))):
    print(f)
    # Load soundfile and compute spectrogram
    X_test = proc.load_unique_file(os.path.join(sounds_to_encode, f)).reshape(1, 128, 126)
    X_test = X_test.astype('float32')/255.0

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(X_test.reshape(128, 126))

    X_test = X_test[:, :, :112]

    axs[1].imshow(X_test.reshape(128, 112))

    X_test = np.expand_dims(X_test, 3)

    # Get prediction
    latent_repre = encoder(X_test)
    np.save(os.path.join('Latent', '{}_latent.npy'.format(f[:-4])), latent_repre.reshape(100))

    final_spec = decoder(latent_repre)
    #np.save(os.path.join('Latent', '{}_spec.npy'.format(f[:-4])), final_spec)
    plt.imshow(final_spec.reshape(128, 112))
    plt.close()

    # Extract blurred representation from early intermediate layer in decoder
    blurred_output = Model(inputs=decoder.input, outputs=decoder.get_layer('gaussian_blur').output)
    blurred = blurred_output(latent_repre)
    np.save(os.path.join('Latent', 'blurred{}_latent.npy'.format(f[:-4])), latent_repre.reshape(100))
    

    plt.imshow(p.normalize(blurred.reshape(10, 10)), cmap='Blues')
    plt.savefig('blurred.svg')
    plt.close()

    # Visualize projection pattern side by side
    fig, axs = plt.subplots(2)
    plt.title(f[:-4])
    axs[0].imshow(p.normalize(latent_repre.reshape(10, 10)), cmap='Blues')
    axs[1].imshow(p.normalize(blurred.reshape(10, 10)), cmap='Blues')
    #plt.show()
    plt.close()

# Enter visualisation routine if specified in the inline command
if args.visualize:
  # Use visual keras to have a quick view of the model architecture
  autoencoder = load_model(os.path.join(paths.path2Models,'Autoencoder_model_{}'.format(args.network)))
  encoder = load_model(os.path.join(paths.path2Models,'Encoder_model_{}'.format(args.network)))
  decoder = load_model(os.path.join(paths.path2Models,'Decoder_model_{}'.format(args.network)))


  # Generate figures and change fontsize to see legend
  font = ImageFont.truetype("Arial.ttf", 26)
  visualkeras.layered_view(encoder, 'encoder.png', legend=True, font=font)
  visualkeras.layered_view(decoder, 'decoder.png', legend=True, font=font)





