import os
import talos
import json
import datetime
import argparse
import numpy as np
import natsort as n
import pickle as pkl
from sklearn import preprocessing as p

from Models import DenseMax

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model, Model
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import Callback

import visualkeras
from PIL import ImageFont
import matplotlib.pyplot as plt
from rich import print, traceback
traceback.install()
from rich.progress import track

import settings as s
import preproc as proc
from Models import Autoencoder

import tensorflow as tf

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, InputLayer, Flatten, Reshape, Layer, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, DepthwiseConv2D, LeakyReLU

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras import backend as K
from tensorflow.keras.constraints import UnitNorm

import tensorflow.experimental.numpy as tnp
import tensorflow_probability as tfp

# from AE import Sampling
# from AE import VAE

from sklearn import preprocessing as p


from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import matplotlib.pyplot as plt


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


# Datasets
input_dataset_file = 'heardat_noise_datasetv2_60_cqt_128_28k.pkl'
output_dataset_file = 'heardat_clean_datasetv2_60_cqt_128_28k.pkl'
toeplitz_true = 'toeplitz_offset_cqt_128_28k.pkl'
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

true_freq_corr = tf.convert_to_tensor(np.load(os.path.join('toeplitz', toeplitz_spec)))
test_freq = tf.convert_to_tensor(np.load(os.path.join('toeplitz', toeplitz_true), allow_pickle=True)[:, :input_shape[0], :input_shape[1]])

# Create a validation set
X_train, X_valid, X_train_c, X_valid_c = train_test_split(X_train, X_train_c, test_size=0.2, shuffle=True)

# train_ds = tf.data.Dataset.from_tensor_slices((X_train, X_train_c)).batch(args.batch_size)
# valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, X_valid_c)).batch(args.batch_size)


# Create network class
# Retrive compiled model from network class
# Launch training with callbacks to tensorboard if specified in inline command

class ToeplitzLogger(Callback):
    def __init__(self):
        self.test_freq = np.load(os.path.join('toeplitz', toeplitz_true), allow_pickle=True)[:, :input_shape[0], :input_shape[1]]

    def on_epoch_end(self, epoch, logs=None):
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


def conv_small_tune(X_train, y_train, X_val, y_val, params):
    print(params)
    def fn_smoothing(y_true, y_pred):

        pred_freq_corr = autoencoder(test_freq)[1]

        def t(a): return tf.transpose(a)

        x = pred_freq_corr
        mean_t = tf.reduce_mean(x, axis=1, keepdims=True)
        #cov_t = x @ t(x)
        cov_t = ((x-mean_t) @ t(x-mean_t))/(pred_freq_corr.shape[1]-1)
        cov2_t = tf.linalg.diag(1/tf.sqrt(tf.linalg.diag_part(cov_t)))
        cor = cov2_t @ cov_t @ cov2_t

        # Fin how to compute autocorrelation matrix
        loss = tf.keras.losses.mean_squared_error(true_freq_corr, cor)
        # May need to return output with batch size 

        return loss

    def normalized_mse(y_true, y_pred):
        loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

        return loss

    optadam = keras.optimizers.Adam(learning_rate=0.001)

    kernel_size = 3

    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x*x + y*y)

    # Considering 1 px = 150um
    sigma = np.sqrt(np.log(2)/2)
    muu = 0.000

    kernel_weights = np.exp(-((dst-muu)**2 / (2.0 * sigma**2)))

    kernel_weights = np.expand_dims(kernel_weights, axis=-1)
    kernel_weights = np.repeat(kernel_weights, 1, axis=-1)
    kernel_weights = np.expand_dims(kernel_weights, axis=-1)

    def gaussian_blur_filter(shape, dtype=None):
        f = np.array(kernel_weights)

        assert f.shape == shape
        return K.variable(f)

    gaussian_blur = Conv2D(1, kernel_size, use_bias=False, kernel_initializer=gaussian_blur_filter, padding='same', trainable=False, name='gaussian_blur')



    inputs = Input((*input_shape, 1))

    x = Conv2D(params['conv_1_units'], kernel_size=(params['kernel_1_size'], params['kernel_1_size']), padding='same', activation='relu', name='E_conv_1')(inputs)
    x = MaxPooling2D((2, 2), padding="same", name='E_pool_1')(x)
    x = Conv2D(params['conv_2_units'], kernel_size=(params['kernel_2_size'], params['kernel_2_size']) , padding='same', activation='relu', name='E_conv_2')(x)
    # x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding="same", name='E_pool_2')(x)
    x = Conv2D(params['conv_3_units'], kernel_size=(params['kernel_3_size'], params['kernel_3_size']), padding='same', activation='relu', name='E_conv_3')(x)
    # x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding="same", name='E_pool_3')(x)
    x = Conv2D(params['conv_4_units'], kernel_size=(params['kernel_4_size'], params['kernel_4_size']), padding='same', activation='relu', name='E_conv_4')(x)
    x = MaxPooling2D((2, 2), padding="same", name='E_pool_4')(x)

    x = Flatten()(x)
    dnmax = DenseMax(100, max_n=10, lambertian=False, kernel_constraint=UnitNorm(), name='Dense_maxn')(x)
    
    x = Reshape((10, 10, 1))(dnmax)

    # x = Dropout(0.1)(x)
    x = gaussian_blur(x)
    encoded = Reshape((100,), dtype=tf.float32)(x)
    
    x = Dense(int(input_shape[0]/16)*int(input_shape[1]/16)*16)(encoded)
    x = Reshape((int(input_shape[0]/16), int(input_shape[1]/16), -1))(x)
    x = UpSampling2D((2, 2), name='D_upsamp_0')(x)
    x = Conv2DTranspose(params['conv_4_units'], (params['kernel_4_size'], params['kernel_4_size']), strides=1, activation='relu', padding="same", name='D_conv_1')(x)
    #x = LeakyReLU(alpha=0.3)(x)
    x = UpSampling2D((2, 2), name='D_upsamp_1')(x)
    # x = Dropout(0.1)(x)
    x = Conv2DTranspose(params['conv_3_units'], (params['kernel_3_size'], params['kernel_3_size']), strides=1, activation='relu', padding="same", name='D_conv_2')(x)
    #x = LeakyReLU(alpha=0.3)(x)
    x = UpSampling2D((2, 2), name='D_upsamp_2')(x)
    # x = Dropout(0.1)(x)
    x = Conv2DTranspose(params['conv_2_units'], (params['kernel_2_size'], params['kernel_2_size']), strides=1, activation='relu', padding="same", name='D_conv_3')(x)
    #x = LeakyReLU(alpha=0.3)(x)
    x = UpSampling2D((2, 2), name='D_upsamp_3')(x)
    # x = Dropout(0.1)(x)
    x = Conv2DTranspose(params['conv_1_units'], (params['kernel_1_size'], params['kernel_1_size']), strides=1, activation='relu', padding="same", name='D_conv_4')(x)
    #x = LeakyReLU(alpha=0.3)(x)

    decoded = Conv2D(1, (1, 1), activation='relu', padding="same", name='output', dtype=tf.float32)(x)
    #decoded = LeakyReLU(alpha=0.3)(x)
    autoencoder = Model(inputs=inputs, outputs=[decoded, encoded])
    
    autoencoder.compile(optimizer=optadam, loss=['mse', fn_smoothing], loss_weights=[0.95, 0.05])

    out = autoencoder.fit(X_train, X_train_c,
            validation_data=(X_valid, X_valid_c),
            epochs=params['epochs'], 
            use_multiprocessing=True,
            batch_size=params['batch_size'],
            verbose=0)
    
    return out, autoencoder

p = {'conv_1_units': list(np.arange(64, 129, 32)),
     'conv_2_units': list(np.arange(32, 97, 16)),
     'conv_3_units': list(np.arange(16, 32, 16)),
     'conv_4_units': list(np.arange(8, 17, 8)),
     'kernel_1_size': list(np.arange(13, 22, 2)),
     'kernel_2_size': list(np.arange(5, 17, 2)),
     'kernel_3_size': list(np.arange(3, 10, 2)),
     'kernel_4_size': list(np.arange(3, 8, 2)),
     'epochs':[args.epochs],
     'batch_size': [args.batch_size]}

scan_object = talos.Scan(x=X_train,
                         y=X_train_c,
                         x_val=X_valid,
                         y_val=X_valid_c,
                         params=p,
                         model=conv_small_tune,
                         experiment_name='Tuning_high_filters',
                         random_method='quantum',
                         reduction_method='correlation',
                         reduction_interval=10,
                         reduction_window=10,
                         reduction_threshold=0.4,
                         reduction_metric='val_loss',
                         minimize_loss=True)

np.save('scanning_results.npy', scan_object, allow_pickle=True)