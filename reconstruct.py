import os
from textwrap import dedent
import numpy as np 
import preproc as proc

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

import matplotlib.pyplot as plt

import scipy.io as sio 

def fn_smoothing(y_true, y_pred):

    pred_freq_corr = autoencoder(self.test_freq)[1]

    def t(a): return tf.transpose(a)

    x = pred_freq_corr
    mean_t = tf.reduce_mean(x, axis=1, keepdims=True)
    #cov_t = x @ t(x)
    cov_t = ((x-mean_t) @ t(x-mean_t))/(pred_freq_corr.shape[1]-1)
    cov2_t = tf.linalg.diag(1/tf.sqrt(tf.linalg.diag_part(cov_t)))
    cor = cov2_t @ cov_t @ cov2_t

    # Fin how to compute autocorrelation matrix
    loss = tf.keras.losses.mean_squared_error(self.true_freq_corr, cor)
    # May need to return output with batch size 

    return loss

model = '/home/user/share/gaia/User_folders/Antonin/Models/Continous_models/31082022_180020_full_tinynet_28k_all/Autoencoder_model_conv_small_full_tinynet_28k_all'
sound = '/home/user/share/gaia/User_folders/Antonin/Models/Continous_models/SoundsHearlight/6kto16k_8step_70dB.wav'

one_encoding_test = np.diag(np.full(100, 0.5))


autoencoder = load_model(model, custom_objects={"fn_smoothing": fn_smoothing})
print(autoencoder.summary())
#decoder = Model(inputs=autoencoder.get_layer('reshape_1').output, outputs=autoencoder.get_layer('output').output)

input_shape = autoencoder.layers[11].get_input_shape_at(0)
layer_input = Input(shape=(100,))

x = layer_input
for layer in autoencoder.layers[11:]:
    x=layer(x)

decoder = Model(layer_input, x)

for i, s in enumerate(one_encoding_test):
    out = decoder(s.reshape(1, 100))[0].numpy()
    plt.imshow(out.reshape(128, 112), cmap='inferno')
    plt.savefig('Figures/half_led_{}.svg'.format(i))
    plt.close()


# print(autoencoder.summary())


# x = np.arange(1, 129)
# y = 0.5*np.exp(-0.022*x)
# y = 1/(np.repeat(y, 126).reshape(128, 126))

# # Load soundfile and compute spectrogram

# test, scale = proc.load_unique_file_cqt(sound, y, reconstruct=True)

# X_test = np.expand_dims(test, 0)
# X_test = X_test.astype('float32')/255.0
# X_test = X_test[:, :, :112]
# X_test = np.expand_dims(X_test, 3)

# decoded = autoencoder.predict(X_test)[0]

# decoded = decoded.reshape(128, 112)
# decoded = np.pad(decoded, ((0, 0), (0, 14)), 'constant', constant_values=0)

# print(np.max(decoded), np.min(decoded))

# decoded = decoded * (scale[1] - scale[0]) + scale[0]
# decoded = np.multiply(decoded, 1/y)
# ttrue = proc.inverse_cqt(decoded)

# X_test = X_test.reshape(128, 112)
# X_test = np.pad(X_test, ((0, 0), (0, 14)), 'constant', constant_values=0)

# print(np.max(X_test), np.min(X_test))

# X_test = X_test * (scale[1] - scale[0]) + scale[0]
# X_test = np.multiply(X_test, 1/y)
# treprod = proc.inverse_cqt(X_test)


# sio.wavfile.write('test_reprode.wav', 64000, ttrue)
# sio.wavfile.write('reconstruction.wav', 64000, treprod)

