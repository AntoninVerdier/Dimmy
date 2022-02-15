import os
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

import preproc as p
import settings as s 

paths = s.paths()

rootdir = '/home/anverdie/Documents/Code/interfaceDalleLed/Stims'


for f in os.listdir(rootdir):
	if not os.path.isdir(os.path.join(rootdir, f)):
		l = np.load(os.path.join(rootdir, f)).reshape(10, 10)
		print(l.shape)
		l = np.repeat(l, 30, axis=0)
		l = np.repeat(l, 30, axis=1)
		pad = np.zeros((90, 300))
		l = np.concatenate((pad, l, pad)).T

		if not os.path.exists(os.path.join(paths.path2OutputD, 'DLP')):
			os.makedirs(os.path.join(paths.path2OutputD, 'DLP'))

		np.save(os.path.join(paths.path2OutputD, 'DLP', '{}_dlp.npy'.format(f[:-4])), l)
		matplotlib.image.imsave(os.path.join(paths.path2OutputD, 'DLP', '{}_dlp.bmp'.format(f[:-4])), 1-l, cmap='Greys', vmin=0, vmax=1)