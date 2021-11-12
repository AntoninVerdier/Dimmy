import os
import tensorflow.keras

class paths():
	def __init__(self):
		self.path2Data = '/home/user/Documents/Antonin/Code/Dimmy/Data'
		self.path2train = os.path.join(self.path2Data, 'nsynth-train/audio')
		self.path2valid = os.path.join(self.path2Data, 'nsynth-valid/audio')
		#self.path2test = os.path.join(self.path2Data, 'DeepenSounds/AutoencoderSounds_Longer/WavFilesLong')
		self.path2test = os.path.join(self.path2Data, 'nsynth-test/audio')

		self.path2soundsSJ = os.path.join(self.path2Data, 'nsynth-test/audio')
		self.path2Models = '/home/user/Documents/Antonin/Code/Dimmy/ModelSave'
		self.path2Output = '/home/user/Documents/Antonin/Code/Dimmy/OutputFig'

class params():
	def __init__(self):
		self.gen_params = {'dim': (513,126),
							'batch_size': 256,
							'shuffle': False}
		self.specshape = self.gen_params['dim']
		self.latent_size = 100
		self.epochs = 50

		self.test_params = {'dim': (513,126),
							'batch_size': 1,
							'shuffle': False}
