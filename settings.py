import os

class paths():
	def __init__(self):
		self.path2Data = '/home/anverdie/Documents/Code/Dimmy/Data/'
		self.path2train = os.path.join(self.path2Data, 'nsynth-train/audio')
		self.path2valid = os.path.join(self.path2Data, 'nsynth-valid/audio')
		self.path2test = os.path.join(self.path2Data, 'nsynth-test/audio')

class params():
	def __init__(self):
		self.gen_params = {'dim': (513,126),
							'batch_size': 1024,
							'shuffle': True}
		self.specshape = self.gen_params['dim']
		self.latent_size = 40
		self.epochs = 120