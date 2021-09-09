import os
import tensorflow.keras as keras
import numpy as np
import pickle as pkl

import settings as s 

paths = s.paths()

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_IDs, dim=(64000,), batch_size=256, shuffle=True, path=None, test=False, phase=False, dataset=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.path = path
        self.test = test
        self.phase = phase
        if dataset:
            self.dataset = pkl.load(open(dataset, 'rb'))

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        print(self.batch_size)
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        if self.dataset:
            X = self.__data_generation_dict(list_IDs_temp)
        else:
            X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Initialization
        X = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            ID = ID[:-4] + '.npy'
            # Store sample
            if self.phase:
                X[i,] = np.load('Data/phases/' + ID)
            else:
                X[i,] = np.load('Data/mags/' + ID)
        if self.test:
            return X
        else:
            return X, X

    def __data_generation_dict(self, list_IDs_temp):
    # Initialization
        X = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            ID = ID[:-4]
            # Store sample
            X[i, :, :] = self.dataset[ID]
        if self.test:
            return X
        else:
            return X, X


class DataGenerator_both(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_IDs, dim=(64000,), batch_size=256, shuffle=True, path=None, test=False, phase=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.path = path
        self.test = test
        self.phase = phase
        self.on_epoch_end()

    def __len__(self):
        print(self.batch_size)
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Initialization
        p = np.empty((self.batch_size, *self.dim))
        m = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            ID = ID[:-4] + '.npy'
            # Store sample
            p[i,] = np.load('Data/phases/' + ID)
            m[i,] = np.load('Data/mags/' + ID)

        X = np.stack((m, p), axis=3)

        return X, X


def get_generators(dim, batch_size, shuffle=True, channels=False, test=False, dataset=None):

    partition = {}
    partition['train'] = os.listdir(paths.path2train)
    partition['validation'] = os.listdir(paths.path2valid)
    partition['test'] = os.listdir(paths.path2test)

    if not test:
        training_generator = DataGenerator(partition['train'], dim, batch_size, shuffle, dataset=dataset)
        validation_generator = DataGenerator(partition['validation'], dim, batch_size, shuffle)
        test_generator = DataGenerator(partition['test'], dim, batch_size, shuffle)

        if channels:
                training_generator = DataGenerator_both(partition['train'], dim, batch_size, shuffle)
                validation_generator = DataGenerator_both(partition['validation'], dim, batch_size, shuffle)
                test_generator = DataGenerator_both(partition['test'], dim, batch_size, shuffle)

        if not os.path.exists('Output/generator/'):
            os.makedirs('Output/generator/')

        pkl.dump(training_generator.indexes, open('Output/generator/train_indexes.pkl', 'wb'))
        pkl.dump(validation_generator.indexes, open('Output/generator/valid_indexes.pkl', 'wb'))
        pkl.dump(test_generator.indexes, open('Output/generator/test_indexes.pkl', 'wb'))

        return training_generator, validation_generator, test_generator
    else:
        test_generator = DataGenerator(partition['test'], dim, batch_size, shuffle, test=True)
        phases = DataGenerator(partition['test'], dim, batch_size, shuffle, test=True, phase=True)

        return test_generator, phases





