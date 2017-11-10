from __future__ import division
import os
import time
import math
from glob import glob
import numpy as np
from six.moves import xrange
import _pickle as cPickle
from utils import *

class unlabeled_data_loader(object):
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        self.load()


    def load(self):
        self.trX, self.teX = self.read(self.data_path)

        shuffle_indices = np.random.permutation(np.arange(len(self.trX)))
        self.trX = self.trX[shuffle_indices]

        self.trX_batches, self.num_batches = \
            create_batches(self.trX, batch_size=self.batch_size, pad=True)
        self.pointer = 0

        print("#train %d" % (len(self.trX)))
        print("#train batches %d" % self.num_batches)
        print("#test %d" % len(self.teX))


    def read(self, data_path):
        trX = []
        with open(os.path.join(data_path,'train.txt'), 'r') as f:
            for line in f:
                line = line.strip().split()
                x = [int(w) for w in line]
                trX.append(x)
        teX = []
        with open(os.path.join(data_path,'test.txt'), 'r') as f:
            for line in f:
                line = line.strip().split()
                x = [int(w) for w in line]
                teX.append(x)
        trX = np.array(trX)
        teX = np.array(teX)
        return trX, teX


    def train_batch_iter(self, num_epochs, pad=True):
        return batch_iter(self.batch_size, num_epochs, self.trX, pad=pad)

    def test_batch_iter(self, num_epochs, pad=True):
        return batch_iter(self.batch_size, num_epochs, self.teX, pad=pad)

    def next_batch(self):
        batch = self.trX_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batches
        return batch


class labeled_data_loader(object):
    def __init__(self, data_path, batch_size, y_dim):
        self.data_path = data_path
        self.batch_size = batch_size
        self.y_dim = y_dim
        self.load()

    def load(self):
        self.trX, self.trY, self.teX, self.teY = self.read(self.data_path)

        shuffle_indices = np.random.permutation(np.arange(len(self.trX)))
        self.trX = self.trX[shuffle_indices]
        self.trY = self.trY[shuffle_indices]

        self.trY = one_hot_code(self.trY, self.y_dim)
        self.trY = np.array(self.trY, dtype=np.int32)
        self.teY = one_hot_code(self.teY, self.y_dim)
        self.teY = np.array(self.teY, dtype=np.int32)

        self.trX_batches, self.trY_batches, self.num_batches = \
            create_batches(self.trX, self.trY, batch_size=self.batch_size, pad=True)
        self.pointer = 0

        print("#train %d" % (len(self.trX)))
        print("#train batches %d" % self.num_batches)
        print("#test %d" % len(self.teX))

    def read(self, data_path):
        trX, trY = [], []
        with open(os.path.join(data_path,'train.txt'), 'r') as f:
            for line in f:
                line = line.strip().split()
                x = [int(w) for w in line]
                trY.append(x[0])
                trX.append(x[1:])
        teX, teY = [], []
        with open(os.path.join(data_path,'test.txt'), 'r') as f:
            for line in f:
                line = line.strip().split()
                x = [int(w) for w in line]
                teY.append(x[0])
                teX.append(x[1:])
        trX = np.array(trX)
        trY = np.array(trY)
        teX = np.array(teX)
        teY = np.array(teY)
        return trX, trY, teX, teY


    def train_batch_iter(self, num_epochs, pad=True):
        return batch_iter(self.batch_size, num_epochs, self.trX, self.trY, pad=pad)

    def test_batch_iter(self, num_epochs, pad=True):
        return batch_iter(self.batch_size, num_epochs, self.teX, self.teY, pad=pad)

    def next_batch(self):
        x_batch = self.trX_batches[self.pointer]
        y_batch = self.trY_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batches
        return x_batch, y_batch


def batch_iter(batch_size, num_epochs, data_x, data_y=None, pad=False):
    """
    Generates a batch iterator for a dataset.
    """
    data_x = np.array(data_x)
    if data_y is not None:
        assert len(data_x) == len(data_y)
        data_y = np.array(data_y)
    data_size = len(data_x)
    num_batches_per_epoch = int((len(data_x) + batch_size - 1) / batch_size)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data_x = data_x[shuffle_indices]
        if data_y is not None:
            shuffled_data_y = data_y[shuffle_indices]
        if pad:
            extra_data_num = batch_size - data_size % batch_size if data_size % batch_size != 0 else 0
        else:
            extra_data_num = 0
        shuffled_data_x = np.append(shuffled_data_x, shuffled_data_x[:extra_data_num], axis=0)
        if data_y is not None:
            shuffled_data_y = np.append(shuffled_data_y, shuffled_data_y[:extra_data_num], axis=0)
        # Generate batches
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(shuffled_data_x))
            if data_y is not None:
                yield shuffled_data_x[start_index:end_index], shuffled_data_y[start_index:end_index]
            else:
                yield shuffled_data_x[start_index:end_index]

def create_batches(x, y=None, batch_size=256, pad=False):
    data_size = len(x)
    if pad==True:
        _x, _y = x, y
        extra_data_num = batch_size - len(x) % batch_size if data_size % batch_size != 0 else 0
        if extra_data_num > 0:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_x = x[shuffle_indices]
            _x = np.append(x, shuffled_x[:extra_data_num], axis=0)
            if y is not None:
                shuffled_y = y[shuffle_indices]
                _y = np.append(y, shuffled_y[:extra_data_num], axis=0)
        num_batches = int(len(_x) / batch_size)
    else:
        num_batches = int(len(x) / batch_size)
        _x = x[:num_batches*batch_size]
        if y is not None:
            _y = y[:num_batches*batch_size]
    x_batch_sequence = np.split(np.array(_x), num_batches, 0)
    if y is not None:
        y_batch_sequence = np.split(np.array(_y), num_batches, 0)
        return x_batch_sequence, y_batch_sequence, num_batches
    else:
        return x_batch_sequence, num_batches

def one_hot_code(y, y_dim):
    y_vec = np.zeros((len(y), y_dim))
    for i, label in enumerate(y_vec):
        y_vec[i,y[i]] = 1
    return y_vec

def load_word_embddings(path):
    _, _, word_embeddings, _, _ = cPickle.load(open(path, "rb"))
    print("vocab_size in word_embeddings %d" % len(word_embeddings))
    return word_embeddings # 2D matrix of shape vocab_size x emb_dim



