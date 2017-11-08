
import numpy as np
import logging

class linear_annealer(object):
    def __init__(self, st_x, end_x, st_y, end_y):
        self.st_x = st_x
        self.end_x = end_x
        self.st_y = st_y
        self.end_y = end_y
        self.y_anneal_per_step = (end_y - st_y) / (end_x - st_x)
        print("annealing per step: %.10f" % self.y_anneal_per_step)
        logging.info("annealing per step: %.10f" % self.y_anneal_per_step)

    def get_value(self, x):
        if x < self.st_x:
            return self.st_y
        elif x > self.end_x:
            return self.end_y
        else:
            return (x - self.st_x) * self.y_anneal_per_step + self.st_y

class sigmoid_annealer(object):
    def __init__(self, st_x, end_x, st_y, end_y, slope):
        self.st_x = st_x
        self.end_x = end_x
        self.st_y = st_y
        self.end_y = end_y
        self.slope = slope

    def get_value(self, x):
        if x < self.st_x:
            return self.st_y
        elif x > self.end_x:
            return self.end_y
        else:
            return 2. / (1. + np.exp(-self.slope*(x-self.st_x))) - 1 + self.st_y


def dump_samples(samples, fn):
    print("dump samples to %s" % fn)
    logging.info("dump samples to %s" % fn)
    np.savetxt(fn, samples, fmt='%d', delimiter=' ')
