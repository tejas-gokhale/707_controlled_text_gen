
__author__="zhiting"

import numpy as np
import _pickle as cPickle
from collections import defaultdict
import sys, re
#import pandas as pd


def parse_sentences(data_file, idx_word_map):
    with open(data_file+".parsed", "w") as fout:
        with open(data_file, "rb") as f:
            for line in f:
                line = line.strip()
                word_inds = line.split()
                for i in xrange(len(word_inds)):
                    idx = int(word_inds[i])
                    if idx not in idx_word_map:
                        fout.write("\n")
                        break
                    elif i == len(word_inds) - 1:
                        fout.write(idx_word_map[idx]+"\n")
                    else:
                        fout.write(idx_word_map[idx]+" ")
    fout.close()


def get_reverse_word_idx_map(word_idx_map):
    idx_word_map = {}
    for w,i in word_idx_map.iteritems():
        idx_word_map[i] = w
    return idx_word_map


if __name__=="__main__":
    data_file = sys.argv[1]
    word_idx_map_pkl_file = '../../data_imdb/imdb.data.binary.p.0.01.l16'

    x = cPickle.load(open(word_idx_map_pkl_file,"rb"))
    _, _, _, word_idx_map, _ = x[0], x[1], x[2], x[3], x[4]

    idx_word_map = get_reverse_word_idx_map(word_idx_map)

    parse_sentences(data_file, idx_word_map)

    print ("Done.")

