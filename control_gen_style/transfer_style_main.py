import tensorflow as tf
from utils.reader import *

import numpy as np
import random
import time
import cPickle
import os
import sys
import uuid
import logging
import shutil
import pprint

from model import Gen
from utils.utils import *

flags = tf.app.flags

# generic options
flags.DEFINE_integer('random_seed', 88, 'Value of random seed [88]')
flags.DEFINE_string('name', 'ctrl_gen', "Name of the instance [ctrl_gen]")
#flags.DEFINE_string('gpu', '0', '')
flags.DEFINE_string('mode', 'full', '[pretrain_vae|pretrain_disc|full|style|test]')

# model
flags.DEFINE_integer('hidden_dim', 300, '')
flags.DEFINE_integer('emb_dim', 300, '')
flags.DEFINE_integer('disc_emb_dim', 300, '')
flags.DEFINE_integer('c_dim', 2, '')
flags.DEFINE_integer('z_dim', 290, '')
flags.DEFINE_string('disc_filter_sizes', '3,4,5', '')
flags.DEFINE_string('disc_filter_nums', '100,100,100', '')
flags.DEFINE_float("disc_l2_lambda", 0.1, "")
flags.DEFINE_float("dropout_keep_prob", 0.75, "")

# data
flags.DEFINE_integer('seq_length', 16, '')
flags.DEFINE_integer('vocab_size', 16188, 'vocabulary size')
flags.DEFINE_integer('bos_token', 0, 'beginning of sequence')
flags.DEFINE_integer('eos_token', 16187, 'end of sequence, = vocab_size-1')
#flags.DEFINE_string('train_data', './data', '')
#flags.DEFINE_string('test_data', './data', '')
flags.DEFINE_string('data_path', './data', '')
flags.DEFINE_string('embedding_path', './w2v.p', '')

# training
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_float("u_gen_w", 0.1, "")
flags.DEFINE_float("u_disc_w", 0.1, "")
flags.DEFINE_float("ind_gen_w", 0.1, "")
flags.DEFINE_float("bow_w", 0.1, "")
flags.DEFINE_float("style_w", 0.1, "")
flags.DEFINE_float("recon_dropout_keep_prob", 0.75, "")
flags.DEFINE_float("disc_lr", 0.001, "learning rate of discriminator training [0.001]")
flags.DEFINE_string('output_path_prefix', './outputs', "")
flags.DEFINE_string('restore_ckpt_path', '', "..")
flags.DEFINE_string('restore_vars_path', '', "..")
flags.DEFINE_string('restore_disc_vars_path', '', "..")
flags.DEFINE_integer('restore_start_step', -1, '')
flags.DEFINE_integer('display_every', -1, "..")
flags.DEFINE_integer('test_every', -1, "..")
flags.DEFINE_integer('sample_every', -1, "..")
flags.DEFINE_integer('checkpoint_every', -1, "..")
# full-train
flags.DEFINE_integer('nepochs', 500, '')
flags.DEFINE_integer('nbatches', 300, '')
flags.DEFINE_float("kld_w", 0.01, "")
# pre-train
flags.DEFINE_integer('pt_nepochs', 10000, '')
flags.DEFINE_integer('pt_kld_anneal_start_epoch', 40, '')
flags.DEFINE_integer('pt_kld_anneal_end_epoch', 2000, '')
flags.DEFINE_integer('pt_restore_epoch', 0, "..")
FLAGS = flags.FLAGS

from tensorflow.python.client import device_lib


word_idx_map_pkl_file = 'data/imdb.data.binary.p.0.01.l16'
x = cPickle.load(open(word_idx_map_pkl_file,"rb"))
_, _, _, word_idx_map, _ = x[0], x[1], x[2], x[3], x[4]

mdl = Gen(FLAGS, load_word_embddings(FLAGS.embedding_path))
with tf.Session() as sess:
  mdl.load(sess, FLAGS.restore_vars_path)
  mdl.load(sess, FLAGS.restore_disc_vars_path)

  #c = np.ones([FLAGS.batch_size, FLAGS.c_dim]) / FLAGS.c_dim
  #if step < FLAGS.restore_start_step:
   #   continue
  #temp_o = 1. # TODO: anneal to 0.

  #kld_w = FLAGS.kld_w
  #u_x_batch = u_data_loader.next_batch()

  feed = {mdl.x : tf.Variable([word_idx_map[word] for word in 'hello this is'.split(' ')])}
  #feed = {model.x: u, model.c: c, model.kld_w: kld_w, \
  #        model.temp_o: temp_o, model.temp_g: 1., \
  #        model.dropout_keep_prob: 1.}

  mdl.train_style_transfer_one_step(sess, step=1, feed=feed, display=True, 
    sample=True, sample_nbatches=5, sample_path='test.txt')
  