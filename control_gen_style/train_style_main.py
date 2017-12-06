
__author__="zhitingh"
__version__="0.1"

import numpy as np
import tensorflow as tf
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
from utils.reader import *
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
flags.DEFINE_string("kld_anneal_method", "constant", "Method to anneal the kld term in the cost function")
# pre-train
flags.DEFINE_integer('pt_nepochs', 10000, '')
flags.DEFINE_integer('pt_kld_anneal_start_epoch', 40, '')
flags.DEFINE_integer('pt_kld_anneal_end_epoch', 2000, '')
flags.DEFINE_integer('pt_restore_epoch', 0, "..")

# Prior terms
flags.DEFINE_string('prior_distr', "normal", "")
flags.DEFINE_float('prior_mu', 0.0, '')
flags.DEFINE_float('prior_sigma', 1.0, '') # Really sigma^2


FLAGS = flags.FLAGS

#os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
from tensorflow.python.client import device_lib
print device_lib.list_local_devices()

np.set_printoptions(precision=3)
pp = pprint.PrettyPrinter()

def build_output_paths():
    run_id = "%s.%s" % (str(uuid.uuid4()), FLAGS.name) if FLAGS.name else str(uuid.uuid4())
    print("run_id %s" % run_id)
    root_path = '%s/%s' % (FLAGS.output_path_prefix, run_id)
    summary_path = '%s/%s/%s' % (FLAGS.output_path_prefix, run_id, "summary")
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    checkpoint_path = '%s/%s/%s/' % (FLAGS.output_path_prefix, run_id, "snapshots")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    sample_path = '%s/%s/%s/' % (FLAGS.output_path_prefix, run_id, "samples")
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    shutil.copy("./train_style.sh", '%s/%s/' % (FLAGS.output_path_prefix, run_id))

    return root_path, summary_path, checkpoint_path, sample_path


def transform_to_nbatches(n, num_batches):
    return n if n > 0 else -n*num_batches


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    assert FLAGS.eos_token==FLAGS.vocab_size-1
    assert FLAGS.mode == 'style'
    # build output paths
    root_path, summary_path, checkpoint_path, sample_path = build_output_paths()
    FLAGS.summary_path = summary_path
    logging.basicConfig(filename=root_path+'/log.txt', level=logging.INFO)

    # load data
    u_data_loader = unlabeled_data_loader(\
        FLAGS.data_path, FLAGS.batch_size)
    # load embedding
    word_embeddings = load_word_embddings(FLAGS.embedding_path)

    num_batches = FLAGS.nbatches
    display_every = transform_to_nbatches(FLAGS.display_every, num_batches)
    sample_every = transform_to_nbatches(FLAGS.sample_every, num_batches)
    test_every = transform_to_nbatches(FLAGS.test_every, num_batches)
    checkpoint_every = transform_to_nbatches(FLAGS.checkpoint_every, num_batches)

    # build model
    model = Gen(FLAGS, word_embeddings, FLAGS.prior_distr, FLAGS.prior_mu, FLAGS.prior_sigma)
    print("Prior distribution: ", FLAGS.prior_distr)
    print("Prior mu: ", FLAGS.prior_mu)
    print("Prior sigma: ", FLAGS.prior_sigma)
    logging.info("Prior distribution: {}".format(FLAGS.prior_distr))
    logging.info("Prior mu: {}".format(FLAGS.prior_mu))
    logging.info("Prior sigma: {}".format(FLAGS.prior_sigma))
    #
    step = 0
    with tf.Session() as sess:
        print("Start running ...")
        logging.info("Start running ...")

        model.build_optimizor(sess, FLAGS)

        sess.run([\
            tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver = tf.train.Saver(tf.global_variables())
        snapshot = tf.train.latest_checkpoint(FLAGS.restore_ckpt_path)
        if snapshot:
            print('Restoring model from {}'.format(snapshot))
            logging.info('Restoring model from {}'.format(snapshot))
            saver.restore(sess, snapshot)
        else:
            if os.path.isfile(FLAGS.restore_vars_path):
                print('Restoring vars from {}'.format(FLAGS.restore_vars_path))
                logging.info('Restoring vars from {}'.format(FLAGS.restore_vars_path))
                model.load(sess, FLAGS.restore_vars_path)
            if os.path.isfile(FLAGS.restore_disc_vars_path):
                print('Restoring disc vars from {}'.format(FLAGS.restore_disc_vars_path))
                logging.info('Restoring disc vars from {}'.format(FLAGS.restore_disc_vars_path))
                model.load(sess, FLAGS.restore_disc_vars_path)

        total_num_steps = FLAGS.nepochs * num_batches
        # arbitrary values
        c = np.ones([FLAGS.batch_size, FLAGS.c_dim]) / FLAGS.c_dim

        kld_w = 0.0

        for e in xrange(FLAGS.nepochs):
            print("------------Epoch {}".format(e))
            for b in xrange(num_batches):
                step = e * num_batches + b
                if step < FLAGS.restore_start_step:
                    continue

                #if e <= 10:
                #    temp_o = 1.
                #else:
                #    temp_o = np.maximum(1e-5, 0.7**((e-10)*5))
                #temp_o = np.maximum(1e-5, 0.7**(e*5))
                temp_o = 1. # TODO: anneal to 0.

                if FLAGS.kld_anneal_method == "constant":
                    kld_w = 1.0 # Always present 0.0909
                elif FLAGS.kld_anneal_method == "onoff":
                    kld_w = 0.0 if kld_w == 1.0 else 1.0
                elif FLAGS.kld_anneal_method == "oscillate":
                    curr_x_pos = e + (float(b) / num_batches) # x position on the graph
                    # 10 chosen as the frequency of the sin curve to get a nice oscillation frequency
                    kld_w = (1.0 / FLAGS.nepochs) * curr_x_pos + (1.0 / FLAGS.nepochs) * curr_x_pos * np.sin(10 * curr_x_pos)
                    if kld_w > 1.0: # Limit ceiling 
                        kld_w = 1.0 
                    print("curr x pos for oscillation: ", curr_x_pos)
                else:
                    raise Exception("Invalid kld anneal method")

                u_x_batch = u_data_loader.next_batch()
                feed = {model.x: u_x_batch, model.c: c, model.kld_w: kld_w, \
                        model.temp_o: temp_o, model.temp_g: 1., \
                        model.dropout_keep_prob: 1.}

                display = step % display_every == 0
                sample = step % sample_every == 0

                model.train_style_transfer_one_step(sess, step, feed, display,
                        sample, 5, sample_path, total_num_steps=total_num_steps)

                if step > 0 and step % checkpoint_every == 0:
                    try:
                        saver.save(sess, checkpoint_path, global_step=step)
                        print("snapshot to %s %d" % (checkpoint_path,step))
                        logging.info("snapshot to %s %d" % (checkpoint_path,step))
                        model.save(sess, checkpoint_path, step)
                    except:
                        print("Checkpoint error ..")
                        logging.info("Checkpoint error ..")

        try:
            saver.save(sess, checkpoint_path, global_step=step)
            print("snapshot to %s %d" % (checkpoint_path,step))
            logging.info("snapshot to %s %d" % (checkpoint_path,step))
            model.save(sess, checkpoint_path, step)
        except:
            print("Checkpoint error ..")
            logging.info("Checkpoint error ..")


if __name__ == '__main__':
    tf.app.run()



