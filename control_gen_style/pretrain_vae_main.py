
__author__="zhitingh"
__version__="0.1"

import numpy as np
import tensorflow as tf
import os
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
flags.DEFINE_string('mode', 'full', '[pretrain_vae|pretrain_disc|full]')

# model
flags.DEFINE_integer('hidden_dim', 300, '')
flags.DEFINE_integer('emb_dim', 300, '')
flags.DEFINE_integer('disc_emb_dim', 300, '')
flags.DEFINE_integer('c_dim', 10, '')
flags.DEFINE_integer('z_dim', 290, '')
flags.DEFINE_string('disc_filter_sizes', '3,4,5', '')
flags.DEFINE_string('disc_filter_nums', '100,100,100', '')
flags.DEFINE_float("disc_l2_lambda", 0.1, "")

# data
flags.DEFINE_integer('seq_length', 16, '')
flags.DEFINE_integer('vocab_size', 16188, 'vocabulary size')
flags.DEFINE_integer('bos_token', 0, 'beginning of sequence')
flags.DEFINE_integer('eos_token', 16187, 'end of sequence, = vocab_size-1')
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
flags.DEFINE_integer('restore_start_step', -1, '')
flags.DEFINE_integer('display_every', -1, "..")
flags.DEFINE_integer('test_every', -1, "..")
flags.DEFINE_integer('sample_every', -1, "..")
flags.DEFINE_integer('checkpoint_every', -1, "..")
# pre-train
flags.DEFINE_integer('pt_nepochs', 10000, '')
flags.DEFINE_integer('pt_kld_anneal_start_epoch', 40, '')
flags.DEFINE_integer('pt_kld_anneal_end_epoch', 2000, '')
flags.DEFINE_integer('pt_restore_epoch', 0, "..")


FLAGS = flags.FLAGS

#os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu
from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())

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

    shutil.copy("./pretrain_vae.sh", '%s/%s/' % (FLAGS.output_path_prefix, run_id))

    return root_path, summary_path, checkpoint_path, sample_path


def transform_to_nbatches(n, num_batches):
    return n if n > 0 else -n*num_batches

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    assert FLAGS.eos_token==FLAGS.vocab_size-1
    # build output paths
    root_path, summary_path, checkpoint_path, sample_path = build_output_paths()
    FLAGS.summary_path = summary_path
    logging.basicConfig(filename=root_path+'/log.txt', level=logging.INFO)

    # load data
    if FLAGS.mode == 'pretrain_vae':
        data_loader = unlabeled_data_loader(FLAGS.data_path, FLAGS.batch_size)
    else:
        assert False, "Not Implemented"
    # load embedding
    word_embeddings = load_word_embddings(FLAGS.embedding_path)

    num_batches = data_loader.num_batches
    display_every = transform_to_nbatches(FLAGS.display_every, num_batches)
    sample_every = transform_to_nbatches(FLAGS.sample_every, num_batches)
    test_every = transform_to_nbatches(FLAGS.test_every, num_batches)
    checkpoint_every = transform_to_nbatches(FLAGS.checkpoint_every, num_batches)

    # build model
    model = Gen(FLAGS, word_embeddings)

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
        elif os.path.isfile(FLAGS.restore_vars_path):
            print('Restoring vars from {}'.format(FLAGS.restore_vars_path))
            logging.info('Restoring vars from {}'.format(FLAGS.restore_vars_path))
            model.load(sess, FLAGS.restore_vars_path)

        if FLAGS.mode == 'pretrain_vae':
            kld_w_annealer = linear_annealer( \
                FLAGS.pt_kld_anneal_start_epoch * data_loader.num_batches,
                FLAGS.pt_kld_anneal_end_epoch * data_loader.num_batches,
                0., 1.)
            #c = np.ones([FLAGS.batch_size, FLAGS.c_dim]) / FLAGS.c_dim
            #
            c = np.random.uniform(0,1,[FLAGS.batch_size, FLAGS.c_dim])
            c = c / np.sum(c,axis=1,keepdims=True)
            #
            #c_values = np.random.randint(0,FLAGS.c_dim,[FLAGS.batch_size])
            #c = np.array(one_hot_code(c_values, FLAGS.c_dim))
            for e in xrange(FLAGS.pt_restore_epoch, FLAGS.pt_nepochs):
                for b in xrange(data_loader.num_batches):
                    step = e * data_loader.num_batches + b
                    if step < FLAGS.restore_start_step:
                        continue
                    kld_w = kld_w_annealer.get_value(step)
                    #kld_w = 0.01
                    temp_o = 1.
                    x_batch = data_loader.next_batch()
                    feed = {model.x: x_batch, model.c: c, \
                            model.kld_w: kld_w, model.temp_g: 1., model.temp_o: temp_o}

                    display = step % display_every == 0
                    sample = step % sample_every == 0

                    model.pretrain_one_step(sess, step, feed, display,
                        sample, 5, sample_path)

                    if step % test_every == 0:
                        model.pretrain_evaluate(sess, data_loader, step)

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
                model.save(checkpoint_path, step)
            except:
                print("Checkpoint error ..")
                logging.info("Checkpoint error ..")


if __name__ == '__main__':
    tf.app.run()



