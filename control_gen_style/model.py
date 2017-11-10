
__author__="zhitingh"

import cPickle
import numpy as np
import tensorflow as tf
#from tensorflow.python.ops import tensor_array_ops, control_flow_ops
#import tf.contrib.rnn as rnn
from tensorflow.contrib.rnn import BasicLSTMCell as lstm_cell
from tensorflow.contrib.rnn import LSTMStateTuple as lstm_state

from utils.utils import *

class Gen(object):
    def __init__(self, config, embeddings, prior_mu, prior_sigma):
        self.hidden_dim = config.hidden_dim
        self.z_dim = config.z_dim
        self.c_dim = config.c_dim
        self.emb_dim = config.emb_dim
        self.seq_length = config.seq_length
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.bow_w = config.bow_w

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        self.disc_emb_dim = config.disc_emb_dim
        self.disc_filter_sizes = \
            [int(sz) for sz in config.disc_filter_sizes.strip().split(',')]
        self.disc_filter_nums = \
            [int(num) for num in config.disc_filter_nums.strip().split(',')]
        self.disc_l2_lambda = config.disc_l2_lambda

        self.grad_clip = 5.0
        self.disc_lr = config.disc_lr
        self.u_gen_w = config.u_gen_w
        self.u_disc_w = config.u_disc_w
        self.ind_gen_w = config.ind_gen_w
        self.style_w = config.style_w

        assert config.vocab_size == config.eos_token+1
        # beginning of sequence
        self.bos_token = tf.constant( \
            [config.bos_token] * self.batch_size, dtype=tf.int32)
        # end of sequence
        self.eos_token = tf.constant( \
            [config.eos_token] * self.batch_size, dtype=tf.int32)
        self.eos_token_single = config.eos_token
        # assign probability 1 to the eos token (= vocab_size - 1)
        eos_prob = tf.concat([\
            tf.zeros([self.batch_size, self.vocab_size-1]) + 1e-10,
            tf.ones([self.batch_size, 1])], 1)
        self.eos_prob_T = tf.transpose(eos_prob)
        self.eos_log_prob_T = tf.log(self.eos_prob_T)
        ## <unk> token
        #self.unk_token_single = config.unk_token

        #
        self.recon_dropout_keep_prob = config.recon_dropout_keep_prob


        self.build_model(embeddings)


    def build_model(self, embeddings):
        with tf.variable_scope('enc'):
            self.enc_rnn_unit = self.create_rnn_unit('enc_rnn')
            self.enc_output_layer = self.create_enc_output_layer()

        with tf.variable_scope('gen'):
            # shared with encoder
            self.gen_emb = tf.Variable(embeddings, dtype=tf.float32, name='gen_enc_emb')
            self.gen_rnn_unit = self.create_rnn_unit('gen_rnn')
            self.gen_output_unit = self.create_gen_output_layer()
            self.gen_init_state_unit = self.create_gen_init_state_layer()

        with tf.variable_scope('disc'):
            self.disc_emb = tf.Variable(embeddings, dtype=tf.float32,
                                        name='disc_emb')

        def _x_rep(x):
            # one_hot coding
            x_oh = tf.one_hot(x, self.vocab_size, 1.0, 0.0)
            # token sequence
            x_ind = tf.TensorArray(
                tf.int32, size=self.seq_length, clear_after_read=False)
            x_ind = x_ind.unstack(tf.transpose(x, perm=[1, 0]))
            # embedding sequence
            with tf.device("/cpu:0"):
                inputs = tf.split(
                    tf.nn.embedding_lookup(self.gen_emb, x), self.seq_length, 1)
                processed_x = tf.stack( # seq_length x batch_size x emb_dim
                    [tf.squeeze(input_, [1]) for input_ in inputs])
            x_emb = tf.TensorArray(
                dtype=tf.float32, size=self.seq_length, clear_after_read=False)
            x_emb = x_emb.unstack(processed_x)
            return x_oh, x_ind, x_emb


        # input sentences
        self.x = tf.placeholder(
            tf.int32, shape=[self.batch_size, self.seq_length], name="x")
        self.x_oh, self.x_ind, self.x_emb = _x_rep(self.x)
        # input attributes (for discriminator training and pre-training)
        self.c = tf.placeholder(tf.float32, shape=[self.batch_size, self.c_dim])
        self

        # labeled input
        self.s_x = tf.placeholder(
            tf.int32, shape=[self.batch_size, self.seq_length], name="s_x")
        self.s_x_oh, self.s_x_ind, self.s_x_emb = _x_rep(self.s_x)
        self.s_c = tf.placeholder(tf.float32, shape=[self.batch_size, self.c_dim])

        # dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # temperatures
        self.temp_o = tf.placeholder(tf.float32, name="temp_o")
        self.temp_g = tf.placeholder(tf.float32, name="temp_g")
        # kld weight
        self.kld_w = tf.placeholder(tf.float32, name="kld_weight")

        # encode
        x_mu, x_log_sigma = self.encode(self.x_emb)
        x_z = self.enc_sampler(x_mu, x_log_sigma)

        self.build_pretrain_loss(x_z, x_mu, x_log_sigma, self.c)
        self.build_unsupervised_loss()
        self.build_independence_constraint()
        self.build_gen_loss()
        self.build_disc_loss()

        self.build_style_transfer_loss(x_mu)

        self.t_vars = tf.trainable_variables()
        self.enc_vars = [var for var in self.t_vars if 'enc_' in var.name]
        self.gen_vars = [var for var in self.t_vars if 'gen_' in var.name]
        self.disc_vars = [var for var in self.t_vars if 'disc_' in var.name]
        self.vae_nongen_vars = [var for var in self.t_vars if \
            ('enc_' in var.name or 'vae_' in var.name) and ('gen_' not in var.name)]
        self.vae_vars = self.vae_nongen_vars + self.gen_vars
        print("#t_vars %d, #enc_vars %d. #gen_vars %d, #disc_vars %d, #vae_vars %d" % (\
            len(self.t_vars), len(self.enc_vars), len(self.gen_vars),
            len(self.disc_vars), len(self.vae_vars)))


    def build_pretrain_loss(self, x_z, x_mu, x_log_sigma, c):
        x_h_pt = tf.concat([x_z, c], 1)
        self.recon_loss, self.kld, gen_p, self.recon_loss_word = self.build_reconstruct_loss( \
            x_h_pt, x_mu, x_log_sigma, self.x, self.x_ind, self.x_oh, self.x_emb)

        self.bow_loss = self.build_pretrain_bow_loss(x_h_pt)

        self.pt_loss = self.recon_loss + self.kld_w * self.kld + self.bow_w * self.bow_loss

        self.test_pt_loss = tf.placeholder(tf.float32, name="test_pt_loss")
        self.test_recon_loss = tf.placeholder(tf.float32, name="test_recon_loss")
        self.test_kld = tf.placeholder(tf.float32, name="test_kld")
        self.test_bow_loss = tf.placeholder(tf.float32, name="test_bow_loss")


    def build_supervised_pretrain_loss(self, x_z, x_mu, x_log_sigma, c):
        x_h_pt = tf.concat([x_z, c], 1)
        self.s_recon_loss, self.s_kld, gen_p, self.s_recon_loss_word = self.build_reconstruct_loss( \
            x_h_pt, x_mu, x_log_sigma, self.s_x, self.s_x_ind, self.s_x_oh, self.s_x_emb)

        self.s_bow_loss = self.build_pretrain_bow_loss(x_h_pt, reuse=True)

        self.s_pt_loss = self.s_recon_loss + self.kld_w * self.s_kld + self.bow_w * self.s_bow_loss


    def build_reconstruct_loss(self, h, x_mu, x_log_sigma, x, x_ind, x_oh, x_emb, reuse=False):
        s = self.gen_init_state_unit(h)

        # batch_size x seq_length x vocab_size
        _, prob, x_length = self.reconstruct( \
            x, x_ind, x_emb, s, dropout_keep_prob=self.recon_dropout_keep_prob, reuse=reuse)

        recon_loss = - tf.reduce_sum(
            x_oh * tf.log(tf.clip_by_value(prob, 1e-20, 1.0)),
            axis=[1,2]) / x_length
        recon_loss = tf.reduce_mean(recon_loss)


        kld = -0.5 * tf.reduce_sum(1.0 - (x_log_sigma - tf.log(self.prior_sigma)) - (self.prior_sigma + (self.prior_mu - x_mu)**2) / tf.exp(x_log_sigma))
        # kld = - tf.reduce_sum(1. + x_log_sigma - x_mu**2 - tf.exp(x_log_sigma), axis=1) / 2. # original std normal KL
        kld = tf.reduce_mean(kld)

        recon_loss_word = - tf.reduce_sum(
            x_oh * tf.log(tf.clip_by_value(prob, 1e-20, 1.0)),
            axis=[1,2]) / x_length
        recon_loss_word = tf.reduce_mean(recon_loss_word)

        return recon_loss, kld, prob, recon_loss_word


    def build_pretrain_bow_loss(self, x_s, reuse=False):
        with tf.variable_scope("vae_mlp") as scope:
            if reuse:
                scope.reuse_variables()
            h1 = tf.contrib.layers.fully_connected(x_s, 400, activation_fn=tf.nn.tanh)
            o = tf.contrib.layers.fully_connected(h1, self.vocab_size, activation_fn=None)

            bow_loss = 0
            for t in xrange(self.seq_length):
                if t > 0:
                    eos = tf.not_equal(self.x_ind.read(t-1), self.eos_token)
                else:
                    eos = tf.ones([self.batch_size])
                eos = tf.to_float(eos)
                o_t = tf.transpose(tf.multiply(eos, tf.transpose(o)) + \
                    tf.multiply(tf.subtract(1., eos), self.eos_log_prob_T))
                bow_loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(\
                    logits=o_t, labels=self.x_ind.read(t)))

            bow_loss /= (self.batch_size * self.seq_length)
            return bow_loss


    def build_style_transfer_loss(self, x_z):
        loss, accu = 0., 0.
        ind_loss = 0.
        recon_loss = 0.
        for c_value in xrange(self.c_dim):
            c_values = np.array([c_value] * self.batch_size, dtype=np.int32)
            h = tf.concat([x_z, self.one_label_tensor(c_values)], 1)
            s = self.gen_init_state_unit(tf.stop_gradient(h))
            _, gen_p = self.generate(\
                s, temp_o=self.temp_o, temp_g=self.temp_g)

            _, _, loss_c, accu_c = \
                self.discriminator(gen_p, h[:,-self.c_dim:], reuse=True)

            loss += loss_c
            accu += accu_c

            ## ind loss
            p_mu, p_log_sigma = self.encode_x(gen_p, reuse=True)
            ind_loss_c = tf.nn.l2_loss(x_z-p_mu) / self.batch_size

            ind_loss += ind_loss_c

            # recon loss
            recon_loss_c, _, _, _ = self.build_reconstruct_loss(\
                h, x_z, 0., self.x, self.x_ind, self.x_oh, self.x_emb, reuse=True)

            recon_loss += recon_loss_c

        loss /= self.c_dim
        self.style_accu = accu / self.c_dim
        self.style_ind_loss = ind_loss / self.c_dim
        self.style_recon_loss = recon_loss / self.c_dim

        self.style_loss = self.style_w * loss + \
            self.ind_gen_w * self.style_ind_loss + self.pt_loss + \
            0.2 * self.style_recon_loss


    def build_unsupervised_loss(self):
        h = self.generate_hidden_code()
        s = self.gen_init_state_unit(h)
        _, gen_p = self.generate(\
            s, temp_o=self.temp_o, temp_g=self.temp_g)

        _, _, self.u_loss, self.u_accu = \
            self.discriminator(gen_p, h[:,-self.c_dim:])


    def build_independence_constraint(self):
        h = self.generate_hidden_code()

        # generate
        s = self.gen_init_state_unit(h)
        _, gen_p = self.generate(s, temp_o=self.temp_o, temp_g=self.temp_g)
        # encode
        p_emb = tf.reshape( \
            tf.matmul(tf.reshape(gen_p, [-1, self.vocab_size]), self.gen_emb),
            [-1, self.seq_length, self.emb_dim])
        p_emb = tf.transpose(p_emb, [1, 0, 2])
        p_emb_ta = tf.TensorArray(dtype=tf.float32, size=self.seq_length)
        p_emb_ta = p_emb_ta.unstack(p_emb)
        p_mu, p_log_sigma = self.encode(p_emb_ta, reuse=True)

        p_z = self.enc_sampler(p_mu, p_log_sigma)
        self.ind_loss = tf.nn.l2_loss(h[:,:-self.c_dim]-p_z) / self.batch_size


    def build_gen_loss(self):
        self.gen_loss = self.u_gen_w * self.u_loss + self.pt_loss
        self.gen_ind_loss = self.ind_gen_w * self.ind_loss


    def build_disc_loss(self):
        # supervised loss
        self.s_prob, _, self.s_loss, self.s_accu  = \
            self.discriminator(self.x_oh, self.c, reuse=True)
        # discriminator pre-train
        self.pt_disc_loss = self.s_loss
        # full training
        self.disc_loss = self.u_disc_w * self.u_loss + self.s_loss

        self.test_accu = tf.placeholder(tf.float32, name="test_accu")

    def sample_x(self, sample_nbatches):
        def sample_x_batch(c_values=None):
            h = self.generate_hidden_code(c_values)
            #s = lstm_state(h, h)
            s = self.gen_init_state_unit(h)
            gen_x, _ = self.generate(s, temp_g=self.temp_g)
            return gen_x

        samples = []
        for c_value in xrange(self.c_dim):
            c_values = np.array([c_value] * self.batch_size, dtype=np.int32)
            for b in xrange(sample_nbatches):
                samples_b = sample_x_batch(c_values)
                samples_b = tf.concat(\
                    [np.expand_dims(c_values, 1), samples_b], axis=1)
                samples.append(samples_b)
        samples = tf.concat(samples, axis=0)
        return samples

    def sample_x_stitch(self, sample_nbatches):
        temp_g = 0.01
        samples = []
        [samples.append([]) for i in xrange(self.c_dim)]
        for b in xrange(sample_nbatches):
            _h = self.generate_hidden_code()
            for c_value in xrange(self.c_dim):
                c_values = np.array([c_value] * self.batch_size, dtype=np.int32)
                h = tf.concat([_h[:,:-self.c_dim], self.one_label_tensor(c_values)], 1)
                #s = lstm_state(h, h)
                s = self.gen_init_state_unit(h)
                samples_b, _ = self.generate(s, temp_g=temp_g)
                samples_b = tf.concat(\
                    [np.expand_dims(c_values, 1), samples_b], axis=1)
                samples[c_value].append(samples_b)
        samples = [tf.concat(samples[i], 0) for i in xrange(self.c_dim)]
        samples = tf.reshape(tf.concat(samples, 1), [-1, self.seq_length+1])
        return samples

    def sample_x_recon(self):
        # encode
        x_mu, x_log_sigma = self.encode(self.x_emb, reuse=True)

        temp_g = 0.01

        # recon based on mean
        x_mu_h_pt = tf.concat([x_mu, self.c], 1)
        x_mu_s_pt = self.gen_init_state_unit(x_mu_h_pt)
        recon_x_mu, _, _ = self.reconstruct( \
            self.x, self.x_ind, self.x_emb, x_mu_s_pt, temp_g=temp_g, reuse=True)
        # gen based on mean
        gen_x_mu, _ = self.generate(x_mu_s_pt, temp_g=temp_g)

        # recon
        x_z = self.enc_sampler(x_mu, x_log_sigma)
        x_h_pt = tf.concat([x_z, self.c], 1)
        x_s_pt = self.gen_init_state_unit(x_h_pt)
        # gen
        gen_x, _ = self.generate(x_s_pt, temp_g=temp_g)

        # edit
        x_mu_h_pt = tf.concat([x_mu, 1.-self.c], 1)
        x_mu_s_pt = self.gen_init_state_unit(x_mu_h_pt)
        gen_x_mu_rev, _ = self.generate(x_mu_s_pt, temp_g=temp_g)

        # combined
        samples = tf.concat([self.x, recon_x_mu, gen_x_mu, gen_x, gen_x_mu_rev], 1)
        samples = tf.reshape(samples, [-1, self.seq_length])
        return samples

    def sample_x_style(self):
        # encode
        x_mu, x_log_sigma = self.encode(self.x_emb, reuse=True)

        temp_g = 0.01

        ## gen based on mean
        #x_mu_h_pt = tf.concat([x_mu, self.c], 1)
        #x_mu_s_pt = self.gen_init_state_unit(x_mu_h_pt)
        #gen_x_mu, _ = self.generate(x_mu_s_pt, temp_g=temp_g)

        ## edit
        #x_mu_h_pt = tf.concat([x_mu, 1.-self.c], 1)
        #x_mu_s_pt = self.gen_init_state_unit(x_mu_h_pt)
        #gen_x_mu_rev, _ = self.generate(x_mu_s_pt, temp_g=temp_g)

        # neg
        c = tf.concat([tf.ones([self.batch_size, 1]), tf.zeros([self.batch_size, 1])], 1)
        x_mu_h_pt = tf.concat([x_mu, c], 1)
        x_mu_s_pt = self.gen_init_state_unit(x_mu_h_pt)
        gen_x_mu_neg, _ = self.generate(x_mu_s_pt, temp_g=temp_g)

        # pos
        c = tf.concat([tf.zeros([self.batch_size, 1]), tf.ones([self.batch_size, 1])], 1)
        x_mu_h_pt = tf.concat([x_mu, c], 1)
        x_mu_s_pt = self.gen_init_state_unit(x_mu_h_pt)
        gen_x_mu_pos, _ = self.generate(x_mu_s_pt, temp_g=temp_g)

        # combined
        samples = tf.concat([self.x, gen_x_mu_neg, gen_x_mu_pos], 1)
        samples = tf.reshape(samples, [-1, self.seq_length])
        return samples


    def generate_hidden_code(self, c_values=None):
        z = tf.random_normal([self.batch_size, self.z_dim], mean=self.prior_mu, stddev=self.prior_sigma)
        if c_values is None:
            prob = tf.ones([self.batch_size, self.c_dim]) / self.c_dim
            c_values = tf.squeeze(tf.multinomial(tf.log(prob), 1)) # self.batch_size
        c = self.one_label_tensor(c_values)
        h = tf.concat([z, c], 1)
        return h


    def one_label_tensor(self, c):
        """
        transform label c_value into one-hot coding
        input: c_value - integer
        output: c_oh - tensor of shape [self.batch_size, self.c_dim]
        adapted from `https://github.com/saemundsson/semisupervised_vae`
    """
        indices = []
        values = []
        for i in xrange(self.batch_size):
        	indices += [[i, c[i]]]
        	values += [1.]
        c_oh = tf.sparse_tensor_to_dense(tf.SparseTensor(\
            indices=indices, values=values, dense_shape=[self.batch_size, self.c_dim]), 0.0)
        return c_oh

    def reconstruct(self, x, x_ind, x_emb, init_state, temp_g=1., dropout_keep_prob=1., reuse=False):
        prob = tf.TensorArray(tf.float32, size=self.seq_length)
        recon_x = tf.TensorArray(tf.int32, size=self.seq_length)
        # actual length of x
        x_length = tf.reduce_sum(tf.to_float( \
            tf.not_equal(self.x, self.eos_token_single)), 1) + 1
        h_0 = init_state.h
        s_t = init_state
        x_t = tf.nn.embedding_lookup(self.gen_emb, self.bos_token)
        with tf.variable_scope("gen_rnn") as scope:
            if reuse:
                scope.reuse_variables()
            for t in xrange(self.seq_length):
                if t > 0:
                    scope.reuse_variables()
                h_t, s_t = self.gen_rnn_unit(x_t, s_t)
                o_t = self.gen_output_unit(h_t)
                prob_t = tf.nn.softmax(o_t)
                # end-of-sequence indicators
                if t > 0:
                    eos = tf.not_equal(x_ind.read(t-1), self.eos_token)
                else:
                    eos = tf.ones([self.batch_size])
                eos = tf.to_float(eos)
                prob_t = tf.transpose(tf.multiply(eos, tf.transpose(prob_t)) + \
                                      tf.multiply(tf.subtract(1., eos), self.eos_prob_T))
                prob = prob.write(t, prob_t)  # batch x vocab_size
                x_t = x_emb.read(t)
                # dropout
                dropout_vec = tf.to_float(tf.less_equal(tf.random_uniform([self.batch_size, 1]), dropout_keep_prob))
                x_t = dropout_vec * x_t

                # reconstruct x
                log_prob_t = tf.log(tf.nn.softmax(o_t / temp_g))
                log_prob_t = tf.transpose(tf.multiply(eos, tf.transpose(log_prob_t)) + \
                    tf.multiply(tf.subtract(1., eos), self.eos_log_prob_T))
                next_token = tf.to_int32( \
                    tf.reshape(tf.multinomial(log_prob_t, 1), [self.batch_size]))
                recon_x = recon_x.write(t, next_token)  # indices, batch_size

        # batch_size x seq_length
        recon_x = tf.transpose(recon_x.stack(), perm=[1, 0])
        # batch_size x seq_length x vocab_size
        prob = tf.transpose(prob.stack(), perm=[1, 0, 2])
        return recon_x, prob, x_length

    def generate(self, init_state, temp_o=1., temp_g=1.):
        gen_x = tf.TensorArray(\
            tf.int32, size=self.seq_length, clear_after_read=False)
        gen_p = tf.TensorArray(tf.float32, size=self.seq_length)

        h_0 = init_state.h
        s_t = init_state
        x_t = tf.nn.embedding_lookup(self.gen_emb, self.bos_token)
        with tf.variable_scope("gen_rnn") as scope:
            scope.reuse_variables()
            for t in xrange(self.seq_length):
                h_t, s_t = self.gen_rnn_unit(x_t, s_t)
                o_t = self.gen_output_unit(h_t)
                if t > 0:
                    eos = tf.not_equal(gen_x.read(t-1), self.eos_token)
                else:
                    eos = tf.ones([self.batch_size])
                eos = tf.to_float(eos)
                # softmax approximation of generations
                prob_t = tf.nn.softmax(o_t / temp_o)
                prob_t = tf.transpose(tf.multiply(eos, tf.transpose(prob_t)) + \
                    tf.multiply(tf.subtract(1., eos), self.eos_prob_T))
                gen_p = gen_p.write(t, prob_t)
                # generate samples
                log_prob_t = tf.log(tf.nn.softmax(o_t / temp_g))
                log_prob_t = tf.transpose(tf.multiply(eos, tf.transpose(log_prob_t)) + \
                    tf.multiply(tf.subtract(1., eos), self.eos_log_prob_T))
                next_token = tf.to_int32( \
                    tf.reshape(tf.multinomial(log_prob_t, 1), [self.batch_size]))
                gen_x = gen_x.write(t, next_token)  # indices, batch_size
                # batch x emb_dim
                x_t = tf.nn.embedding_lookup(self.gen_emb, next_token)

        # batch_size x seq_length
        gen_x = tf.transpose(gen_x.stack(), perm=[1, 0])
        # batch_size x seq_length x vocab_size
        gen_p = tf.transpose(gen_p.stack(), perm=[1, 0, 2])

        return gen_x, gen_p


    def build_optimizor(self, sess, config):
        # pre training
        self.pt_global_step = tf.Variable(0, name="pt_global_step", trainable=False)
        pt_optim = tf.train.AdamOptimizer()
        pt_grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.pt_loss, self.vae_vars), self.grad_clip)
        self.pt_updates = pt_optim.apply_gradients(zip(pt_grads, self.vae_vars))

        ## generator training
        gen_optim = tf.train.AdamOptimizer()

        vae_nongen_grads = tf.gradients(self.gen_loss, self.vae_nongen_vars)
        vae_gen_grads = tf.gradients(self.gen_loss+self.gen_ind_loss, self.gen_vars)
        gen_grads = vae_nongen_grads + vae_gen_grads
        self.gen_updates = gen_optim.apply_gradients(zip(gen_grads, self.vae_vars))

        ## style transfer training
        style_optim = tf.train.AdamOptimizer()
        style_grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.style_loss, self.vae_vars), self.grad_clip)
        self.style_updates = style_optim.apply_gradients(zip(style_grads, self.vae_vars))

        # discriminator training
        self.pt_disc_global_step = tf.Variable(0, name="pt_disc_global_step", trainable=False)
        self.pt_disc_optim = tf.train.AdamOptimizer().minimize(\
            self.pt_disc_loss, var_list=self.disc_vars,
            global_step=self.pt_disc_global_step)

        #
        self.disc_global_step = tf.Variable(0, name="disc_global_step", trainable=False)
        self.disc_optim = tf.train.AdamOptimizer().minimize(\
            self.disc_loss, var_list=self.disc_vars,
            global_step=self.disc_global_step)

        self.build_summaries(sess, config)


    def build_summaries(self, sess, config):
        # pre-train
        with tf.name_scope("pt_summary"):
            self.pt_loss_sum = tf.summary.scalar("pt_loss", self.pt_loss)
            self.recon_loss_sum = tf.summary.scalar("recon_loss", self.recon_loss)
            self.kld_sum = tf.summary.scalar("kld", self.kld)
            self.kld_w_sum = tf.summary.scalar("kld_w", self.kld_w)
            self.bow_loss_sum = tf.summary.scalar("bow_loss", self.bow_loss)

            self.pt_sum = tf.summary.merge([ \
                self.pt_loss_sum, self.recon_loss_sum, self.kld_sum, self.kld_w_sum,
                self.bow_loss_sum])

            self.test_pt_loss_sum = tf.summary.scalar( \
                "test_pt_loss", self.test_pt_loss)
            self.test_recon_loss_sum = tf.summary.scalar( \
                "test_recon_loss", self.test_recon_loss)
            self.test_kld_sum = tf.summary.scalar( \
                "test_kld", self.test_kld)
            self.test_bow_loss_sum = tf.summary.scalar( \
                "test_bow_loss", self.test_bow_loss)

            self.test_pt_sum = tf.summary.merge([ \
                self.test_pt_loss_sum, self.test_recon_loss_sum, self.test_kld_sum,
                self.test_bow_loss_sum])

        # full training
        with tf.name_scope("gen_summary"):
            self.gen_u_loss_sum = tf.summary.scalar(\
                "scaled_gen_u_loss", self.u_loss*self.u_gen_w)
            self.gen_ind_loss_sum = tf.summary.scalar(\
                "scaled_gen_ind_loss", self.ind_loss*self.ind_gen_w)
            self.gen_pt_loss_sum = tf.summary.scalar("gen_pt_loss", self.pt_loss)
            self.gen_kld_sum = tf.summary.scalar("scaled_gen_kld", self.kld*self.kld_w)
            self.gen_recon_loss_sum = tf.summary.scalar("gen_recon_loss", self.recon_loss)
            self.gen_bow_loss_sum = tf.summary.scalar("gen_bow_loss", self.bow_loss)
            self.gen_loss_sum = tf.summary.scalar("gen_loss", self.gen_loss)
            self.gen_u_accu_sum = tf.summary.scalar("gen_u_accu", self.u_accu)

            self.gen_sum = tf.summary.merge([ \
                self.gen_u_loss_sum, self.gen_ind_loss_sum, self.gen_pt_loss_sum,
                self.gen_kld_sum, self.gen_recon_loss_sum, self.gen_bow_loss_sum,
                self.gen_loss_sum, self.gen_u_accu_sum])

        with tf.name_scope("pt_disc_summary"):
            self.pt_disc_loss_sum = tf.summary.scalar("pt_disc_loss", self.pt_disc_loss)
            self.disc_s_accu_sum = tf.summary.scalar("disc_s_accu", self.s_accu)
            self.pt_disc_sum = tf.summary.merge([self.pt_disc_loss_sum, self.disc_s_accu_sum])

        with tf.name_scope("disc_summary"):
            self.disc_u_loss_sum = tf.summary.scalar(\
                "scaled_disc_u_loss", self.u_loss*self.u_disc_w)
            self.disc_s_loss_sum = tf.summary.scalar("disc_s_loss", self.s_loss)
            self.disc_loss_sum = tf.summary.scalar("disc_loss", self.disc_loss)
            self.disc_sum = tf.summary.merge([ \
                self.disc_u_loss_sum, self.disc_s_loss_sum, self.disc_loss_sum,
                self.disc_s_accu_sum])

            self.test_accu_sum = tf.summary.scalar("test_accu", self.test_accu)
            self.test_disc_sum = tf.summary.merge([self.test_accu_sum])

        self.writer = tf.summary.FileWriter(config.summary_path, sess.graph)


    def train_one_step(self, sess, step, feed=None, display=False,
            sample=False, sample_nbatches=1, sample_path=None):
        # infer c of x
        s_prob = sess.run(self.s_prob, feed_dict=feed)
        #
        feed[self.c] = s_prob
        _, gen_loss, u_accu, u_loss, ind_loss, pt_loss, recon_loss, kld, bow_loss = sess.run([ \
            self.gen_updates, self.gen_loss, self.u_accu, self.u_loss, self.ind_loss,
            self.pt_loss, self.recon_loss, self.kld, self.bow_loss], feed_dict=feed)

        if display:
            summary_str = sess.run(self.gen_sum, feed_dict=feed)
            self.writer.add_summary(summary_str, step)

            temp_o = feed[self.temp_o]
            print("iter: %d, gen_loss: %.4f, u_accu: %.4f, u_loss: %.4f, ind_loss: %.4f " \
                  "pt_loss: %.4f, recon_loss: %.4f, kld: %.4f, bow_loss: %.4f, temp_o: %.4f" % \
                  (step, gen_loss, u_accu, u_loss, ind_loss, pt_loss, recon_loss, kld, bow_loss, temp_o))
            logging.info("iter: %d, gen_loss: %.4f, u_accu: %.4f, u_loss: %.4f, ind_loss: %.4f " \
                  "pt_loss: %.4f, recon_loss: %.4f, kld: %.4f, bow_loss: %.4f, temp_o: %.4f" % \
                  (step, gen_loss, u_accu, u_loss, ind_loss, pt_loss, recon_loss, kld, bow_loss, temp_o))

        if sample:
            samples = sess.run(self.sample_x_stitch(sample_nbatches), feed_dict=feed)
            dump_samples(samples, "%s/samples_%d.txt"%(sample_path,step))
            samples = sess.run(self.sample_x_recon(), feed_dict=feed)
            dump_samples(samples, "%s/recon_samples_%d.txt"%(sample_path,step))


    def train_style_transfer_one_step(self, sess, step, feed=None, display=False,
            sample=False, sample_nbatches=1, sample_path=None):
        # infer c of x
        s_prob = sess.run(self.s_prob, feed_dict=feed)
        # binarize
        s_prob = np.asarray(np.greater(s_prob, 0.5), dtype=np.float32)
        #
        feed[self.c] = s_prob
        _, style_loss, style_accu, pt_loss, recon_loss, kld, style_ind_loss, style_recon_loss = sess.run([ \
            self.style_updates, self.style_loss, self.style_accu,
            self.pt_loss, self.recon_loss, self.kld, self.style_ind_loss, self.style_recon_loss], feed_dict=feed)

        if display:
            temp_o = feed[self.temp_o]
            print("iter: %d, style_loss: %.4f, style_accu: %.4f " \
                    "pt_loss: %.4f, recon_loss: %.4f, kld: %.4f, ind_loss: %.4f, style_recon_loss: %.4f, temp_o: %.4f" % \
                  (step, style_loss, style_accu, pt_loss, recon_loss, kld, style_ind_loss, style_recon_loss, temp_o))
            logging.info("iter: %d, style_loss: %.4f, style_accu: %.4f " \
                    "pt_loss: %.4f, recon_loss: %.4f, kld: %.4f, ind_loss: %.4f, style_recon_loss: %.4f, temp_o: %.4f" % \
                  (step, style_loss, style_accu, pt_loss, recon_loss, kld, style_ind_loss, style_recon_loss, temp_o))

        if sample:
            samples = sess.run(self.sample_x_style(), feed_dict=feed)
            dump_samples(samples, "%s/recon_samples_%d.txt"%(sample_path,step))


    def pretrain_disc_one_step(self, sess, step, feed=None, display=False):
        _, pt_disc_loss, s_accu = sess.run([ \
            self.pt_disc_optim, self.pt_disc_loss, self.s_accu], feed_dict=feed)

        if display:
            summary_str = sess.run(self.pt_disc_sum, feed_dict=feed)
            self.writer.add_summary(summary_str, step)

            print("iter: %d, pt_disc_loss: %.4f, s_accu: %.4f" % (step, pt_disc_loss, s_accu))
            logging.info("iter: %d, pt_disc_loss: %.4f, s_accu: %.4f" % (step, pt_disc_loss, s_accu))


    def disc_evaluate(self, sess, data_loader, step):
        test_accu = []
        x_batches, y_batches = \
            zip(*data_loader.test_batch_iter(self.batch_size, 1))
        for x_batch, y_batch in zip(x_batches, y_batches):
            feed = {self.x: x_batch, self.c: y_batch, self.dropout_keep_prob: 1.}
            accu = sess.run(self.s_accu, feed_dict=feed)
            test_accu.append(accu)

        test_accu = np.mean(test_accu)
        summary_str = sess.run(\
            self.test_disc_sum, feed_dict={self.test_accu: test_accu})
        self.writer.add_summary(summary_str, step)
        print("test iter: %d, accu: %.4f" % (step, test_accu))
        logging.info("test iter: %d, accu: %.4f" % (step, test_accu))


    def pretrain_one_step(self, sess, step, feed=None, display=False,
            sample=False, sample_nbatches=1, sample_path=None):
        _, pt_loss, recon_loss, recon_loss_word, kld, kld_w, bow_loss = sess.run([ \
            self.pt_updates, self.pt_loss, self.recon_loss, self.recon_loss_word, self.kld,
            self.kld_w, self.bow_loss], feed_dict=feed)

        if display:
            print("iter: %d, pt_loss: %.4f, recon_loss: %.4f, recon_loss_word: %.4f, kld: %.4f, " \
                  "kld_w: %.4f, bow_loss %.4f" % \
                  (step, pt_loss, recon_loss, recon_loss_word, kld, kld_w, bow_loss))
            logging.info("iter: %d, pt_loss: %.4f, recon_loss: %.4f, recon_loss_word: %.4f, kld: %.4f, " \
                  "kld_w: %.4f, bow_loss %.4f" % \
                  (step, pt_loss, recon_loss, recon_loss_word, kld, kld_w, bow_loss))

        if sample:
            samples = sess.run(self.sample_x_stitch(sample_nbatches), feed_dict=feed)
            dump_samples(samples, "%s/samples_%d.txt"%(sample_path,step))
            samples = sess.run(self.sample_x_recon(), feed_dict=feed)
            dump_samples(samples, "%s/recon_samples_%d.txt"%(sample_path,step))

    def pretrain_evaluate(self, sess, data_loader, step, has_y=False):
        pt_losses, recon_losses, klds, bow_losses = [], [], [], []
        c = np.ones([self.batch_size, self.c_dim]) / self.c_dim
        if has_y:
            x_batches, _ = \
                zip(*data_loader.test_batch_iter(self.batch_size, 1))
        else:
            x_batches = data_loader.test_batch_iter(1, pad=True)
        for x_batch in x_batches:
            feed = {self.x: x_batch, self.c: c, self.kld_w: 1., self.temp_g: 1.}
            pt_loss, recon_loss, kld, bow_loss = sess.run([ \
                self.pt_loss, self.recon_loss, self.kld, self.bow_loss], feed_dict=feed)
            pt_losses.append(pt_loss)
            recon_losses.append(recon_loss)
            klds.append(kld)
            bow_losses.append(bow_loss)

        pt_loss = np.mean(pt_losses)
        recon_loss = np.mean(recon_losses)
        kld = np.mean(klds)
        bow_loss = np.mean(bow_losses)
        summary_str = sess.run(self.test_pt_sum, feed_dict={\
            self.test_pt_loss: pt_loss, self.test_recon_loss: recon_loss,
            self.test_kld: kld, self.test_bow_loss: bow_loss})
        self.writer.add_summary(summary_str, step)
        print("test iter: %d, pt_loss: %.4f, recon_loss: %.4f, kld: %.4f, bow_loss: %.4f" % \
              (step, pt_loss, recon_loss, kld, bow_loss))
        logging.info("test iter: %d, pt_loss: %.4f, recon_loss: %.4f, kld: %.4f, bow_loss: %.4f" % \
              (step, pt_loss, recon_loss, kld, bow_loss))


    def encode_x(self, x_vec, reuse=False):
        """
        x_vec: batch_size x seq_length x vocab_size
        """
        x_emb = tf.reshape( \
            tf.matmul(tf.reshape(x_vec, [-1, self.vocab_size]), self.gen_emb),
            [-1, self.seq_length, self.emb_dim])
        x_emb = tf.transpose(x_emb, [1, 0, 2])
        x_emb_ta = tf.TensorArray(dtype=tf.float32, size=self.seq_length)
        x_emb_ta = x_emb_ta.unstack(x_emb)
        x_mu, x_log_sigma = self.encode(x_emb_ta, reuse=True)
        return x_mu, x_log_sigma


    def encode(self, x_emb_ta, reuse=False):
        """
        x_emb_ta: TensorArray - embedding sequence
        """
        with tf.variable_scope("enc_rnn") as scope:
            if reuse:
                scope.reuse_variables()
            s_t = self.enc_rnn_unit.zero_state(self.batch_size, tf.float32)
            for t in xrange(self.seq_length):
                if t > 0:
                    scope.reuse_variables()
                x_t = x_emb_ta.read(t)
                _, s_t = self.enc_rnn_unit(x_t, s_t)
            mu, log_sigma = self.enc_output_layer(s_t.h)

        return mu, log_sigma

    def create_rnn_unit(self, scope_name):
        with tf.variable_scope(scope_name):
            cell = lstm_cell(self.hidden_dim)
            return cell

    def create_enc_output_layer(self):
        with tf.variable_scope('enc_output'):
            self.W_hmu = tf.Variable(self.init_matrix([self.hidden_dim, self.z_dim]),
                                     name="W_hmu")
            self.b_hmu = tf.Variable(self.init_matrix([self.z_dim]), name="b_hmu")
            self.W_hsigma = tf.Variable(self.init_matrix([self.hidden_dim, self.z_dim]),
                                        name="W_hsigma")
            self.b_hsigma = tf.Variable(self.init_matrix([self.z_dim]), name="b_hsigma")
        def unit(hidden_state):
            mu = tf.matmul(hidden_state, self.W_hmu) + self.b_hmu
            log_sigma = tf.matmul(hidden_state, self.W_hsigma) + self.b_hsigma
            return mu, log_sigma

        return unit

    def enc_sampler(self, mu, log_sigma, num_samples=None):
        if num_samples is None:
            eps = tf.random_normal(tf.shape(mu))
            # batch_size x z_dim
            z = mu + tf.exp(0.5 * log_sigma) * eps
        else:
            eps = tf.random_normal([self.batch_size, self.z_dim, num_samples])
            z = tf.expand_dims(mu, -1) + tf.exp(0.5 * log_sigma) * tf.expand_dims(eps, -1)
            # batch_size x num_sampels x z_dim
            z = tf.transpose(z, [0, 2, 1])

        return z

    def create_gen_init_state_layer(self):
        with tf.variable_scope('gen_init_state'):
            self.Wi = tf.Variable( \
                self.init_matrix([self.z_dim+self.c_dim, self.hidden_dim*2]), name='gen_Wi')
            self.bi = tf.Variable( \
                self.init_matrix([self.hidden_dim*2]), name='gen_bi')

        def unit(hidden_state):
            s = tf.matmul(hidden_state, self.Wi) + self.bi
            c = s[:,:self.hidden_dim]
            h = s[:,self.hidden_dim:]
            return lstm_state(c, h)

        return unit

    def create_gen_output_layer(self):
        with tf.variable_scope('gen_output'):
            self.Wo = tf.Variable( \
                self.init_matrix([self.hidden_dim, self.vocab_size]), name='gen_Wo')
            self.bo = tf.Variable( \
                self.init_matrix([self.vocab_size]), name='gen_bo')

        def unit(hidden_state):
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            return logits

        return unit

    def discriminator(self, input, y, scope=None, reuse=False):
        with tf.variable_scope(scope or 'disc_net') as scope:
            if reuse:
                scope.reuse_variables()

            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)

            # Embedding layer
            # average over word embeddings
            input_emb = tf.reshape(tf.matmul(tf.reshape(input, [-1, self.vocab_size]),
                                             self.disc_emb),
                                   [-1, self.seq_length, self.disc_emb_dim])
            # add the channel dimension
            input_emb = tf.expand_dims(input_emb, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(self.disc_filter_sizes, self.disc_filter_nums):
                with tf.name_scope("conv-%s" % filter_size):
                    # conv layer
                    filter_shape = [filter_size, self.disc_emb_dim, 1, num_filter]
                    W = tf.get_variable("conv-%s-W" % filter_size,
                            initializer=tf.to_float(tf.truncated_normal(filter_shape, stddev=0.1)))
                    b = tf.get_variable("conv-%s-b" % filter_size,
                            initializer=tf.constant(0.1, shape=[num_filter]))
                    conv = tf.nn.conv2d(
                        input_emb,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # maxpooling
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = sum(self.disc_filter_nums)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            # Add highway
            with tf.name_scope("highway"):
                h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)
            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_highway, self.dropout_keep_prob)
            # Final logits and predictions
            with tf.name_scope("output"):
                W = tf.get_variable( \
                    "output-W", initializer=tf.truncated_normal(\
                    [num_filters_total,self.c_dim], stddev=0.1))
                b = tf.get_variable(\
                    "output-b", initializer=tf.constant(0.1, shape=[self.c_dim]))
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                logits = tf.nn.xw_plus_b(h_drop, W, b, name="logits")
                prob = tf.nn.softmax(logits)
                pred = tf.argmax(logits, 1, name="pred")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                    logits=logits, labels=tf.to_float(y))) + self.disc_l2_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_preds = tf.equal(pred, tf.argmax(y, 1))
                accu = tf.reduce_mean(tf.cast(correct_preds, "float"), name="accu")

            return prob, pred, loss, accu


    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def load(self, sess, path):
        print("restore params from %s" % path)
        params = cPickle.load(open(path, "rb"))
        for i in xrange(len(self.t_vars)):
            if self.t_vars[i].name in params:
                #print(self.t_vars[i].name)
                sess.run(self.t_vars[i].assign(params[self.t_vars[i].name]))

    def save(self, sess, path, step=None):
        params = {}
        for i in xrange(len(self.t_vars)):
            params[self.t_vars[i].name] = sess.run(self.t_vars[i])
        ckpt_name = path
        if step != None:
            ckpt_name += "_vars_%d.p" % step
        else:
            ckpt_name += "_vars.p"
        print("snapshot params to %s" % ckpt_name)
        cPickle.dump(params, open(ckpt_name, "wb"))

    def save_disc(self, sess, path, step=None):
        params = {}
        for i in xrange(len(self.t_vars)):
            if "disc_" in self.t_vars[i].name:
                params[self.t_vars[i].name] = sess.run(self.t_vars[i])
        print("save #disc_vars %d" % len(params))
        ckpt_name = path
        if step != None:
            ckpt_name += "_disc_vars_%d.p" % step
        else:
            ckpt_name += "_disc_vars.p"
        print("snapshot params to %s" % ckpt_name)
        cPickle.dump(params, open(ckpt_name, "wb"))


# highway layer that borrowed from https://github.com/carpedm20/lstm-char-cnn-tensorflow
def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    output = input_
    for idx in xrange(layer_size):
        with tf.variable_scope('output_lin_%d' % idx):
            output = f(tf.contrib.layers.fully_connected(output, size.value))

        with tf.variable_scope('transform_lin_%d' % idx):
            transform_gate = tf.sigmoid(\
                tf.contrib.layers.fully_connected(input_, size.value) + bias)
        carry_gate = 1. - transform_gate

        output = transform_gate * output + carry_gate * input_

    return output



