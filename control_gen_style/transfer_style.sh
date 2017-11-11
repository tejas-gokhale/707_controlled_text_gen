#!/bin/bash

# generic options
mode='style'
restore_ckpt_path='/'
#
root_path='.'
restore_disc_vars_path="${root_path}/example_models/pre_disc/snapshots/_disc_vars_50880.p"
#
restore_vars_path="${root_path}/example_models/pre_vae/snapshots/_vars_464000.p"
restore_start_step=0
kld_w=0.0909

# model
hidden_dim=300
emb_dim=300
disc_emb_dim=$emb_dim
c_dim=2
z_dim=50
disc_filter_sizes='3,4,5'
disc_filter_nums='100,100,100'
disc_l2_lambda=0.2
dropout_keep_prob=0.75

# data
seq_length=16
vocab_size=16188
bos_token=0
eos_token=$((vocab_size-1))
data_path="./data"
embedding_path='./data/imdb.data.binary.p.0.01.l16'

# training
batch_size=256
dics_lr=0.001
u_gen_w=0.1
u_disc_w=0.1
ind_gen_w=0.5
bow_w=0.2
recon_dropout_keep_prob=1.
style_w=0.1
output_path_prefix="${root_path}/outputs/"
display_every=-1
test_every=-1
sample_every=-1
checkpoint_every=-10
# ful train
nepochs=1 #50 #500
nbatches=300
# pre-train
pt_nepochs=1 # 30 # dumb
pt_kld_anneal_start_epoch=20 # dumb
pt_kld_anneal_end_epoch=2000 # dumb
pt_restore_epoch=0 # dumb

name="ctrl_${mode}"

CUDA_VISIBLE_DEVICES=0 python transfer_style_main.py