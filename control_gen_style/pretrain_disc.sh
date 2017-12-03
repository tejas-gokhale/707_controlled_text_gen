#!/bin/bash

# generic options
mode='pretrain_disc'
restore_ckpt_path='./mult_hot_out/models/disc/'
restore_vars_path='./mult_hot_out/models/disc/'
restore_start_step=0
#
root_path='./'

# model
hidden_dim=300
emb_dim=300
disc_emb_dim=$emb_dim
c_dim=10
z_dim=300
disc_filter_sizes='3,4,5'
disc_filter_nums='100,100,100'
disc_l2_lambda=0.2
dropout_keep_prob=0.75

# data
seq_length=40
vocab_size=16188
bos_token=0
eos_token=$((vocab_size-1))
data_path="./mult_hot_out/disc_data/"
embedding_path='./data/imdb.data.binary.p.0.01.l16'
hot='mult'

# training
batch_size=64
dics_lr=0.001
u_gen_w=0.1
u_disc_w=0.1
output_path_prefix="${root_path}mult_hot_out/outputs"
display_every=800
test_every=-1
sample_every=-10 # dumb
checkpoint_every=-5
# pre-train
pt_nepochs=60
pt_kld_anneal_start_epoch=20 # dumb
pt_kld_anneal_end_epoch=2000 # dumb
pt_restore_epoch=0

name="ctrl_${mode}"

TZ=US/Eastern python pretrain_disc_main.py \
  --name $name \
  --mode $mode \
  --hidden_dim $hidden_dim \
  --emb_dim $emb_dim \
  --disc_emb_dim $disc_emb_dim \
  --c_dim $c_dim \
  --z_dim $z_dim \
  --disc_filter_sizes $disc_filter_sizes \
  --disc_filter_nums $disc_filter_nums \
  --disc_l2_lambda $disc_l2_lambda \
  --dropout_keep_prob $dropout_keep_prob \
  --seq_length $seq_length \
  --vocab_size $vocab_size \
  --bos_token $bos_token \
  --eos_token $eos_token \
  --data_path $data_path \
  --embedding_path $embedding_path \
  --batch_size $batch_size \
  --dics_lr $dics_lr \
  --u_gen_w $u_gen_w \
  --u_disc_w $u_disc_w \
  --output_path_prefix $output_path_prefix \
  --restore_ckpt_path $restore_ckpt_path \
  --restore_vars_path $restore_vars_path \
  --restore_start_step $restore_start_step \
  --display_every $display_every \
  --test_every $test_every \
  --sample_every $sample_every \
  --checkpoint_every $checkpoint_every \
  --pt_nepochs $pt_nepochs \
  --pt_kld_anneal_start_epoch $pt_kld_anneal_start_epoch \
  --pt_kld_anneal_end_epoch $pt_kld_anneal_end_epoch \
  --pt_restore_epoch $pt_restore_epoch \
  --hot $hot

