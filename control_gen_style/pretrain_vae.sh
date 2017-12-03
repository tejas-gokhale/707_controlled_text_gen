#!/bin/bash

# generic options
mode='pretrain_vae'
restore_ckpt_path='/'
restore_vars_path='/'
restore_start_step=0
#
root_path='./'

# model
hidden_dim=300
emb_dim=300
disc_emb_dim=$emb_dim
c_dim=2
z_dim=50 #TODO
disc_filter_sizes='3,4,5'
disc_filter_nums='100,100,100'
disc_l2_lambda=0.2

# data
seq_length=16
vocab_size=16188
bos_token=0
eos_token=$((vocab_size-1))
data_path="./mult_hot_out/vae_data/"
embedding_path='./data/imdb.data.binary.p.0.01.l16'

# training
batch_size=64
dics_lr=0.001
u_gen_w=0.1
u_disc_w=0.1
bow_w=0.2
recon_dropout_keep_prob=1.
output_path_prefix="${root_path}/mult_hot_out/outputs/"
display_every=-10 #-1
test_every=-100 #-5
sample_every=-100 #-10
checkpoint_every=-100 #-10
# pre-train
pt_nepochs=100000
pt_kld_anneal_start_epoch=200 #20 #200
pt_kld_anneal_end_epoch=20000 #2000 #20000
pt_restore_epoch=0

name="ctrl_${mode}"

CUDA_VISIBLE_DEVICES=0 python pretrain_vae_main.py \
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
  --bow_w $bow_w \
  --recon_dropout_keep_prob $recon_dropout_keep_prob \
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
  --pt_restore_epoch $pt_restore_epoch
