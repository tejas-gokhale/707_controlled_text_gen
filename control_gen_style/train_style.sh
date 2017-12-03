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

# prior: normal, exponential, beta
prior_distr='normal'
prior_mu=0.0
prior_sigma=1.0

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
display_every=1
test_every=-1
sample_every=-1
checkpoint_every=-10
# ful train
nepochs=50 #500
nbatches=300
# pre-train
pt_nepochs=30 # dumb
pt_kld_anneal_start_epoch=20 # dumb
pt_kld_anneal_end_epoch=2000 # dumb
pt_restore_epoch=0 # dumb

name="ctrl_${mode}"

python train_style_main.py \
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
  --ind_gen_w $ind_gen_w \
  --bow_w $bow_w \
  --recon_dropout_keep_prob $recon_dropout_keep_prob \
  --style_w $style_w \
  --output_path_prefix $output_path_prefix \
  --restore_ckpt_path $restore_ckpt_path \
  --restore_vars_path $restore_vars_path \
  --restore_disc_vars_path $restore_disc_vars_path \
  --restore_start_step $restore_start_step \
  --display_every $display_every \
  --test_every $test_every \
  --sample_every $sample_every \
  --checkpoint_every $checkpoint_every \
  --nepochs $nepochs \
  --nbatches $nbatches \
  --kld_w $kld_w \
  --pt_nepochs $pt_nepochs \
  --pt_kld_anneal_start_epoch $pt_kld_anneal_start_epoch \
  --pt_kld_anneal_end_epoch $pt_kld_anneal_end_epoch \
  --pt_restore_epoch $pt_restore_epoch \
  --prior_distr $prior_distr \
  --prior_mu $prior_mu \
  --prior_sigma $prior_sigma 
