#! /bin/bash

dims="256-256-256"
num_blocks=1
batch_size=64
lr=2e-3
epochs=100
train_sample_points=256
val_sample_points=256
noise_bit=3
log_name="byrallies"
data_dir="data/byrallies"
layer_type="concatsquash"
player_style_dim=32
context_cat='concat'
multires=8
reg_type='nll'
reg_w=1.0
condition_type='continuous'
num_past_balls=2

CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --log_name ${log_name} \
    --lr ${lr} \
    --data_dir ${data_dir} \
    --dims ${dims} \
    --num_blocks ${num_blocks} \
    --batch_size ${batch_size} \
    --epochs ${epochs} \
    --layer_type ${layer_type} \
    --sample_points ${train_sample_points} \
    --val_sample_points ${val_sample_points} \
    --context_cat ${context_cat} \
    --noise_bit ${noise_bit} \
    --multires ${multires} \
    --regularization_type ${reg_type} \
    --regularization_w ${reg_w} \
    --condition_type ${condition_type} \
    --num_past_balls ${num_past_balls} \
    --save_freq 5 \
    --viz_freq 1 \
    --log_freq 1 \
    --val_freq 1 \
    --player_style_dim ${player_style_dim} \
    --pair_condition \
    --w_space 30 \
    --h_space 30 \
    --regularization
    # --no_player_embedding
# done