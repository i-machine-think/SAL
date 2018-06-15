#! /bin/sh

TRAIN_PATH='./train_parity.tsv'
DEV_PATH='./test_parity.tsv'
EXPT_DIR=example

# set values
EMB_SIZE=128
H_SIZE=128
N_LAYERS=1
CELL='lstm'
EPOCH=10
PRINT_EVERY=100
TF=0.5

# Start training
echo "Train model on example data"
python train_model.py --max_len 100 --train $TRAIN_PATH --output_dir $EXPT_DIR --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF --attention 'pre-rnn' --attention_method 'mlp' --ponder_decoder --max_ponder_steps 100 --ponder_epsilon 0.01 --ponder_penalty_scale 0.0001 --batch_size 32 --eval_batch_size 1000
