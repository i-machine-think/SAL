#! /bin/sh

TRAIN_PATH='./train_parity.tsv'
DEV_PATH='./test_parity.tsv'
EXPT_DIR=example

# set values
EMB_SIZE=32
H_SIZE=32
N_LAYERS=1
CELL='lstm'
EPOCH=1000
PRINT_EVERY=100
TF=0.5

# Start training
echo "Train model on example data"
python train_model.py --max_len 100 --train $TRAIN_PATH --output_dir $EXPT_DIR --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF --ponder_encoder --ponder_decoder --max_ponder_steps 10 --ponder_epsilon 0.05 --ponder_penalty_scale 0.001 --batch_size 100 --eval_batch_size 1000 --ignore_output_eos
