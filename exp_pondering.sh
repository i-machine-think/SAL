#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=1:00:00:00
#PBS -e error.txt
#PBS -o output.txt

module load python

cd ~/exp_pondering

TRAIN_PATH='./train_sort.tsv'
DEV_PATH='./test_sort.tsv'
EXPT_DIR=example

# set values
EMB_SIZE=256
H_SIZE=256
N_LAYERS=1
CELL='lstm'
EPOCH=1000
PRINT_EVERY=50
TF=0.5

# Start training
echo "Train model on example data"
python train_model.py --max_len 100 --train $TRAIN_PATH \
--output_dir $EXPT_DIR --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --save_every 9999999999999999 --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF --attention 'pre-rnn' --attention_method 'mlp' \
--ponder_encoder \
--ponder_decoder \
--max_ponder_steps 100 \
--ponder_epsilon 0.01 \
--ponder_penalty_scale 0.1 \
--batch_size 16 \
--eval_batch_size 1000
--dev $DEV_PATH --monitor $DEV_PATH \
