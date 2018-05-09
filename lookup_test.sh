#! /bin/sh

TRAIN_PATH=data/lookup-3bit/train.csv
DEV_PATH=data/lookup-3bit/validation.csv
TEST1_PATH=data/lookup-3bit/test1_heldout.csv
TEST2_PATH=data/lookup-3bit/test2_subset.csv
TEST3_PATH=data/lookup-3bit/test3_hybrid.csv
TEST4_PATH=data/lookup-3bit/test4_unseen.csv
TEST5_PATH=data/lookup-3bit/test5_longer.csv
EXPT_DIR=example

# use small parameters for quicker testing
EMB_SIZE=300
H_SIZE=300
CELL='gru'
EPOCH=100
PRINT_EVERY=9999999
SAVE_EVERY=9999999
ATTN='post-rnn'
ATTN_METHOD='hard'
BATCH_SIZE=10

python train_model.py \
	--train $TRAIN_PATH \
	--dev $DEV_PATH \
	--output_dir $EXPT_DIR \
	--print_every $PRINT_EVERY \
	--embedding_size $EMB_SIZE \
	--hidden_size $H_SIZE \
	--rnn_cell $CELL \
	--attention $ATTN \
	--attention_method $ATTN_METHOD \
	--epoch $EPOCH \
	--save_every $SAVE_EVERY \
	--teacher_forcing_ratio 0 \
    --scale_attention_loss 1 \
	--batch_size $BATCH_SIZE \
	--optim adam \

python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --test_data $TRAIN_PATH
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --test_data $DEV_PATH
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --test_data $TEST1_PATH
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --test_data $TEST2_PATH
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --test_data $TEST3_PATH
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --test_data $TEST4_PATH
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --test_data $TEST5_PATH