#! /bin/sh

TRAIN_PATH=data/lookup-3bit/train_attn.tsv
DEV_PATH=data/lookup-3bit/validation_attn.tsv
TEST1_PATH=data/lookup-3bit/test1_heldout_attn.tsv
TEST2_PATH=data/lookup-3bit/test2_subset_attn.tsv
TEST3_PATH=data/lookup-3bit/test3_hybrid_attn.tsv
TEST4_PATH=data/lookup-3bit/test4_unseen_attn.tsv
TEST5_PATH=data/lookup-3bit/test5_longer_attn.tsv
EXPT_DIR=example

# use small parameters for quicker testing
EMB_SIZE=300
H_SIZE=300
CELL='gru'
EPOCH=100
PRINT_EVERY=9999999
SAVE_EVERY=9999999
ATTN='post-rnn'
ATTN_METHOD='provided'
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
	--batch_size $BATCH_SIZE \
	--optim adam \

python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --attention_method $ATTN_METHOD --test_data $TRAIN_PATH
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --attention_method $ATTN_METHOD --test_data $DEV_PATH
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --attention_method $ATTN_METHOD --test_data $TEST1_PATH
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --attention_method $ATTN_METHOD --test_data $TEST2_PATH
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --attention_method $ATTN_METHOD --test_data $TEST3_PATH
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --attention_method $ATTN_METHOD --test_data $TEST4_PATH
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --attention_method $ATTN_METHOD --test_data $TEST5_PATH