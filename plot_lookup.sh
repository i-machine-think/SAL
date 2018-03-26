#! /bin/sh

TRAIN_PATH=data/lookup-3bit/train.csv
TEST_PATH=data/lookup-3bit/test1_heldout.csv
EXPT_DIR=checkpoint
OUT_DIR=results

echo "Run in inference mode"
python infer.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --train $TRAIN_PATH --test $TEST_PATH --output_dir $OUT_DIR

