#! /bin/sh

TRAIN_PATH=../machine-tasks/LookupTables/lookup-3bit/samples
TEST_PATH=../machine-tasks/temp_data
EXPT_DIR=../elia_results/attention_models
OUT_DIR=attention_plots/longer
#/$(ls -t $EXPT_DIR/ | head -1)
echo "Run in inference mode"
python infer_longer.py --checkpoint_path $EXPT_DIR --ignore_output_eos --train $TRAIN_PATH --test $TEST_PATH --output_dir $OUT_DIR