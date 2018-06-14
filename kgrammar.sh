#!/usr/bin/env bash

TEST_PATH=../machine-tasks/k-grammar
EXPT_DIR=../elia_results/k-grammar
OUT_DIR=attention_plots/k-grammar
#/$(ls -t $EXPT_DIR/ | head -1)
echo "Run in inference mode"
python infer_kgrammar.py --checkpoint_path $EXPT_DIR --ignore_output_eos --test $TEST_PATH --output_dir $OUT_DIR