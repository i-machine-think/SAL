#! /bin/sh

TEST_PATH=../machine-tasks/temp_data/longer_compositions
EXPT_DIR=../elia_results/attention_models
OUT_DIR=longer_accuracies
MODEL=full_focus
#MODEL=full_focus_hard
#/$(ls -t $EXPT_DIR/ | head -1)
echo "Accuracy for longer plots "
python long_accuracy.py --checkpoint_path $EXPT_DIR --ignore_output_eos --test $TEST_PATH --output_dir $OUT_DIR --model_type $MODEL --use_attention_loss