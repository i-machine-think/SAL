import os
import argparse
import logging
import pandas as pd
import torch

import seq2seq
from seq2seq.evaluator import Predictor, PlotAttention
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input      # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
parser.add_argument('--test', help='Path to test data')
parser.add_argument('--train', help='Path to train data')
parser.add_argument('--output_dir', help='Path to save results')
parser.add_argument('--plot_test_only', action='store_true')
parser.add_argument('--run_infer', action='store_true')

opt = parser.parse_args()

#################################################################################
#prepare data
df1 = pd.read_csv(opt.train,header=None, sep='\t')
df2 = pd.read_csv(opt.test,header=None, sep='\t')

train = df1.values
test = df2.values

#################################################################################

if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

#################################################################################
# load model

logging.info("loading checkpoint from {}".format(os.path.join(opt.checkpoint_path)))
checkpoint = Checkpoint.load(opt.checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

#################################################################################
# Generate predictor

predictor = Predictor(seq2seq, input_vocab, output_vocab)
attn_plotter = PlotAttention(train)

master_data = [train, test]
data_name = ['train', 'test']

print("Begin Plotting")
for step, data in enumerate(master_data):
    if (opt.plot_test_only and data_name[step]=='train'):
        continue
    for i in range(0,data.shape[0]):
        ipt_sentence = data[i,0]
        seq = ipt_sentence.strip().split()
        outputs, attention = predictor.predict(seq)
        if (attention==0):
            break
        name = os.path.join(opt.output_dir, data_name[step], data_name[step]+'{}'.format(i))
        attn_plotter.evaluateAndShowAttention(ipt_sentence, outputs, attention, name)

if opt.run_infer:
    while True:
            seq_str = raw_input("Type in a source sequence:")
            seq = seq_str.strip().split()
            out = predictor.predict(seq)
            print(out[0])
else:
    print("Exiting Inference Mode")
