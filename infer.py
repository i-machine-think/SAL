import os
import argparse
import logging
import numpy as np
import random
import torch
import torchtext

import seq2seq
from seq2seq.evaluator import Predictor, PlotAttention
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
parser.add_argument('--test', help='Path to test data')
parser.add_argument('--train', help='Path to train data')
parser.add_argument('--output_dir', help='Path to save results')
parser.add_argument('--run_infer', action='store_true')
parser.add_argument('--max_plots', type=int, help='Maximum sequence length', default=10)
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--use_input_eos', action='store_true', help='EOS symbol in input sequences is not used by default. Use this flag to enable.')
parser.add_argument('--ignore_output_eos', action='store_true', help='Ignore end of sequence token during training and evaluation')

opt = parser.parse_args()

def getarr(data):
    data_src = []
    data_tgt = []
    for i in range(len(data)):
        data_src.append(vars(data[i])[seq2seq.src_field_name])
        data_tgt.append(vars(data[i])[seq2seq.tgt_field_name][1:])
    master_data = np.zeros((len(data_src),2), dtype=object)
    for i in range(len(data_src)):
        master_data[i, 0] = ' '.join(map(str, data_src[i]))
        master_data[i, 1] = ' '.join(map(str, data_tgt[i]))
    return master_data

#################################################################################
#prepare data
use_output_eos = not opt.ignore_output_eos
src = SourceField(use_input_eos=opt.use_input_eos)
tgt = TargetField(include_eos=use_output_eos)
max_len = opt.max_len

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

# generate training and testing data
train = torchtext.data.TabularDataset(
    path=opt.train, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)

if opt.test:
    test = torchtext.data.TabularDataset(
        path=opt.test, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
else:
    test = None

train_data = getarr(train)
test_data = getarr(test)
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
attn_plotter = PlotAttention(train_data)
if(opt.max_plots > test_data.shape[0]):
    raise ValueError("max plots exceeded total number of data points in the dataset")
idxs = random.sample(np.arange(test_data.shape[0]).tolist(),opt.max_plots)

print("Begin Plotting")
for i in range(len(idxs)):
    ipt_sentence = test_data[i,0]
    seq = ipt_sentence.strip().split()
    outputs, attention = predictor.predict(seq)
    if (attention.all()==0):
        break
    name = os.path.join(opt.output_dir, 'plot'+'{}'.format(i))
    attn_plotter.evaluateAndShowAttention(ipt_sentence, outputs, attention, name)

if opt.run_infer:
    while True:
            seq_str = raw_input("Type in a source sequence:")
            seq = seq_str.strip().split()
            out = predictor.predict(seq)
            print(out[0])
else:
    print("Exiting Inference Mode")
