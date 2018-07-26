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
# <<<<<<< HEAD
# parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
# parser.add_argument('--test', help='Path to test data')
# parser.add_argument('--train', help='Path to train data')
# parser.add_argument('--output_dir', help='Path to save results')
# parser.add_argument('--run_infer', action='store_true')
# parser.add_argument('--max_plots', type=int, help='Maximum sequence length', default=10)
# parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
# parser.add_argument('--use_input_eos', action='store_true', help='EOS symbol in input sequences is not used by default. Use this flag to enable.')
# parser.add_argument('--ignore_output_eos', action='store_true', help='Ignore end of sequence token during training and evaluation')
# =======
parser.add_argument('--cuda_device', default=0, type=int, help='Set cuda device to use')
parser.add_argument('--debug', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--log-level', default='info', help='Logging level.')

opt = parser.parse_args()
final_level = ['full_focus', 'full_focus_baseline', 'pre_rnn', 'pre_rnn_baseline']
data_folders = ['samples']
regular = [('longer_compositions_seen', 50), ('longer_compositions_incremental', 50), ('longer_compositions_new', 50)]
#[('heldout_inputs', 50), ('heldout_compositions',50), ('heldout_tables',50), ('new_compositions', 50),]
#longer = []
test_folders = [regular]

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)


if torch.cuda.is_available():
        logging.info("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

use_output_eos = not opt.ignore_output_eos
src = SourceField(use_input_eos=opt.use_input_eos)
tgt = TargetField(include_eos=use_output_eos)
max_len = opt.max_len


def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

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

def load_model(checkpoint_path):
    logging.info("loading checkpoint from {}".format(os.path.join(checkpoint_path)))
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
    return seq2seq, input_vocab, output_vocab

def prepare_data(data_path):
    # generate training and testing data
    tabular_data_fields = [('src', src), ('tgt', tgt)]
    gen_data = torchtext.data.TabularDataset(
        path=data_path, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    )
    data_arr = getarr(gen_data)
    return data_arr

def plot_attention(idxs, test_data, trunc_length, opt_path):
    print("Begin Plotting")
    for x in idxs:
        ipt_sentence = test_data[x, 0]
        seq = ipt_sentence.strip().split()
        outputs, attention = predictor.predict(seq)
        # if (attention.all()==0):
        #     break
        name = os.path.join(opt_path, 'plot' + '{}'.format(x))
        attn_plotter.evaluateAndShowAttention(ipt_sentence, outputs, attention[:trunc_length], name)



for i in range(2, 6):
    level1 = 'sample{}'.format(i)
    for j in range(1, 3):
        level2 = 'run{}'.format(j)
        for level3 in final_level:
            # load model
            model_path = os.path.join(opt.checkpoint_path, level1, level2, level3)
            level4 = os.listdir(model_path)[0]
            level5 = os.listdir(os.path.join(model_path, level4))[0]
            final_model_path = os.path.join(model_path, level4, level5)
            model, input_vocab, output_vocab = load_model(final_model_path)
            predictor = Predictor(model, input_vocab, output_vocab)
            train_data = prepare_data(os.path.join(opt.train,level1, 'train.tsv'))
            # print(train_data)
            # input()
            attn_plotter = PlotAttention(train_data)
            for k,folder in enumerate(test_folders):
                for sub in folder:
                    test_path = os.path.join(opt.test,data_folders[k],level1,sub[0]+'.tsv')
                    test_data = prepare_data(test_path)
                    opt_lengths = [len(test_data[i, 1].strip().split()) for i in range(test_data.shape[0])]
                    trunc_length = max(opt_lengths)
                    if(sub[1] <= test_data.shape[0]):
                        idxs = random.sample(np.arange(test_data.shape[0]).tolist(), sub[1])
                    else:
                        idxs = random.sample(np.arange(test_data.shape[0]).tolist(), test_data.shape[0])
                    opt_path = os.path.join(opt.output_dir,level1,level2,level3,level4, sub[0])
                    if not os.path.exists(opt_path):
                        os.makedirs(opt_path)
                    plot_attention(idxs, test_data, trunc_length, opt_path)
    print("finished plotting for sample{}".format(i))






# if(opt.max_plots > test_data.shape[0]):
#     opt.max_plots = test_data.shape[0]
#     #raise ValueError("max plots exceeded total number of data points in the dataset")



# <<<<<<< HEAD
# # if opt.run_infer:
# #     while True:
# #             seq_str = raw_input("Type in a source sequence:")
# #             seq = seq_str.strip().split()
# #             out = predictor.predict(seq)
# #             print(out[0])
# # else:
# #     print("Exiting Inference Mode")
# =======
if opt.debug:
    exit()

while True:
        seq_str = raw_input("\n\nType in a source sequence: ")
        if seq_str == 'q':
            exit()
        seq = seq_str.strip().split()
        print(predictor.predict(seq))

