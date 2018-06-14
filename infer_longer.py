import os
import argparse
import logging
import numpy as np
import random
import torch
import torchtext
from torch.autograd import Variable
import seq2seq
from seq2seq.evaluator import Predictor, PlotAttention
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField, AttentionField

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
final_level = ['full_focus_hard','pre_rnn_hard']
data_folders = ['longer_compositions']
regular = [('heldout_compositions3', 50), ('heldout_tables3', 50), ('new_compositions3', 50), ('heldout_compositions4', 50),
           ('heldout_tables4', 50), ('new_compositions4', 50), ('heldout_compositions5', 50), ('heldout_tables5', 50),
           ('new_compositions5', 50), ('heldout_compositions6', 50), ('heldout_tables6', 50), ('new_compositions6', 50),
           ('heldout_compositions7', 50), ('heldout_tables7', 50), ('new_compositions7', 50), ('heldout_compositions8', 50),
           ('heldout_tables8', 50), ('new_compositions8', 50), ('heldout_compositions9', 50), ('heldout_tables9', 50),
           ('new_compositions9', 50), ('heldout_compositions10', 50), ('heldout_tables10', 50), ('new_compositions10', 50)]

test_folders = [regular]

if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

IGNORE_INDEX=-1
use_output_eos = not opt.ignore_output_eos
src = SourceField(use_input_eos=opt.use_input_eos)
tgt = TargetField(include_eos=use_output_eos)
attn = AttentionField(use_vocab=False, ignore_index=IGNORE_INDEX)
max_len = opt.max_len



def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

def getarr(data):
    data_src = []
    data_tgt = []
    data_attn = []
    for i in range(len(data)):
        data_src.append(vars(data[i])[seq2seq.src_field_name])
        data_tgt.append(vars(data[i])[seq2seq.tgt_field_name])
        data_attn.append(vars(data[i])[seq2seq.attn_field_name])
    master_data = np.zeros((len(data_src),3), dtype=object)
    for i in range(len(data_src)):
        master_data[i, 0] = ' '.join(map(str, data_src[i]))
        master_data[i, 1] = ' '.join(map(str, data_tgt[i]))
        master_data[i, 2] = data_attn[i]
    return master_data

def load_model(checkpoint_path):
    logging.info("loading checkpoint from {}".format(os.path.join(checkpoint_path)))
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    src.vocab = input_vocab
    src.eos_id = src.vocab.stoi[src.SYM_EOS]
    output_vocab = checkpoint.output_vocab
    tgt.vocab = output_vocab
    tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
    tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]
    return seq2seq, input_vocab, output_vocab

def prepare_data(data_path):
    # generate training and testing data
    tabular_data_fields = [('src', src), ('tgt', tgt), ('attn', attn)]
    gen_data = torchtext.data.TabularDataset(
        path=data_path, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    )
    data_arr = getarr(gen_data)
    return data_arr

def plot_attention(attn_plotter, idxs, test_data, trunc_length, opt_path):
    print("Begin Plotting")
    for x in idxs:
        ipt_sentence = test_data[x, 0]
        opt_sentence = test_data[x,1]
        tgt_var = list(map(int,test_data[x,2]))
        seq = ipt_sentence.strip().split()
        tgt_seq = opt_sentence.strip().split()
        #tgt_seq.append('000')
        outputs, attention = predictor.predict(seq,tgt_seq,tgt_var)
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
            for l in range(train_data.shape[0]):
                train_data[l,1] = ' '.join(map(str, train_data[l,1].split()[1:]))
            # print(train_data)
            # input()
            attn_plotter = PlotAttention(train_data)
            for k,folder in enumerate(test_folders):
                for sub in folder:
                    test_path = os.path.join(opt.test,data_folders[k],level1,sub[0]+'.tsv')
                    test_data= prepare_data(test_path)
                    opt_lengths = [len(test_data[i, 1].strip().split()) for i in range(test_data.shape[0])]
                    trunc_length = max(opt_lengths)
                    if(sub[1] <= test_data.shape[0]):
                        idxs = random.sample(np.arange(test_data.shape[0]).tolist(), sub[1])
                    else:
                        idxs = random.sample(np.arange(test_data.shape[0]).tolist(), test_data.shape[0])
                    opt_path = os.path.join(opt.output_dir,level1,level2,level3,level4, sub[0])
                    if not os.path.exists(opt_path):
                        os.makedirs(opt_path)
                    plot_attention(attn_plotter, idxs, test_data, trunc_length, opt_path)

    print("finished plotting for sample{}".format(i))
