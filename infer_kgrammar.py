import os
import argparse
import logging
import numpy as np
import random
import torch
import torchtext
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seq2seq
from seq2seq.evaluator import Predictor
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
parser.add_argument('--output_dir', help='Path to save results')
parser.add_argument('--run_infer', action='store_true')
parser.add_argument('--max_plots', type=int, help='Maximum sequence length', default=10)
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--use_input_eos', action='store_true', help='EOS symbol in input sequences is not used by default. Use this flag to enable.')
parser.add_argument('--ignore_output_eos', action='store_true', help='Ignore end of sequence token during training and evaluation')

opt = parser.parse_args()
model_levels = ['learned_models','baseline_models']
test_sets = [('std', 10), ('short', 10), ('repeat', 10), ('long', 10)]

if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

IGNORE_INDEX=-1
use_output_eos = not opt.ignore_output_eos
src = SourceField(use_input_eos=opt.use_input_eos)
tgt = TargetField(include_eos=use_output_eos)
attn = AttentionField(use_vocab=False, ignore_index=IGNORE_INDEX)
max_len = opt.max_len

grammar_vocab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'T',
                          'U', 'V', 'W', 'X', 'Y', 'Z', 'AS', 'BS', 'CS', 'DS', 'ES', 'FS', 'GS', 'HS', 'IS', 'JS',
                          'KS', 'LS', 'MS', 'NS', 'OS']

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

def showAttention(input_sentence, output_words, attentions,name,colour):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(10,14))
    ax = fig.add_subplot(111)
    ax.yaxis.tick_right()
    cax = ax.matshow(attentions, cmap='bone', vmin=0, vmax=1)
    #fig.colorbar(cax)
    cbaxes = fig.add_axes([0.05, 0.1, 0.03, 0.8])
    cb = plt.colorbar(cax, cax=cbaxes)
    cbaxes.yaxis.set_ticks_position('left')

    # Set up axes
    ax.set_xticks(np.arange(len(input_sentence.split())+1))
    ax.set_yticks(np.arange(len(output_words)+1))
    ax.set_xticklabels([''] + input_sentence.split(' '), rotation=0) #+['<EOS>']
    ax.set_yticklabels([''] + output_words)
    # print(ax.get_yticklabels())
    # input()
    #Colour ticks
    for ytick, color in zip(ax.get_yticklabels()[1:], colour):
        ytick.set_color(color)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #X and Y labels
    ax.set_xlabel("INPUT")
    ax.set_ylabel("OUTPUT")
    ax.yaxis.set_label_position('right')
    ax.xaxis.set_label_position('top')

    plt.savefig("{}.png".format(name))
    plt.close(fig)
    #plt.show()

def plot_attention(idxs, test_data,opt_path):
    print("Begin Plotting")
    for x in idxs:
        ipt_sentence = test_data[x, 0]
        opt_sentence = test_data[x,1]
        #tgt_var = list(map(int,test_data[x,2]))
        seq = ipt_sentence.strip().split()
        trunc_length = 3*len(seq)
        tgt_seq = opt_sentence.strip().split()
        outputs, attention = predictor.predict(seq)
        if (len(outputs) > trunc_length ):
            outputs = outputs[:trunc_length]
            attention = attention[:trunc_length]
        if('<eos>' in outputs):
            attention = attention[:outputs.index('<eos>')]
            outputs = outputs[:outputs.index('<eos>')]

        # print(len(seq), len(outputs), attention.shape)
        # input()
        for l in range(test_data.shape[0]):
            test_data[l, 1] = ' '.join(map(str, test_data[l, 1].split()[1:]))
        # print(test_data)
        # input()
        colour = []
        for idx, inp in enumerate(seq):
            vocab_idx = grammar_vocab.index(inp) + 1
            span = outputs[idx * 3:idx * 3 + 3]
            span_str = " ".join(span)
            if (not all(
                    int(item.replace("A", "").replace("B", "").replace("C", "").split("_")[0]) == vocab_idx for item in
                    span)
                    or (not ("A" in span_str and "B" in span_str and "C" in span_str))):
                colour.extend(['r']*3)
            else:
                colour.extend(['g']*3)
        # print(colour, len(colour))
        # input()
        name = os.path.join(opt_path, 'plot' + '{}'.format(x))
        showAttention(ipt_sentence, outputs, attention, name, colour)


for level1 in model_levels:
    second_level = os.listdir(os.path.join(opt.checkpoint_path, level1))
    for level2 in second_level:
        # load model
        model_path = os.path.join(opt.checkpoint_path, level1, level2)
        model, input_vocab, output_vocab = load_model(model_path)
        predictor = Predictor(model, input_vocab, output_vocab)
        for k,sub in enumerate(test_sets):
            test_path = os.path.join(opt.test,'grammar_'+sub[0]+'.tst.full.tsv')
            test_data= prepare_data(test_path)
            if(sub[1] <= test_data.shape[0]):
                idxs = random.sample(np.arange(test_data.shape[0]).tolist(), sub[1])
            else:
                idxs = random.sample(np.arange(test_data.shape[0]).tolist(), test_data.shape[0])

            if (sub[0]=='short'):
                idxs = [3, 4, 6, 10, 16, 25, 28, 31, 37, 40]
            opt_path = os.path.join(opt.output_dir,level1,level2,sub[0])
            if not os.path.exists(opt_path):
                os.makedirs(opt_path)
            plot_attention(idxs, test_data,opt_path)

    print("finished plotting for mode={}".format(level1))
