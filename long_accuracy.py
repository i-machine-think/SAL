import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torchtext

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.dataset import SourceField, TargetField, AttentionField
from seq2seq.loss import AttentionLoss, NLLLoss
from seq2seq.metrics import WordAccuracy, SequenceAccuracy, FinalTargetAccuracy
from seq2seq.trainer import LookupTablePonderer

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
parser.add_argument('--test', help='Path to test data')
parser.add_argument('--output_dir', help='Path to save results')
parser.add_argument('--model_type', help='pre_rnn or pre_rnn_baseline etc.')
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--use_input_eos', action='store_true', help='EOS symbol in input sequences is not used by default. Use this flag to enable.')
parser.add_argument('--ignore_output_eos', action='store_true', help='Ignore end of sequence token during training and evaluation')
parser.add_argument('--pondering', action='store_true')
parser.add_argument('--use_attention_loss', action='store_true')
parser.add_argument('--eval_batch_size', type=int, help='Batch size', default=128)
parser.add_argument('--xent_loss', type=float, default=1.)
parser.add_argument('--scale_attention_loss', type=float, default=1.)

opt = parser.parse_args()
final_level = opt.model_type
print(opt.use_attention_loss)

if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

use_output_eos = not opt.ignore_output_eos
src = SourceField(use_input_eos=opt.use_input_eos)
tgt = TargetField(include_eos=use_output_eos)
max_len = opt.max_len
IGNORE_INDEX=-1

def get_batch_data(batch):
    input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
    target_variables = {'decoder_output': getattr(batch, seq2seq.tgt_field_name)}

    # If available, also get provided attentive guidance data
    if hasattr(batch, seq2seq.attn_field_name):
        attention_target = getattr(batch, seq2seq.attn_field_name)
        target_variables['attention_target'] = attention_target

    return input_variables, input_lengths, target_variables


def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

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

def loss_acc():
    # Prepare loss and metrics
    pad = output_vocab.stoi[tgt.pad_token]
    loss = [NLLLoss(ignore_index=pad)]
    # loss_weights = [1.]
    loss_weights = [float(opt.xent_loss)]

    if opt.use_attention_loss:
        loss.append(AttentionLoss(ignore_index=IGNORE_INDEX))
        loss_weights.append(opt.scale_attention_loss)

    metrics = [WordAccuracy(ignore_index=pad), SequenceAccuracy(ignore_index=pad),
               FinalTargetAccuracy(ignore_index=pad, eos_id=tgt.eos_id)]
    if torch.cuda.is_available():
        for loss_func in loss:
            loss_func.cuda()

    ponderer = None
    if opt.pondering:
        ponderer = LookupTablePonderer(input_eos_used=opt.use_input_eos, output_eos_used=use_output_eos)

    return (loss,loss_weights, metrics, ponderer)


def prepare_data(data_path):
    # generate training and testing data
    tabular_data_fields = [('src', src), ('tgt', tgt)]
    if opt.use_attention_loss:
        attn = AttentionField(use_vocab=False, ignore_index=IGNORE_INDEX)
        tabular_data_fields.append(('attn', attn))
    gen_data = torchtext.data.TabularDataset(
        path=data_path, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    )
    return gen_data

for i in range(2, 6):
    level1 = 'sample{}'.format(i)
    for j in range(1, 3):
        level2 = 'run{}'.format(j)
        #for level3 in final_level:
            # load model
        level3 = final_level
        model_path = os.path.join(opt.checkpoint_path, level1, level2, level3)
        level4 = os.listdir(model_path)[0]
        level5 = os.listdir(os.path.join(model_path, level4))[0]
        final_model_path = os.path.join(model_path, level4, level5)
        model, input_vocab, output_vocab = load_model(final_model_path)
        loss, loss_weights, metrics, ponderer = loss_acc()
        evaluator = Evaluator(loss=loss, metrics=metrics, batch_size=opt.eval_batch_size)
        test_files = os.listdir(os.path.join(opt.test,level1))
        temp_arr = np.zeros((len(test_files),3), dtype=object)
        for k ,test_data in enumerate(test_files):
            data = prepare_data(os.path.join(opt.test, level1, test_data))
            _, accuracies = evaluator.evaluate(model, data, get_batch_data, ponderer=ponderer)
            name1 = test_data.split('.')[0]
            if(int(name1[-1]) != 0):
                temp_arr[k,0] = int(name1[-1])
                temp_arr[k,1] = name1[:-1]
            else:
                temp_arr[k,0] = int(name1[-2:])
                temp_arr[k,1] = name1[:-2]
            temp_arr[k,2] = accuracies[1].get_val()
        df = pd.DataFrame(temp_arr)
        opt_path = os.path.join(opt.output_dir, level1, level2, level3, level4)
        if not os.path.exists(opt_path):
            os.makedirs(opt_path)
        df.to_csv(os.path.join(opt_path,'longer.tsv'), sep='\t', header=False, index=False)
    print("finished dumping data for sample{}".format(i))
