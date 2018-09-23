"""Generate sort sequences of original ACT paper."""

import numpy as np
from random import randint

min_seq_len = 2
max_seq_len = 5
n_train = 10000
n_test = 100


def generate(min_seq_len, max_seq_len, n_examples, output_file):
    """Generate sort sequences of original ACT paper."""
    for i in range(n_examples):
        sequence_length = randint(min_seq_len, max_seq_len)

        x = np.random.randn(sequence_length)
        y = np.argsort(x, axis=0)

        x = " ".join([str(i) for i in x])
        y = " ".join([str(i) for i in y])

        output_file.write("{}\t{}\n".format(x, y))


generate(min_seq_len=min_seq_len,
         max_seq_len=max_seq_len,
         n_examples=n_train,
         output_file=open('train_sort.tsv', 'w'))

generate(min_seq_len=min_seq_len,
         max_seq_len=max_seq_len,
         n_examples=n_test,
         output_file=open('test_sort.tsv', 'w'))
