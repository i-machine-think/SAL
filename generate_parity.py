"""Generate parity sequences."""

import numpy as np

length = 64
n_train = 1
n_test = 2


def gen(sequence_length, n_examples, output_file):
    """Generate parity sequences."""
    for i in range(n_examples):
        x = np.random.randint(3, size=(sequence_length)) - 1
        unique, counts = np.unique(x, return_counts=True)

        try:
            y = dict(zip(unique, counts))[1] % 2
        except Exception:
            y = 0

        x = " ".join([str(i) for i in x])

        output_file.write("{}\t{}\n".format(x, y))

gen(sequence_length=length, n_examples=n_train, output_file=open('train_parity.tsv', 'w'))
gen(sequence_length=length, n_examples=n_test, output_file=open('test_parity.tsv', 'w'))
