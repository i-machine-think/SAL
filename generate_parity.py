import numpy as np

def gen(sequence_length, n_examples, output_file):
    for i in range(n_examples):
        x = np.random.randint(3, size=(sequence_length)) - 1
        unique, counts = np.unique(x, return_counts=True)

        try:
            y = dict(zip(unique, counts))[1] % 2
        except Exception:
            y = 0

        x = " ".join([str(i) for i in x])

        output_file.write("{}\t{}\n".format(x, y))

gen(sequence_length=4, n_examples=10000, output_file=open('train_parity.tsv', 'w'))
gen(sequence_length=4, n_examples=100, output_file=open('test_parity.tsv', 'w'))
