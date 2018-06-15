import numpy as np

sequence_length = 64
n_examples = 10000
output_file = open('train_parity.tsv', 'w')

for i in range(n_examples):
    x = np.random.randint(3, size=(sequence_length)) - 1
    unique, counts = np.unique(x, return_counts=True)

    try:
        y = dict(zip(unique, counts))[1] % 2
    except Exception:
        y = 0

    x = " ".join([str(i) for i in x])

    output_file.write("{}\t{}\n".format(x, y))
