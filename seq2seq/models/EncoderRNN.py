import torch
import torch.nn as nn

from .baseRNN import BaseRNN
from .Ponderer import Ponderer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(BaseRNN):
    """
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        embedding_size (int): the size of the embedding of input variables
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encoder (default False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, vocab_size, max_len, hidden_size, embedding_size,
            input_dropout_p=0, dropout_p=0,
            n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False,
            ponder=False, max_ponder_steps=100, ponder_epsilon=0.01):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

        self.use_pondering = ponder
        if self.use_pondering:
            self.rnn = Ponderer(model=self.rnn, hidden_size=hidden_size, output_size=hidden_size, max_ponder_steps=max_ponder_steps, eps=ponder_epsilon, optimize=True)

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        # TODO: Ponderer currently does not support PackedSequence input. This means we will unroll and also run for <pad> inputs
        if self.variable_lengths and not self.use_pondering:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        # Unroll to support pondering
        if self.use_pondering:
            batch_size, input_length = input_var.size()
            # Init hidden0
            if self.rnn_cell == nn.LSTM:
                hidden = (
                    torch.zeros([self.n_layers, batch_size, self.hidden_size], device=device),
                    torch.zeros([self.n_layers, batch_size, self.hidden_size], device=device)
                    )
            else:
                hidden = torch.zeros([self.n_layers, batch_size, self.hidden_size], device=device)

            output_list = []
            ponder_penalties = []
            for i in range(input_length):
                output, hidden, _, ponder_penalty = self.rnn(embedded[:, i, :].unsqueeze(1), hidden)
                output_list.append(output.squeeze(1))
                ponder_penalties.append(ponder_penalty)
            output = torch.stack(output_list, dim=1)
            other = {'encoder_ponder_penalty': ponder_penalties}
        else:
            output, hidden = self.rnn(embedded)
            other = {}

        if self.variable_lengths and not self.use_pondering:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden, other
