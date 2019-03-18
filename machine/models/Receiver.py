import torch
import torch.nn as nn

from .baseRNN import BaseRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Receiver(BaseRNN):
	"""
		Applies a rnn to a message produced by a Sender.

		Args:
			vocab_size (int): size of the vocabulary
			embedding_size (int): the size of the embedding of input variables
			hidden_size (int): the size of the hidden dimension of the rnn
			rnn_cell (str, optional): type of RNN cell (default: gru)

		Inputs:
		    m (torch.tensor): The message produced by the Sender. Shape [batch_size, max_seq_len]

		Outputs:
		    output (torch.tensor): The batch of the appended hidden states at each timestep.
		    state (torch.tensor): (h,c) of the last timestep if LSTM, h if GRU

	"""

	def __init__(self, vocab_size, embedding_size, hidden_size, rnn_cell='gru'):
		super().__init__(vocab_size, -1, hidden_size,
						 input_dropout_p=0, dropout_p=0,
						 n_layers=1, rnn_cell=rnn_cell)

		self.rnn = self.rnn_cell(embedding_size, hidden_size, num_layers=1, batch_first=True)
		self.embedding = nn.Parameter(torch.empty((vocab_size, embedding_size), dtype=torch.float32))

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.normal_(self.embedding, 0.0, 0.1)

		nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
		nn.init.orthogonal_(self.rnn.weight_hh_l0)
		nn.init.constant_(self.rnn.bias_ih_l0, val=0)
		# cuDNN bias order: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
		# add some positive bias for the forget gates [b_i, b_f, b_o, b_g] = [0, 1, 0, 0]
		nn.init.constant_(self.rnn.bias_hh_l0, val=0)
		nn.init.constant_(self.rnn.bias_hh_l0[self.hidden_size:2 * self.hidden_size], val=1)

	def forward(self, m):
		batch_size = m.shape[0]

		# h0
		h = torch.zeros([batch_size, self.hidden_size], device=device)

		if type(self.rnn) is nn.LSTM:
			# c0
			c = torch.zeros([batch_size, self.hidden_size], device=device)

			state = (h[None, ...], c[None, ...])

		else:
			state = h[None, ...]

		emb = torch.matmul(m, self.embedding) if self.training else self.embedding[m]
		return self.rnn(emb, state)