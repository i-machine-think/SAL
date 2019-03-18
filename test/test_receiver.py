import unittest
import mock

import torch

from machine.models.Receiver import Receiver

class TestReceiver(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.vocab_size = 4
        self.embedding_size = 8
        self.hidden_size = 16
        self.seq_len = 5

    def test_lstm_train(self):
        receiver = Receiver(self.vocab_size, self.embedding_size,
                     self.hidden_size, rnn_cell='lstm')

        batch_size = 2
        m = torch.rand([batch_size, self.seq_len, self.vocab_size])

        receiver.train()

        outputs, (h,c) = receiver(m)

        self.assertEqual(outputs.shape, (batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(h.squeeze().shape, (batch_size, self.hidden_size))
        self.assertEqual(c.squeeze().shape, (batch_size, self.hidden_size))

    def test_gru_train(self):
        receiver = Receiver(self.vocab_size, self.embedding_size,
                     self.hidden_size, rnn_cell='gru')

        batch_size = 2
        m = torch.rand([batch_size, self.seq_len, self.vocab_size])

        receiver.train()

        outputs, h = receiver(m)

        self.assertEqual(outputs.shape, (batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(h.squeeze().shape, (batch_size, self.hidden_size))

    def test_lstm_eval(self):
        receiver = Receiver(self.vocab_size, self.embedding_size,
                     self.hidden_size, rnn_cell='lstm')

        batch_size = 2
        m = torch.randint(high=self.vocab_size, size=(batch_size, self.seq_len))

        receiver.eval()

        outputs, (h,c) = receiver(m)

        self.assertEqual(outputs.shape, (batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(h.squeeze().shape, (batch_size, self.hidden_size))
        self.assertEqual(c.squeeze().shape, (batch_size, self.hidden_size))

    def test_gru_eval(self):
        receiver = Receiver(self.vocab_size, self.embedding_size,
                     self.hidden_size, rnn_cell='gru')

        batch_size = 2
        m = torch.randint(high=self.vocab_size, size=(batch_size, self.seq_len))

        receiver.eval()

        outputs, h = receiver(m)

        self.assertEqual(outputs.shape, (batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(h.squeeze().shape, (batch_size, self.hidden_size))

