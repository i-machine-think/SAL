import unittest
import mock

import torch

from machine.models.Sender import Sender

class TestSender(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.vocab_size = 4
        self.max_len = 5
        self.embedding_size = 8
        self.hidden_size = 16
        self.sos_id = 0
        self.eos_id = self.vocab_size - 1
        self.tau = 1.2

    @mock.patch('machine.models.Sender.gumbel_softmax')
    def test_lstm_hidden_train(self, mock_gumbel):
        sender = Sender(self.vocab_size, self.max_len, self.embedding_size,
                     self.hidden_size, self.sos_id, self.eos_id, 
                     rnn_cell='lstm', greedy=True)

        batch_size = 2
        n_sos_tokens = 1
        hidden_state = torch.rand([batch_size, self.hidden_size])

        sender.train()

        # One per max_len (2 elements in batch - a and b)
        a1 = [1.0, 0.0, 0.0, 0.0]
        a2 = [1.0, 0.0, 0.0, 0.0]
        a3 = [0.0, 1.0, 0.0, 0.0]
        a4 = [0.0, 0.0, 1.0, 0.0]
        a5 = [0.0, 0.0, 0.0, 1.0]

        b1 = [0.0, 1.0, 0.0, 0.0]
        b2 = [0.0, 0.0, 1.0, 0.0]
        b3 = [0.0, 0.0, 0.0, 1.0]
        b4 = [0.0, 0.0, 1.0, 0.0]
        b5 = [1.0, 0.0, 0.0, 0.0]

        # Return max_len times a tensor with shape [batch_size, vocab_size]
        mock_gumbel.side_effect = ([
            torch.tensor([a1, b1]),
            torch.tensor([a2, b2]),
            torch.tensor([a3, b3]),
            torch.tensor([a4, b4]),
            torch.tensor([a5, b5]) 
            ])

        res, seq_lengths = sender(self.tau, hidden_state)

        self.assertEqual(mock_gumbel.call_count, self.max_len)

        self.assertEqual(res.shape[0], batch_size)
        self.assertEqual(res.shape[1], self.max_len + n_sos_tokens)

        self.assertTrue(torch.all(torch.eq(
            res[0], 
            torch.tensor([[1.0, 0.0, 0.0, 0.0], # sos token
                        a1,
                        a2,
                        a3,
                        a4,
                        a5
                        ]))))

        self.assertTrue(torch.all(torch.eq(
            res[1],
            torch.tensor([[1.0, 0.0, 0.0, 0.0], # sos token
                        b1,
                        b2,
                        b3,
                        b4,
                        b5
                        ]))))

        self.assertEqual(seq_lengths[0], 6)
        self.assertEqual(seq_lengths[1], 4)

    @mock.patch('machine.models.Sender.gumbel_softmax')
    def test_gru_hidden_train(self, mock_gumbel):
        sender = Sender(self.vocab_size, self.max_len, self.embedding_size,
                     self.hidden_size, self.sos_id, self.eos_id, 
                     rnn_cell='gru', greedy=True)

        batch_size = 2
        n_sos_tokens = 1
        hidden_state = torch.rand([batch_size, self.hidden_size])

        sender.train()

        # One per max_len (2 elements in batch - a and b)
        a1 = [1.0, 0.0, 0.0, 0.0]
        a2 = [1.0, 0.0, 0.0, 0.0]
        a3 = [0.0, 1.0, 0.0, 0.0]
        a4 = [0.0, 0.0, 1.0, 0.0]
        a5 = [0.0, 0.0, 0.0, 1.0]

        b1 = [0.0, 1.0, 0.0, 0.0]
        b2 = [0.0, 0.0, 1.0, 0.0]
        b3 = [0.0, 0.0, 0.0, 1.0]
        b4 = [0.0, 0.0, 1.0, 0.0]
        b5 = [1.0, 0.0, 0.0, 0.0]

        # Return max_len times a tensor with shape [batch_size, vocab_size]
        mock_gumbel.side_effect = ([
            torch.tensor([a1, b1]),
            torch.tensor([a2, b2]),
            torch.tensor([a3, b3]),
            torch.tensor([a4, b4]),
            torch.tensor([a5, b5]) 
            ])

        res, seq_lengths = sender(self.tau, hidden_state)

        self.assertEqual(mock_gumbel.call_count, self.max_len)

        self.assertEqual(res.shape[0], batch_size)
        self.assertEqual(res.shape[1], self.max_len + n_sos_tokens)

        self.assertTrue(torch.all(torch.eq(
            res[0], 
            torch.tensor([[1.0, 0.0, 0.0, 0.0], # sos token
                        a1,
                        a2,
                        a3,
                        a4,
                        a5
                        ]))))

        self.assertTrue(torch.all(torch.eq(
            res[1],
            torch.tensor([[1.0, 0.0, 0.0, 0.0], # sos token
                        b1,
                        b2,
                        b3,
                        b4,
                        b5
                        ]))))

        self.assertEqual(seq_lengths[0], 6)
        self.assertEqual(seq_lengths[1], 4)

    @mock.patch('machine.models.Sender.gumbel_softmax')
    def test_lstm_not_hidden_train(self, mock_gumbel):
        sender = Sender(self.vocab_size, self.max_len, self.embedding_size,
                     self.hidden_size, self.sos_id, self.eos_id, 
                     rnn_cell='lstm', greedy=True)

        batch_size = 1
        n_sos_tokens = 1

        sender.train()

        # One per max_len (1 element in batch - a)
        a1 = [0.0, 0.0, 0.0, 1.0]
        a2 = [1.0, 0.0, 0.0, 0.0]
        a3 = [0.0, 1.0, 0.0, 0.0]
        a4 = [0.0, 0.0, 1.0, 0.0]
        a5 = [0.0, 0.0, 0.0, 1.0]

        # Return max_len times a tensor with shape [batch_size, vocab_size]
        mock_gumbel.side_effect = ([
            torch.tensor([a1]),
            torch.tensor([a2]),
            torch.tensor([a3]),
            torch.tensor([a4]),
            torch.tensor([a5]) 
            ])

        res, seq_lengths = sender(self.tau)

        self.assertEqual(mock_gumbel.call_count, self.max_len)

        self.assertEqual(res.shape[0], batch_size)
        self.assertEqual(res.shape[1], self.max_len + n_sos_tokens)

        self.assertTrue(torch.all(torch.eq(
            res[0], 
            torch.tensor([[1.0, 0.0, 0.0, 0.0], # sos token
                        a1,
                        a2,
                        a3,
                        a4,
                        a5
                        ]))))

        self.assertEqual(seq_lengths[0], 2)

    @mock.patch('machine.models.Sender.F.softmax')
    def test_lstm_hidden_eval_greedy(self, mock_softmax):
        sender = Sender(self.vocab_size, self.max_len, self.embedding_size,
                     self.hidden_size, self.sos_id, self.eos_id, 
                     rnn_cell='lstm', greedy=True)

        batch_size = 2
        n_sos_tokens = 1
        hidden_state = torch.rand([batch_size, self.hidden_size])

        # Return max_len times a tensor with shape [batch_size, vocab_size]
        mock_softmax.side_effect = [
            torch.tensor([[0.3, 0.4, 0.1, 0.2], [0.4, 0.3, 0.1, 0.2]]),
            torch.tensor([[0.2, 0.2, 0.2, 0.4], [0.1, 0.4, 0.2, 0.3]]),
            torch.tensor([[0.6, 0.4, 0.0, 0.0], [0.5, 0.3, 0.1, 0.2]]),
            torch.tensor([[0.4, 0.3, 0.0, 0.3], [0.8, 0.1, 0.1, 0.0]]),
            torch.tensor([[0.1, 0.1, 0.1, 0.7], [0.0, 0.5, 0.2, 0.3]])
            ]

        sender.eval()

        res, seq_lengths = sender(self.tau, hidden_state)

        self.assertEqual(res.shape[0], batch_size)
        self.assertEqual(res.shape[1], self.max_len + n_sos_tokens)

        self.assertTrue(torch.all(torch.eq(
            res[0], 
            torch.tensor([self.sos_id,
                        1,
                        3,
                        0,
                        0,
                        3
                        ]))))

        self.assertTrue(torch.all(torch.eq(
            res[1],
            torch.tensor([self.sos_id,
                        0,
                        1,
                        0,
                        0,
                        1
                        ]))))

        self.assertEqual(seq_lengths[0], 3)
        self.assertEqual(seq_lengths[1], 6)

    @mock.patch('machine.models.Sender.F.softmax')
    def test_lstm_hidden_eval_not_greedy(self, mock_softmax):
        sender = Sender(self.vocab_size, self.max_len, self.embedding_size,
                     self.hidden_size, self.sos_id, self.eos_id, 
                     rnn_cell='lstm', greedy=False)

        batch_size = 2
        n_sos_tokens = 1
        hidden_state = torch.rand([batch_size, self.hidden_size])

        # Return max_len times a tensor with shape [batch_size, vocab_size]
        mock_softmax.side_effect = [
            torch.tensor([[0.3, 0.4, 0.1, 0.2], [0.4, 0.3, 0.1, 0.2]]),
            torch.tensor([[0.2, 0.2, 0.2, 0.4], [0.1, 0.4, 0.2, 0.3]]),
            torch.tensor([[0.6, 0.4, 0.0, 0.0], [0.5, 0.3, 0.1, 0.2]]),
            torch.tensor([[0.4, 0.3, 0.0, 0.3], [0.8, 0.1, 0.1, 0.0]]),
            torch.tensor([[0.1, 0.1, 0.1, 0.7], [0.0, 0.5, 0.2, 0.3]])
            ]

        sender.eval()

        res, seq_lengths = sender(self.tau, hidden_state)

        self.assertEqual(mock_softmax.call_count, self.max_len)

        self.assertEqual(res.shape[0], batch_size)
        self.assertEqual(res.shape[1], self.max_len + n_sos_tokens)

        self.assertEqual(res[0][0], self.sos_id)
        self.assertEqual(res[1][0], self.sos_id)

    @mock.patch('machine.models.Sender.F.softmax')
    def test_lstm_not_hidden_eval_greedy(self, mock_softmax):
        sender = Sender(self.vocab_size, self.max_len, self.embedding_size,
                     self.hidden_size, self.sos_id, self.eos_id, 
                     rnn_cell='lstm', greedy=True)

        batch_size = 1
        n_sos_tokens = 1

        # Return max_len times a tensor with shape [batch_size, vocab_size]
        mock_softmax.side_effect = [
            torch.tensor([0.3, 0.2, 0.1, 0.4]),
            torch.tensor([0.4, 0.3, 0.1, 0.2]),
            torch.tensor([0.2, 0.4, 0.2, 0.2]),
            torch.tensor([0.2, 0.5, 0.1, 0.2]),
            torch.tensor([0.0, 0.0, 0.8, 0.2]),
            ]

        sender.eval()

        res, seq_lengths = sender(self.tau)

        self.assertEqual(mock_softmax.call_count, self.max_len)

        self.assertEqual(res.shape[0], batch_size)
        self.assertEqual(res.shape[1], self.max_len + n_sos_tokens)

        self.assertTrue(torch.all(torch.eq(
            res[0], 
            torch.tensor([self.sos_id,
                        3,
                        0,
                        1,
                        1,
                        2
                        ]))))

        self.assertEqual(seq_lengths[0], 2)

    @mock.patch('machine.models.Sender.F.softmax')
    def test_lstm_not_hidden_eval_not_greedy(self, mock_softmax):
        sender = Sender(self.vocab_size, self.max_len, self.embedding_size,
                     self.hidden_size, self.sos_id, self.eos_id, 
                     rnn_cell='lstm', greedy=False)

        batch_size = 1
        n_sos_tokens = 1

        # Return max_len times a tensor with shape [batch_size, vocab_size]
        mock_softmax.side_effect = [
            torch.tensor([0.3, 0.2, 0.1, 0.4]),
            torch.tensor([0.4, 0.3, 0.1, 0.2]),
            torch.tensor([0.2, 0.4, 0.2, 0.2]),
            torch.tensor([0.2, 0.5, 0.1, 0.2]),
            torch.tensor([0.0, 0.0, 0.8, 0.2]),
            ]

        sender.eval()

        res, seq_lengths = sender(self.tau)

        self.assertEqual(mock_softmax.call_count, self.max_len)

        self.assertEqual(res.shape[0], batch_size)
        self.assertEqual(res.shape[1], self.max_len + n_sos_tokens)

        self.assertEqual(res[0][0], self.sos_id)

    @mock.patch('machine.models.Sender.F.softmax')
    def test_gru_hidden_eval_greedy(self, mock_softmax):
        sender = Sender(self.vocab_size, self.max_len, self.embedding_size,
                     self.hidden_size, self.sos_id, self.eos_id, 
                     rnn_cell='gru', greedy=True)

        batch_size = 2
        n_sos_tokens = 1
        hidden_state = torch.rand([batch_size, self.hidden_size])

        # Return max_len times a tensor with shape [batch_size, vocab_size]
        mock_softmax.side_effect = [
            torch.tensor([[0.3, 0.4, 0.1, 0.2], [0.4, 0.3, 0.1, 0.2]]),
            torch.tensor([[0.2, 0.2, 0.2, 0.4], [0.1, 0.4, 0.2, 0.3]]),
            torch.tensor([[0.6, 0.4, 0.0, 0.0], [0.5, 0.3, 0.1, 0.2]]),
            torch.tensor([[0.4, 0.3, 0.0, 0.3], [0.8, 0.1, 0.1, 0.0]]),
            torch.tensor([[0.1, 0.1, 0.1, 0.7], [0.0, 0.5, 0.2, 0.3]])
            ]

        sender.eval()

        res, seq_lengths = sender(self.tau, hidden_state)

        self.assertEqual(res.shape[0], batch_size)
        self.assertEqual(res.shape[1], self.max_len + n_sos_tokens)

        self.assertTrue(torch.all(torch.eq(
            res[0], 
            torch.tensor([self.sos_id,
                        1,
                        3,
                        0,
                        0,
                        3
                        ]))))

        self.assertTrue(torch.all(torch.eq(
            res[1],
            torch.tensor([self.sos_id,
                        0,
                        1,
                        0,
                        0,
                        1
                        ]))))

        self.assertEqual(seq_lengths[0], 3)
        self.assertEqual(seq_lengths[1], 6)
