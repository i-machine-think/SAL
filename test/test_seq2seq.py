import unittest


from machine.models.EncoderRNN import EncoderRNN
from machine.models.DecoderRNN import DecoderRNN
from machine.models.seq2seq import Seq2seq


class TestSeq2seq(unittest.TestCase):
    def setUp(self):
        self.decoder = DecoderRNN(100, 50, 16, 0, 1, input_dropout_p=0)
        self.encoder = EncoderRNN(100, 10, 50, 16, n_layers=2, dropout_p=0.5)

    def test_standard_init(self):
        Seq2seq(self.encoder, self.decoder)
        Seq2seq(self.encoder, self.decoder, uniform_init=-1)

    def test_uniform_init(self):
        Seq2seq(self.encoder, self.decoder, uniform_init=1)

    def test_xavier_init(self):
        Seq2seq(self.encoder, self.decoder, init_glorot=True)

    def test_uniform_xavier_init(self):
        Seq2seq(self.encoder, self.decoder, uniform_init=1, init_glorot=True)


if __name__ == '__main__':
    unittest.main()
