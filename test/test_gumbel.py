import unittest
import torch

from machine.util.gumbel import gumbel_softmax


class TestGumbelSoftmax(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.probs = torch.tensor([
            [0.3, 0.4, 0.1, 0.1, 0.1],
            [0.1, 0.3, 0.2, 0.3, 0.1],
            [0.6, 0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2]])
        self.tau = 1.2

    def test_hard(self):
        res = gumbel_softmax(self.probs, self.tau, hard=True)

        self.assertEqual(res.shape, (4, 5))
        for r in res:
            self.assertTrue(r.sum(), 1.0)

    def test_soft(self):
        res = gumbel_softmax(self.probs, self.tau, hard=False)

        self.assertEqual(res.shape, (4, 5))

    