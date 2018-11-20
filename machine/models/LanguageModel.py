
import torch.nn as nn
import torch.nn.functional as F

from .baseModel import Model

class LanguageModel(Model):
    """
    Implements a language model
    """

    def forward(self, inputs, input_lengths=None):
        raise NotImplementedError("Language model should be implemented")
     
