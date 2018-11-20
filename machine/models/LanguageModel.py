
import torch.nn as nn
import torch.nn.functional as F

from .baseModel import Model

class LanguageModel(Model):
    """
    Implements a language model
    """
    
    def flatten_parameters(self):
        raise NotImplementedError()

    def forward(self, inputs, input_lengths=None):
        raise NotImplementedError("Language model should be implemented")
     
