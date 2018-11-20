import torch.nn as nn
import torch.nn.functional as F

from .baseModel import Model

class Seq2seq(Model):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.
    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2seq, self).__init__(encoder_module=encoder, decoder_module=decoder, decode_function=decode_function)

    def forward(self, inputs, input_lengths=None, targets={},
                teacher_forcing_ratio=0):
        # Unpack target variables
        target_output = target_variables.get('decoder_output', None)

        encoder_outputs, encoder_hidden = self.encoder_module(inputs, input_lengths)
        result = self.decoder_module(inputs=target_output,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
