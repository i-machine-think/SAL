import torch.nn.functional as F
import torch.nn as nn

from .baseModel import BaseModel


class Seq2seq(BaseModel):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.
    """

    def __init__(self, encoder, decoder, decode_function=F.log_softmax,
                 uniform_init=0, glorot_init=False):
        super(Seq2seq, self).__init__(encoder_module=encoder,
                                      decoder_module=decoder,
                                      decode_function=decode_function)
        # Initialize Weights
        self._init_weights(uniform_init, glorot_init)

    def flatten_parameters(self):
        """
        Flatten parameters of all components in the model.
        """
        self.encoder_module.rnn.flatten_parameters()
        self.decoder_module.rnn.flatten_parameters()

    def forward(self, inputs, input_lengths=None, targets={},
                teacher_forcing_ratio=0):
        # Unpack target variables
        target_output = targets.get('decoder_output', None)

        encoder_outputs, encoder_hidden = self.encoder_module(
            inputs, input_lengths=input_lengths)
        result = self.decoder_module(inputs=target_output,
                                     encoder_hidden=encoder_hidden,
                                     encoder_outputs=encoder_outputs,
                                     function=self.decode_function,
                                     teacher_forcing_ratio=teacher_forcing_ratio)
        return result

    def _init_weights(self, uniform_init=0.0, glorot_init=False):
        # initialize weights using uniform distribution
        if uniform_init > 0.0:
            for p in self.parameters():
                p.data.uniform_(-uniform_init, uniform_init)

        # xavier/glorot initialization if glorot_init
        if glorot_init:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
