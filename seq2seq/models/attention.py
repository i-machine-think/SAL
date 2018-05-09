import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output
        method(str): The method to compute the alignment, mlp or dot

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
        method (torch.nn.Module): layer that implements the method of computing the attention vector

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """

    def __init__(self, dim, method):
        super(Attention, self).__init__()
        self.mask = None
        self.method = self.get_method(method, dim)

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, decoder_states, encoder_states, **kwargs):

        batch_size = decoder_states.size(0)
        decoder_states_size = decoder_states.size(2)
        input_size = encoder_states.size(1)

        # compute mask
        mask = encoder_states.eq(0.)[:, :, :1].transpose(1, 2).data

        # Compute pre-softmax attention scores. Pass on through kwargs when provided
        if kwargs:
            attn = self.method(decoder_states, encoder_states, **kwargs)
        else:
            attn = self.method(decoder_states, encoder_states)

        attn_before = attn.data.clone()

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))

        # apply local mask
        attn.data.masked_fill_(mask, -float('inf'))

        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
       
        # In the case of hard attentive guidance with variable length examples in a single batch,
        # The attention will be on the last encoder state. However, the mask will set this to -inf, which will
        # make all attention scores -inf. Taking the softmax of this results in NaNs. With the following, we set
        # all NaNs to zero.
        # Example:
        # 001 T1 T2  -> 001 101 111 EOS
        # 101 T1 PAD -> 101 110 PAD EOS
        # will have attention scores for the last decoder step:
        # -inf -inf 1
        # -inf -inf 1
        # The mask would set this to
        # -inf -inf 1
        # -inf -inf -inf
        # which results in NaNs
        attn_containing_nan = attn
        attn = attn_containing_nan.clone()
        attn.masked_fill_(attn_containing_nan != attn_containing_nan, 0)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        context = torch.bmm(attn, encoder_states)

        return context, attn

    def get_method(self, method, dim):
        """
        Set method to compute attention
        """
        if method == 'mlp':
            method = MLP(dim)
        elif method == 'concat':
            method = Concat(dim)
        elif method == 'dot':
            method = Dot()
        elif method == 'hard':
            method = HardGuidance()
        elif method == 'provided':
            method = Provided()
        else:
            return ValueError("Unknown attention method")

        return method


class Concat(nn.Module):
    """
    Implements the computation of attention by applying an
    MLP to the concatenation of the decoder and encoder
    hidden states.
    """

    def __init__(self, dim):
        super(Concat, self).__init__()
        self.mlp = nn.Linear(dim * 2, 1)

    def forward(self, decoder_states, encoder_states):
        # apply mlp to all encoder states for current decoder

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, hl_size = encoder_states.size()
        _,          dec_seqlen, _       = decoder_states.size()

        # (batch, enc_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        encoder_states_exp = encoder_states.unsqueeze(1)
        encoder_states_exp = encoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # (batch, dec_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        decoder_states_exp = decoder_states.unsqueeze(2)
        decoder_states_exp = decoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # reshape encoder and decoder states to allow batchwise computation. We will have
        # batch_size x enc_seqlen x dec_seqlen batches. So we apply the Linear
        # layer for each of them
        decoder_states_tr = decoder_states_exp.contiguous().view(-1, hl_size)
        encoder_states_tr = encoder_states_exp.contiguous().view(-1, hl_size)

        mlp_input = torch.cat((encoder_states_tr, decoder_states_tr), dim=1)

        # apply mlp and respape to get in correct form
        mlp_output = self.mlp(mlp_input)
        attn = mlp_output.view(batch_size, dec_seqlen, enc_seqlen)

        return attn


class Dot(nn.Module):

    def __init__(self):
        super(Dot, self).__init__()

    def forward(self, decoder_states, encoder_states):
        attn = torch.bmm(decoder_states, encoder_states.transpose(1, 2))
        return attn


class MLP(nn.Module):

    def __init__(self, dim):
        super(MLP, self).__init__()
        self.mlp = nn.Linear(dim * 2, dim)
        self.activation = nn.ReLU()
        self.out = nn.Linear(dim, 1)

    def forward(self, decoder_states, encoder_states):
        # apply mlp to all encoder states for current decoder

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, hl_size = encoder_states.size()
        _,          dec_seqlen, _       = decoder_states.size()

        # (batch, enc_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        encoder_states_exp = encoder_states.unsqueeze(1)
        encoder_states_exp = encoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # (batch, dec_seqlen, hl_size) -> (batch, dec_seqlen, enc_seqlen, hl_size)
        decoder_states_exp = decoder_states.unsqueeze(2)
        decoder_states_exp = decoder_states_exp.expand(batch_size, dec_seqlen, enc_seqlen, hl_size)

        # reshape encoder and decoder states to allow batchwise computation. We will have
        # batch_size x enc_seqlen x dec_seqlen batches. So we apply the Linear
        # layer for each of them
        decoder_states_tr = decoder_states_exp.contiguous().view(-1, hl_size)
        encoder_states_tr = encoder_states_exp.contiguous().view(-1, hl_size)

        mlp_input = torch.cat((encoder_states_tr, decoder_states_tr), dim=1)

        # apply mlp and reshape to get in correct form
        mlp_output = self.mlp(mlp_input)
        mlp_output = self.activation(mlp_output)
        out = self.out(mlp_output)
        attn = out.view(batch_size, dec_seqlen, enc_seqlen)

        return attn


class HardGuidance(nn.Module):

    """
    Hard attentive guidance module for the lookup table task (diagonal attentive guidance)
    """

    def forward(self, decoder_states, encoder_states, **kwargs):
        """
        Forward pass method. Computes the hard, non-differentiable attentive guidance for
        the lookup tables task. Or any other task that requires a diagonal attentive guidance pattern.

        Args:
            decoder_states (torch.autograd.Variable): Variable containing one or multiple decoder hidden states (batch_size X dec_seqlen X hidden_size)
            encoder_states (torch.autograd.Variable): Variable containing all encoder hidden states (batch_size X enc_seqlen X hidden_size)
            step (int): Current step of the decoder. Set to -1 if the decoder is not unrolled.

        Returns:
            attn: The attention distribution over the encoder states for each of the provided decoder states (batch_size X dec_seqlen X enc_seqlen)
        """
        step = kwargs['step']

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, _ = encoder_states.size()
        _,          dec_seqlen, _ = decoder_states.size()

        # Add hard guidance attention vector for all decoder steps
        if step == -1:
            # Initialize attention vectors. These are the pre-softmax scores, so any
            # -inf will become 0 (if there is at least one value not -inf)
            attn = torch.zeros(batch_size, dec_seqlen, enc_seqlen).fill_(-float('inf'))

            # For decoder step 'step' we will attend to encoder step 'step'. So we generate a diagonal attentive guidance matrix with torch.arange
            # However, if the decoder EOS is present, and the input EOS is not, the decoder will be longer than the encoder.
            # In this case, we will attend the output EOS to the last input token.
            # When doing inference, the decoder will be run for max_len (50) and is often much longer than the encoder. In
            # this case we will also attend these extra decoder steps to the encoders last state. However, these should be ignored for calculating
            # the loss and metrics.
            indices = torch.arange(enc_seqlen).view(1, enc_seqlen, 1)
            if dec_seqlen > enc_seqlen:
                indices = torch.cat((
                    indices,
                    (enc_seqlen - 1) * torch.ones(1, dec_seqlen - enc_seqlen, 1)), dim=1)

            indices = indices.expand(batch_size, dec_seqlen, 1).long()

            # Fill the attention guidance with 1's
            attn = attn.scatter_(dim=2, index=indices, value=1)

        # Add guidance attention vector for only one single unrolled decoder step
        else:
            # Step should only be passed if we are truly unrolling the decoder and processing it
            # one step at a time.
            assert dec_seqlen == 1, "Decoder must be unrolled if 'step' is passed"

            # Initialize attention vectors. These are the pre-softmax scores, so any
            # -inf will become 0 (if there is at least one value not -inf)
            attn = torch.zeros(batch_size, 1, enc_seqlen).fill_(-float('inf'))

            # For decoder step 'step' we will attend to encoder step 'step'.
            # However, if the decoder EOS is present, and the input EOS is not, the decoder will be longer than the encoder.
            # In this case, we will attend the output EOS to the last input token.
            # When doing inference, the decoder will be run for max_len (50) and is often much longer than the encoder. In
            # this case we will also attend these extra decoder steps to the encoders last state. However, these should be ignored for calculating
            # the loss and metrics.
            if step < enc_seqlen:
                # Fill the attention vectors with a 1 at the specified indices/step
                indices = step * torch.ones(batch_size, 1, 1).long()
            else:
                indices = (enc_seqlen - 1) * torch.ones(batch_size, 1, 1).long()

            attn = attn.scatter_(dim=2, index=indices, value=1)

        # Convert into non-grad Variable
        attn = torch.autograd.Variable(attn, requires_grad=False)

        return attn

class Provided(nn.Module):
    """
    Attention method / attentive guidance method for data sets that are annotated with attentive guidance.
    """

    def forward(self, decoder_states, encoder_states, step, input_lengths, provided_attention):
        """
        Forward method that receives provided attentive guidance indices and returns proper
        attention scores vectors.

        Args:
            decoder_states (torch.autograd.Variable): Hidden layer of all decoder states (batch, dec_seqlen, hl_size)
            encoder_states (torch.autograd.Variable): Output layer of all encoder states (batch, dec_seqlen, hl_size)
            step (int): The current decoder step for unrolled RNN. Set to -1 for rolled RNN
            input_lengths (list(int)): The input length for all elements in the batch
            provided_attention (torch.autograd.Variable): Variable containing the provided attentive guidance indices (batch, max_provided_attention_length)

        Returns:
            torch.autograd.Variable: Attention score vectors (batch, dec_seqlen, hl_size)
        """

        # decoder_states --> (batch, dec_seqlen, hl_size)
        # encoder_states --> (batch, enc_seqlen, hl_size)
        batch_size, enc_seqlen, _ = encoder_states.size()
        _,          dec_seqlen, _ = decoder_states.size()

        input_lengths = torch.LongTensor(input_lengths)
        provided_attention = provided_attention.data

        last_encoder_state_indices = input_lengths - 1

        max_input_length = max(input_lengths)
        max_provided_attention_length = provided_attention.size(1)

        # In the case of rolled RNN
        if step == -1:
            attention_indices = provided_attention

            # If there are more decoder states than attentions provided, we extend the provided attentions.
            # These states will all attend to the last encoder state.
            if dec_seqlen > max_provided_attention_length:
                filler = -1 * torch.ones(batch_size, dec_seqlen -
                                         max_provided_attention_length).long()
                attention_indices = torch.cat([attention_indices, filler], dim=1)

                # Replace no_attention_provided spots with last_encoder_state
                no_attention_provided_mask = attention_indices.eq(-1)
                # Match dimension of attention_indices and no_attention_provided_mask
                last_encoder_state_indices = last_encoder_state_indices.unsqueeze(
                    1).expand(-1, dec_seqlen)
                attention_indices[no_attention_provided_mask] = last_encoder_state_indices[
                    no_attention_provided_mask]

            attention_indices = attention_indices.unsqueeze(2)

        # In the case of unrolled RNN
        else:
            # If there are more decoder states than attention provided, we always attend to the last
            # encoder state. In the case that we have input and output EOS, this will make sure that
            # the output EOS will attend to the input EOS. In other cases, these attentions should be
            # ignored for loss and metric calculations, if implemented correctly.
            if step + 1 > max_provided_attention_length:
                attention_indices = last_encoder_state_indices

            # We attend to the encoder state provided by provided_attention. If some examples in the batch
            # are shorter, we attend to the last encoder state. These should be ignored however for loss and
            # metric calculations
            else:
                attention_indices = provided_attention[:, step].clone()

                no_attention_provided_mask = provided_attention[:, step].eq(-1)

                attention_indices[no_attention_provided_mask] = last_encoder_state_indices[
                    no_attention_provided_mask]

            # Add two dimensions
            attention_indices = attention_indices.unsqueeze(1).unsqueeze(2)

        # Initialize attention vectors. These are the pre-softmax scores, so any
        # -inf will become 0 (if there is at least one value not -inf)
        attention_scores = torch.zeros(batch_size, dec_seqlen, enc_seqlen).fill_(-float('inf'))
        attention_scores = attention_scores.scatter_(dim=2, index=attention_indices, value=1)
        attention_scores = torch.autograd.Variable(attention_scores, requires_grad=False)

        return attention_scores
