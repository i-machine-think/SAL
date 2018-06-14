"""
- Forward pass expects input and hidden as first arguments. There may be more model-specific arguments
- Forward pass of underlying model should return model_output, hidden (and optionally extra return values)
- 'model_output' refers not necessarily to the output of the RNN, but the final output.
- Currently does not work for packed_sequence or for rolled-up RNN.
  Should be possible to implement for rolled RNN as well (and maybe also packed sequence), but we must be really careful with parameters and return values
  For example, we now pass args and kwargs directly to the underlying model, but maybe they need to be sliced when we would unroll the underlying model.
  Might be best to just only allow unrolled behaviour
- Assumes batch_first=True for now

Returns model_output, hidden, and a list of all extra return values
"""

import torch
import torch.nn as nn

class Ponderer(nn.Module):
    def __init__(self, model, hidden_size, ponder_steps):
        super(Ponderer, self).__init__()

        self.model = model
        self.hidden_size = hidden_size
        self.ponder_steps = ponder_steps

    # Automatic delegation of undefined attributes.
    # First we check whether the parent class nn.Module has implemented this function
    # If not, we delegate the call to the model
    def __getattr__(self, attr):
        try:
            return super(Ponderer, self).__getattr__(attr)
        except AttributeError:
            return self.model.__getattr__(attr)

    def forward(self, *args, **kwargs):
        # First two args should be the input and hidden variable. Everything after that is passed to
        # the model
        input, hidden = args[0], args[1]
        args = args[2:]

        assert input.size(1) == 1, "Ponderer currently only works for unrolled RNN"
        assert not isinstance(input, torch.nn.utils.rnn.PackedSequence), "Ponderer currently does not work for PackedSequence"

        states = []
        cells = [] # For LSTMs
        outputs = []
        extra_return_values = [] # If the underlying model returns more than expected, we just add it to a list
                                 # We might want to deal with this in a different way. Maybe automatically take weighted average if it is a Tensor

        # For all elements in the batch, set one element to either 0 or 1
        input0 = input.clone()
        input0[:,0] = 0
        input1 = input.clone()
        input0[:,0] = 1

        for i in range(self.ponder_steps):
            # Indicate whether this is the first ponder step
            if i == 0:
                input = input1
            else:
                input = input0

            # Perform ponder step
            return_values = self.model(input, hidden, *args, **kwargs)

            # Extract return values
            model_output, hidden = return_values[0], return_values[1]
            if len(return_values) > 2:
                extra_return_values.append(return_values[2:])

            # Store return values
            if isinstance(hidden, tuple):
                states.append(hidden[0])
                cells.append(hidden[1])
            else:
                states.append(hidden)
            outputs.append(model_output)

        # With static pondering, we don't take a weighted average, but just the last output for now
        model_output = outputs[-1]
        if cells:
            hidden = (states[-1], cells[-1])
        else:
            hidden = states[-1]

        return model_output, hidden, extra_return_values