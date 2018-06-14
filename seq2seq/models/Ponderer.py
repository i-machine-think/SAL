"""
- Forward pass expects input and hidden as first arguments. There may be more model-specific arguments
- Forward pass of underlying model should return model_output, hidden (and optionally extra return values)
- 'model_output' refers not necessarily to the output of the RNN, but the final output.
- Currently does not work for packed_sequence or for rolled-up RNN.
  Should be possible to implement for rolled RNN as well (and maybe also packed sequence), but we must be really careful with parameters and return values
  For example, we now pass args and kwargs directly to the underlying model, but maybe they need to be sliced when we would unroll the underlying model.
  Might be best to just only allow unrolled behaviour
- Assumes batch_first=True for now

Returns model_output, hidden, ponder_penalty, and a list of all extra return values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Ponderer(nn.Module):
    def __init__(self, model, hidden_size, max_ponder_steps, eps):
        super(Ponderer, self).__init__()

        self.model = model
        self.hidden_size = hidden_size
        self.max_ponder_steps = max_ponder_steps
        self.eps = eps

        self.halt_layer = nn.Linear(hidden_size, 1)
        # TODO: Prevent this from being reinitialized in train_model.py
        self.halt_layer.bias = nn.Parameter(torch.Tensor([1.])) # Avoid too long pondering at start of training

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

        assert input.size(0) == 1, "Ponderer currently does not work with batches. Should definitely be fixed."
        assert input.size(1) == 1, "Ponderer currently only works for unrolled RNN"
        assert not isinstance(input, torch.nn.utils.rnn.PackedSequence), "Ponderer currently does not work for PackedSequence"

        halting_probabilities = []
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

        halt_sum = 0
        for ponder_step in range(0, self.max_ponder_steps):
            # Indicate whether this is the first ponder step
            if ponder_step == 0:
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

            # Compute and store halting probability
            # TODO: Does view-1 work for batches?
            halting_probabilities.append(F.sigmoid(self.halt_layer(states[-1])).view(-1))
            halt_sum += halting_probabilities[-1].item()

            if halt_sum >= 1 - self.eps:
                break
        ponder_steps = ponder_step + 1

        # Residual is 1 - sum of all halting probabilities before the final step
        residual = torch.Tensor([1.])
        if len(halting_probabilities) > 1:
            residual = residual - torch.sum(torch.cat(halting_probabilities[:-1]))

        # Last steps halt probability is residual to make the probabilitiy distribution sum up to 1.
        halting_probabilities[-1] = residual

        # Convert lists to tensors
        outputs = torch.stack(outputs, dim=1)
        states = torch.stack(states, dim=1)
        halting_probabilities = torch.cat(halting_probabilities)
        if cells:
            cells = torch.states(cells, dim=1)

        # Get weighted averages
        model_output = torch.mv(outputs, halting_probabilities)
        if cells:
            hidden = (
                torch.mv(states, halting_probabilities),
                torch.mv(cells, halting_probabilities)
                )
        else:
            hidden = torch.mv(states, halting_probabilities)

        ponder_penalty = ponder_steps + residual
        
        return model_output, hidden, ponder_penalty, extra_return_values
