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

Inspired by https://github.com/zphang/adaptive-computation-time-pytorch/blob/master/act/models.py
and https://github.com/imatge-upc/danifojo-2018-repeatrnn/blob/PyTorch/src/act_cell.py
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Ponderer(nn.Module):
    def __init__(self, model, hidden_size, output_size, max_ponder_steps, eps, optimize=False):
        super(Ponderer, self).__init__()

        self.model = model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_ponder_steps = max_ponder_steps
        self.eps = eps
        self.optimize = optimize

        self.halt_layer = nn.Linear(hidden_size, 1)
        # Avoid too long pondering at start of training. Do not change name, as otherwise it would be reinitialized in train_model.py
        self.halt_layer.bias = nn.Parameter(torch.Tensor([1.], device=device))

    # Automatic delegation of undefined attributes.
    # First we check whether the parent class nn.Module has implemented this function
    # If not, we delegate the call to the model
    def __getattr__(self, attr):
        try:
            return super(Ponderer, self).__getattr__(attr)
        except AttributeError:
            pass
        try:
            return self.model.__getattribute__(attr)
        except AttributeError:
            return self.model.__getattr__(attr)

    def forward(self, *args, **kwargs):
        # First two args should be the input and hidden variable. Everything after that is passed to the model
        input, hidden = args[0], args[1]
        args = args[2:]

        is_lstm = isinstance(hidden, tuple)

        batch_size, decoder_length, input_size = input.size()
        n_layers = hidden[0].size(0) if is_lstm else hidden.size(0)

        assert decoder_length == 1, "Ponderer currently only works for unrolled RNN"
        assert not isinstance(input, torch.nn.utils.rnn.PackedSequence), "Ponderer currently does not work for PackedSequence"

        # Initialize tensors in which we will store the halting probabilities and weighted averages
        accumulative_halting_probabilities = torch.zeros([batch_size], device=device)
        accumulative_states = torch.zeros([n_layers, batch_size, self.hidden_size], device=device)
        if is_lstm:
            accumulative_cells = torch.zeros([n_layers, batch_size, self.hidden_size], device=device)
        accumulative_outputs = torch.zeros([batch_size, self.output_size], device=device)

        ponder_steps = torch.zeros([batch_size], device=device)
        ponder_penalty = torch.zeros([batch_size], device=device, requires_grad=True).clone()

        extra_return_values = []  # If the underlying model returns more than expected, we just add it to a list
        # We might want to deal with this in a different way. Maybe automatically take weighted average if it is a Tensor?

        # For all elements in the batch, set one element to either 0 or 1 to indicate whether it is the first ponder step
        input0 = input.clone()
        input1 = input.clone()
        input0[:, :, 0] = 0
        input1[:, :, 0] = 1

        # ByteTensor containing 1's for all elements that have not halted yet
        batch_element_selector = (accumulative_halting_probabilities < 1 - self.eps)

        for ponder_step in range(self.max_ponder_steps):
            # Select the elements in the batch that still need processing. Convert ByteTensor to indices
            batch_element_idx = batch_element_selector.nonzero().squeeze(1)

            # Increment the nuber of executed ponder steps
            ponder_steps[batch_element_idx] = ponder_steps[batch_element_idx] + 1

            # Indicate whether this is the first ponder step
            if ponder_step == 0:
                input = input1
            else:
                input = input0

            # In, for example, the encoder, we can optimize by not executing the recurrent model for batch elements that have halted.
            # For the decoder, for example, this is not possible since it uses an attention mechanism that would have no way of knowing
            # which decoder batch element to align with which encoder batch element.
            if self.optimize:
                # Only select the necessary batch elements (which is the second dimension for the hidden states. n_layers is first dimension)
                selected_input = input[batch_element_idx]
                if is_lstm:
                    selected_hidden = hidden[0][:, batch_element_idx, :], hidden[1][:, batch_element_idx, :]
                else:
                    selected_hidden = hidden[:, batch_element_idx, :]
            else:
                # We can't slice with batch_element_idx to reduce the number of computations as the attention mechanism
                # would have no way of knowing which decoder batch element to align with which encoder batch element
                selected_input = input
                selected_hidden = hidden

            # Perform single ponder step
            return_values = self.model(selected_input, selected_hidden, *args, **kwargs)

            # Extract return values
            model_output, state = return_values[0], return_values[1]
            model_output = model_output.squeeze(1)  # We used unrolled RNN and can remove the time step dimension

            # Unpack LSTM, update hidden state for next step, and select only relevant states/outputs
            # In the case of not optimizing the recurrent model, we need to select only the updates outputs/states that
            # are relevant / have not halted.
            if not self.optimize:
                if is_lstm:
                    hidden = state  # Update hidden state
                    state, cell = state  # Unpack
                    state = state[:, batch_element_idx, :]  # Select relevant non-halting
                    cell = cell[:, batch_element_idx, :]
                else:
                    hidden = state  # Update hidden state

                model_output = model_output[batch_element_idx]  # Select relevant
            else:
                if is_lstm:
                    state, cell = state  # Unpack
                    hidden[0][:, batch_element_idx, :] = state  # Update hidden state
                    hidden[1][:, batch_element_idx, :] = cell
                else:
                    hidden[:, batch_element_idx, :] = state

            if len(return_values) > 2:
                extra_return_values.append(return_values[2:])

            # Compute and store halting probability. The input will be only the last layer of the hidden state
            halt_layer_input = state[-1].squeeze(0)
            halting_probabilities = F.sigmoid(self.halt_layer(halt_layer_input))
            # For batches it would add a second dimension, but not for single examples. Remove this extra dim if present
            if halting_probabilities.dim() > 1:
                halting_probabilities = halting_probabilities.squeeze(1)

            # Within the currently already selected elements, check which ones would halt after this ponder step.
            # If we reached the maximum number of steps, we will replace the probability with the remainder for ALL elements
            if ponder_step == self.max_ponder_steps - 1:
                last_ponder_step_selector = torch.ones_like(batch_element_idx)
            # Else, we will select the elements (within already selected) that would halt
            else:
                next_accumulative_halt_probs = accumulative_halting_probabilities[batch_element_idx] + halting_probabilities
                last_ponder_step_selector = (next_accumulative_halt_probs >= 1 - self.eps)

            # Convert ByteTensor selector to indices
            last_ponder_step_idx = last_ponder_step_selector.nonzero()
            # Only has the extra dimension if there actually are halting elements
            if last_ponder_step_idx.dim() > 1:
                last_ponder_step_idx = last_ponder_step_idx.squeeze(1)

            # Compute remainder
            if ponder_step == 0:
                remainder = torch.ones(last_ponder_step_idx.size(), device=device)
            else:
                # R(t) = 1 - Sum_{n-1} (halt_prob_i)
                remainder = 1 - (accumulative_halting_probabilities[batch_element_idx[last_ponder_step_idx]])

            # For all halting elements, replace probability with remainder
            halting_probabilities = halting_probabilities.clone()
            halting_probabilities[last_ponder_step_idx] = remainder

            # In the original paper the authors minimize the remainder, which comes down to maximizing the halting probabilities
            # of all time steps [1,N-1]. Instead we could also minimize the halting probability of time step N.
            # Sometimes, I found this to be beneficial for convergence. To do this, simply move this line to before the halting probabilities are replaced
            # By the remainders
            ponder_penalty[batch_element_idx[last_ponder_step_idx]] = halting_probabilities[last_ponder_step_idx]

            # Add halting probability (or remainder) to the sum
            accumulative_halting_probabilities[batch_element_idx] = accumulative_halting_probabilities[batch_element_idx] + halting_probabilities

            # For all non-halting ponder steps, add the state/cell/output, weighted by halting probability
            # For all terminating ponder steps, add the state/cell/output, weighted by remainder
            accumulative_states[:, batch_element_idx] = halting_probabilities.view(1, -1, 1).expand_as(state) * state
            if is_lstm:
                accumulative_cells[:, batch_element_idx] = halting_probabilities.view(1, -1, 1).expand_as(cell) * cell
            accumulative_outputs[batch_element_idx] = halting_probabilities.view(-1, 1).expand_as(model_output) * model_output

            # Stop if all batch elements have halted
            batch_element_selector = (accumulative_halting_probabilities < 1 - self.eps)
            if not batch_element_selector.any():
                break

        # We add the decoder_length dimension again
        model_output = accumulative_outputs.unsqueeze(1)

        if is_lstm:
            hidden = accumulative_states, accumulative_cells
        else:
            hidden = accumulative_states

        # rho(t) = N(t) + R(t)
        ponder_penalty = ponder_steps + ponder_penalty

        return model_output, hidden, extra_return_values, ponder_penalty
