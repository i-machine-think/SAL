import torch
from torch.autograd import Variable

class AttentionGenerator(object):
    """
    Base class for encapsulation of functions generating attentive guidance.

    This class defines interfaces that should be implemented to interact with
    the AttentionTrainer class.

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        name (str): name of the attention generation mechanism
        key (str): key under which targets are stored in target dict
        pad_value (int): target token to use for padding

    Attributes:
        name (str): name of the attention generation mechanism
        key (str): key under which targets are stored in target dict
        pad_value (int): target token to use for padding
    """

    def __init__(self, name, key, pad_value=-1):
        self.name = name
        self.key = key
        self.pad_value = pad_value

    def add_attention_targets(self, input_variables, input_lengths, target_variables):
        """ Generate attention targets
        
        This  method defines how to compute the attention targets given the input
        variables, and adds it to the inputted dictionary containing previously
        defined target variables.

        Args:
            input_variables (torch.Tensor): inputs to a batch
            input_lengths (torch.Tensor): input lengths
            target_variables (dict): mapping keys to torch.Tensors representing target variables

        Returns:
            target_variables (dict): dictionary with attention targets steps added

        """
        raise NotImplementedError("Implement in subclass")


class LookupTableAttention(AttentionGenerator):
    """ Attention for lookup tables postfix annotation
    The class implements attention that slides over
    the input.

    Args:
        pad_value (int): target token to use for padding

    """
    _NAME = "lookup_table"
    _KEY = "attention_target"

    def __init__(self, pad_value):
        super(LookupTableAttention, self).__init__(name=self._NAME, key=self._KEY, pad_value=pad_value)

    def add_attention_targets(self, input_variables, input_lengths, target_variables):
        max_val = max(input_lengths) + 1
        batch_size = input_lengths.size(0)

        # get target attentions
        target_attentions = Variable(torch.cat(tuple([torch.cat((torch.ones(1), torch.arange(l), self.pad_value*torch.ones(max_val-l)), 0) for l in input_lengths]), 0).view(batch_size, max_val+1).long())

        if torch.cuda.is_available():
            target_attentions = target_attentions.cuda()

        target_variables['attention_target'] = target_attentions
        return target_variables


class PonderGenerator(object):
    """
    Base class for encapsulation of functions generating a pondering regime.

    This class defines interfaces that should be implemented to interact with
    the SupervisedTrainer class, to allow the trainer to mask out 'silent' steps
    for the computation of the loss.

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        name (str): name of the attention generation mechanism
        key (str): key under which targets are stored in target dict
        pad_value (int): target token to use for padding

    Attributes:
        name (str): name of the attention generation mechanism
        key (str): key under which targets are stored in target dict
        pad_token (int): target token to be ignored

    """

    def __init__(self, name, key, pad_token=-1):
        self.name = name
        self.key = key
        self.pad_token = pad_token

    def mask_silent_steps(self, input_variable, input_lengths, decoder_outputs):
        """ Generate non silent steps and remove from decoder_outputs
        
        This  method defines how to compute the attention targets given the input
        variables, and adds it to the inputted dictionary containing previously
        defined target variables.

        Args:
            input_variables (torch.Tensor): inputs to a batch
            input_lengths (torch.Tensor): input lengths
            target_variables (dict): mapping keys to torch.Tensors representing target variables

        Returns:
            target_variables (dict): dictionary with non-silent steps added

        """
        raise NotImplementedError("Implement in subclass")


class LookupTablePonderer(PonderGenerator):
    """ Attention for lookup tables postfix annotation

    """
    _NAME = "lookup_table"
    _KEY = "attention_target"

    def __init__(self):
        super(LookupTablePonderer, self).__init__(name=self._NAME, key=self._KEY, pad_token=-1)

    def mask_silent_outputs(self, input_variable, input_lengths, decoder_outputs):
        """ Find the last steps for every output sequence sequence.

        Args:
            input_variables (torch.Tensor): inputs to a batch
            decoder_targets (list): list of step outputs for batch
            input_lengths (torch.Tensor): input lengths

        Returns:
            outputs (list): a list containng the first and last step for every input sequence
        """

        # evaluate first (copy) step
        first_step = decoder_outputs[0]

        # for seq i in the batch, target is decoder_outputs[input_lengths[i]-1][i]
        last_step = torch.zeros_like(decoder_outputs[0])
        eos_step = torch.zeros_like(decoder_outputs[0])
        for i, l in enumerate(input_lengths):
            last_step[i] = decoder_outputs[l-1][i,:]
            eos_step[i] = decoder_outputs[l][i, :]

        decoder_outputs_non_silent = [first_step, last_step, eos_step]

        return decoder_outputs_non_silent

    def mask_silent_targets(self, input_variable, input_lengths, decoder_targets):
        """ Find the last steps for every target sequence.

        Args:
            input_variables (torch.Tensor): inputs to a batch
            decoder_targets (torch.Tensor): targets of a batch # batch x maxlen
            input_lengths (torch.Tensor): input lengths

        Returns:
            outputs (torch.Tensor): a tensor containing the last step, the final target for every input sequence
        """
        # take first and second decoder output
        targets_non_silent = decoder_targets[:,0:4].clone()

        # for seq i in the batch, final target is decoder_target[i][input_lengths[i]]
        for i, l in enumerate(input_lengths):
            targets_non_silent[i,2] = decoder_targets[i][l]
            targets_non_silent[i, 3] = decoder_targets[i][l+1]

        return targets_non_silent
