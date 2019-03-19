import torch
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

def gumbel_softmax(probs, tau, hard):
	""" Computes samplig from the Gumbel Softmax (GS) distribution

    Args:
        probs (torch.tensor): probabilities of shape [batch_size, n_classes] 
        tau (float): temperature parameter for the GS
        hard (bool): discretize if True
    """

	rohc = RelaxedOneHotCategorical(tau, probs)
	y = rohc.rsample()

	if hard:
		y_hard = torch.zeros_like(y)
		y_hard.scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
		y = (y_hard - y).detach() + y

	return y