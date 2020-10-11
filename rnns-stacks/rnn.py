""" RNN class wrapper, taking symbols, producing vector representations of prefixes

Sort of a vestigial part of more complex older code, but here in case we'd like
to hand-write RNNs again.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import math

import utils

class PytorchRecurrentModel(nn.Module):
  """
  Class for mapping sequences of symbols to sequences
  of vectors representing prefixes, using PyTorch
  RNN classes.
  """

  def __init__(self, args, input_size, hidden_size, num_layers):
    super(PytorchRecurrentModel, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    if args['lm']['lm_type'] == 'RNN':
      self.recurrent_model = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
    elif args['lm']['lm_type'] == 'GRU':
      self.recurrent_model = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
    elif args['lm']['lm_type'] == 'LSTM':
      self.recurrent_model = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
    self.recurrent_model.to(args['device'])
    tqdm.write('Constructing a {} pytorch model w hidden size {}, layers {}, dropout {}'.format(args['lm']['lm_type'], hidden_size, num_layers, 0.0))

  def forward(self, batch):
    """ Computes the forward pass to construct prefix representations.
    Arguments:
      batch: (batch_len, seq_len) vectors representing
             contexts
    Returns:
      hiddens: (batch_len, seq_len, hidden_size)
               recurrent state vectors for each token in input.
    """
    return self.recurrent_model(batch)
