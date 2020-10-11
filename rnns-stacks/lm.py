""" Language model class agnostic to how context representations are built.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import math

import utils

class LanguageModel(nn.Module):
  """ Language model class, for assigning distributions over a vocab.

  """

  def forward(self, batch):
    """
    Takes in a batch of integer ids and emits a sequence
    of _unnormalized_ probability logits for softmax

    Args:
      batch: (batch_size, seq_len)
    Returns:
      A tuple of (logits, hiddens) where
        logits: (batch_size, seq_len, vocab_len)
        hiddens: (batch_size, seq_len, hidden_size)
    """
    raise NotImplementedError

class TraditionalLanguageModel(LanguageModel):
  """
  Class for converting vector representations of prefixes
  to probability distributions over the next token,
  using the traditional language modeling parameterization:
    
        P_t = softmax(Wh_{t-1}+b)
  """
  def __init__(self, args, recurrent_model):
    super(LanguageModel, self).__init__()
    print("Making traditional language model heads")
    self.args = args
    self.vocab_size = args['language']['vocab_size']
    hidden_size = args['lm']['hidden_dim']
    input_size = args['lm']['embedding_dim']
    self.recurrent_model = recurrent_model
    self.dropout = args['training']['dropout'] if 'dropout' in args['training'] else 0.0

    # Specify embedding and softmax matrices
    # Does not tie embedding matrices; could add option to tie
    self.embeddings = nn.Embedding(self.vocab_size, input_size)
    self.embeddings.to(args['device'])
    self.readout_W = nn.Parameter(data=torch.zeros(self.vocab_size, hidden_size))
    upper_init = math.sqrt(hidden_size)
    nn.init.uniform_(self.readout_W, -upper_init, upper_init)
    self.readout_W.to(args['device'])
    
    self.readout_b = nn.Parameter(data=torch.zeros(self.vocab_size))
    self.readout_b.to(args['device'])

    self.to(args['device'])


  def forward(self, batch):
    """ Computes the forward pass to construct log-probabilities.
    Arguments:
      batch: (batch_len, seq_len, feature_count) vectors representing
             contexts
    Returns:
      logits: (batch_len, seq_len, vocab_size) log-probabilities over
              vocabulary predicting next token, for each token in input.
      hiddens: (batch_len, seq_len, feature_count)
               recurrent state vectors for each token in input.
    """
    vecs = self.embeddings(batch)
    hiddens, _ = self.recurrent_model(vecs)
    if self.dropout:
      hiddens = torch.nn.functional.dropout(hiddens, p=self.dropout)
    logits = torch.matmul(hiddens, self.readout_W.t()) + self.readout_b
    return logits, hiddens
