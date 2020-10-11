""" Loading data from disk and providing DataLoaders for PyTorch.
"""
import os
import sys
from tqdm import tqdm
from collections import Counter, defaultdict, namedtuple
import json

import torch
from torch.utils.data import DataLoader
import utils
import torch.nn as nn
from generate_mbounded_dyck import DyckPDFA

class Dataset:
  """ Loading data from disk and providing DataLoaders for PyTorch.

  Note: adds START token to the beginning of each sequence. 
  """
  def __init__(self, args):
    self.args = args
    self.observation_class = namedtuple('observation', ['sentence'])
    self.vocab, _ = utils.get_vocab_of_bracket_types(args['language']['bracket_types'])
    args['language']['vocab_size'] = len(self.vocab)
    self.batch_size = args['training']['batch_size']

    train_dataset_path = utils.get_corpus_paths_of_args(args)['train']
    dev_dataset_path = utils.get_corpus_paths_of_args(args)['dev']
    test_dataset_path = utils.get_corpus_paths_of_args(args)['test']

    self.train_dataset = ObservationIterator(self.load_tokenized_dataset(train_dataset_path, 'train'))
    self.dev_dataset = ObservationIterator(self.load_tokenized_dataset(dev_dataset_path, 'dev'))
    self.test_dataset = ObservationIterator(self.load_tokenized_dataset(test_dataset_path, 'test'))

  def load_tokenized_dataset(self, filepath, split_name):
    """Reads in a conllx file; generates Observation objects
    
    For each sentence in a conllx file, generates a single Observation
    object.

    Args:
      filepath: the filesystem path to the conll dataset
  
    Returns:
      A list of Observations 
    """
    tqdm.write('Getting dataset from {}'.format(filepath))
    observations = []
    lines = (x for x in open(filepath))
    for line in lines:
      tokens = [x.strip() for x in line.strip().split()]
      tokens = ['START'] + tokens
      if self.vocab: 
        tokens = [self.vocab[x] for x in tokens]
      observation = self.observation_class(tokens)
      observations.append(observation)
    return observations

  def custom_pad(self, batch_observations):
    seqs = [torch.tensor(x[0].sentence[:-1], device=self.args['device']) for x in batch_observations] # Cut out the last token
    lengths = [len(x) for x in seqs]
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    labels = [torch.tensor(x[1], device=self.args['device']) for x in batch_observations]
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return seqs, labels, lengths

  def get_train_dataloader(self, shuffle=True):
    """Returns a PyTorch dataloader over the training dataset.

    Args:
      shuffle: shuffle the order of the dataset.
      use_embeddings: ignored

    Returns:
      torch.DataLoader generating the training dataset (possibly shuffled)
    """
    return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=shuffle)

  def get_dev_dataloader(self):
    """Returns a PyTorch dataloader over the development dataset.

    Args:
      use_embeddings: ignored

    Returns:
      torch.DataLoader generating the development dataset
    """
    return DataLoader(self.dev_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

  def get_test_dataloader(self):
    """Returns a PyTorch dataloader over the test dataset.

    Args:
      use_embeddings: ignored

    Returns:
      torch.DataLoader generating the test dataset
    """
    return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

class ObservationIterator:
  """ List Container for lists of Observations and labels for them.

  Used as the iterator for a PyTorch dataloader.
  """

  def __init__(self, observations, train=False):
    self.observations = observations
    self.set_labels(observations)

  def set_labels(self, observations):
    """ Constructs aand stores label for each observation.

    Args:
      observations: A list of observations describing a dataset
    """
    self.labels = []
    for observation in tqdm(observations, desc='[computing labels]'):
      labels = observation.sentence[1:] # LM must predict EOS
      self.labels.append(labels)

  def __len__(self):
    return len(self.observations)

  def __getitem__(self, idx):
    return self.observations[idx], self.labels[idx]
