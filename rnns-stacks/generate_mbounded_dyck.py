"""Implements Dyck-(k,m) languages sampling and writing to disk.

Used as a standalone script to generate datasets for training
LMs on the Dyck-(k,m) language.

Note: random seed is hardcoded to ensure reproducible data.

Usage:
    
    python rnns-stacks/generate_mbounded_dyck.py <config>
"""

from collections import OrderedDict, defaultdict, Counter
import torch
from tqdm import tqdm
from copy import copy
import yaml
import sys
import os
import numpy as np
from argparse import ArgumentParser
import random
import math

import utils


class DyckPDFA:
  """
  Implements a probabilistic finite automata (PFA) that
  generates the dyck language
  """

  def __init__(self, args, max_stack_depth, bracket_types):
    self.args = args
    self.max_stack_depth = max_stack_depth
    self.bracket_types = bracket_types
    self.vocab, self.ids = utils.get_vocab_of_bracket_types(bracket_types)
    self.vocab_list = list(sorted(self.vocab.keys(), key=lambda x: self.vocab[x]))
    self.distributions = {}

  def get_token_distribution_for_state(self, state_vec):
    """
    Given a stack state (list of ids, e.g., ['a', 'b']
    produces the probability distribution over next tokens
    """
    if state_vec in self.distributions:
      return self.distributions[state_vec]
    distrib_vec = torch.zeros(len(self.vocab))
    if len(state_vec) == 0:
      for id_str in self.ids:
        distrib_vec[self.vocab['(' + id_str]] = 1/len(self.ids)
      distrib_vec[self.vocab['END']] += 1
    elif len(state_vec) == self.max_stack_depth:
      distrib_vec[self.vocab[state_vec[-1] + ')']] = 1
    else:
      for id_str in self.ids:
        distrib_vec[self.vocab['(' + id_str]] = 1/len(self.ids)
      distrib_vec[self.vocab[state_vec[-1] + ')']] = 1
    self.distributions[tuple(state_vec)] = torch.distributions.Categorical(distrib_vec / torch.sum(distrib_vec))
    return self.distributions[state_vec]

  def update_state(self, state_vec, new_char_string):
    """
    Updates the DFA state based on the character new_char_string

    For a valid open/close bracket, pushes/pops as necessary.
    For an invalid open/close bracket, leaves state unchanged.
    """
    state_vec = list(state_vec)
    if ')' in new_char_string:
      bracket_type = new_char_string.strip(')')
      if len(state_vec) > 0 and state_vec[-1] == bracket_type:
        state_vec = state_vec[:-1]
    if '(' in new_char_string:
      bracket_type = new_char_string.strip('(')
      if len(state_vec) < self.max_stack_depth:
        state_vec.append(bracket_type)
    return state_vec

  def sample(self, length_min, length_max=-1):
    """
    Returns a sample from the Dyck language, as well
    as the maximum number of concurrently-open brackets,
    and the number of times traversed from empty-stack to
    full-stack and back.
    """
    state_vec = []
    string = []
    max_state_len = 0
    empty_full_empty_traversals = 0
    empty_flag = True
    full_flag = False
    while True:
      #probs = torch.distributions.Categorical(self.get_token_distribution_for_state(state_vec))
      probs = self.get_token_distribution_for_state(tuple(state_vec))
      new_char = probs.sample()
      new_char_string = self.vocab_list[int(new_char)]
      # Break from generation if END is permitted and sampled
      if new_char_string == 'END':
        if len(string) < length_min:
          continue
        else:
          string.append(new_char_string)
          break
      # Otherwise, update the state vector
      string.append(new_char_string)
      state_vec = self.update_state(state_vec, new_char_string)
      max_state_len = max(max_state_len, len(state_vec))
      if len(state_vec) == self.max_stack_depth and empty_flag:
        full_flag = True
        empty_flag = False
      if len(state_vec) == 0 and full_flag:
        full_flag = False
        empty_flag = True
        empty_full_empty_traversals += 1

    return string, max_state_len, empty_full_empty_traversals


def write_dataset(args, pfa, min_length, max_length, split_name):
  """
  Samples from the Dyck language and writes the given number
  of examples (with min and max lengths) to disk
  """
  dataset_path = utils.get_corpus_paths_of_args(args)[split_name]
  tqdm.write('Writing {} corpus to {}'.format(split_name, dataset_path))
  os.makedirs('/'.join(dataset_path.split('/')[:-1]), exist_ok=True)
  sample_count = args['language']['{}_sample_count'.format(split_name)]
  empty_full_empty_cumsum = 0
  empty_full_empty_count = 0
  total_tokens_written = 0
  with open(dataset_path, 'w') as fout:
    lengths = [] 
    #for i in tqdm(range(sample_count),smoothing=0.01):
    pbar = tqdm(total=sample_count)
    while total_tokens_written < sample_count:
      sample_met_constraints = False
      while not sample_met_constraints:
        string, max_state_len, empty_full_empty_traversals = pfa.sample(min_length, max_length)
        length = len(string)
        if length >= min_length and length <= max_length:
          sample_met_constraints = True
          fout.write(' '.join(string))
          empty_full_empty_cumsum += empty_full_empty_traversals
          empty_full_empty_count += 1
          lengths.append(length)
          pbar.update(length)
          total_tokens_written += length
      fout.write('\n')
    pbar.close()


def evaluate_datasets(args):
  """ Conmputes a few statistical properies of a generated datasets.

  Usage:
      
      python rnns_stacks/generate_mbounded_dyck.py --evaluate 1 <config>
  """
  dyck_pfsa = DyckPDFA(args, args['language']['train_max_stack_depth'], args['language']['bracket_types'])

  train_dataset_path = utils.get_corpus_paths_of_args(args)['train']
  dev_dataset_path = utils.get_corpus_paths_of_args(args)['dev']
  test_dataset_path = utils.get_corpus_paths_of_args(args)['test']

  max_possible_states = sum((args['language']['bracket_types']**i for i in range(args['language']['train_max_stack_depth']+1)))
  max_possible_transitions = sum((3*(args['language']['bracket_types']**i) for i in range(0, args['language']['train_max_stack_depth']))) + args['language']['bracket_types']**args['language']['train_max_stack_depth']

  dev_states = set([])
  dev_state_transition_pairs = set()
  dev_examples = set()
  dev_there_and_back_agains = []
  for line in tqdm(open(dev_dataset_path), smoothing=0):
    there_and_back_again = 0
    going_to_full = True
    state = []
    tokens = [x.strip() for x in line.strip().split()]
    dev_examples.add(tuple(tokens))
    for token in tokens:
      dev_state_transition_pairs.add((tuple(state), token))
      state = dyck_pfsa.update_state(state, token)
      dev_states.add(tuple(state))
      if len(state) == args['language']['train_max_stack_depth'] and going_to_full:
        going_to_full = False
      elif len(state) == 0 and (not going_to_full):
        going_to_full = True
        there_and_back_again += 1
    dev_there_and_back_agains.append(there_and_back_again)
  dev_there_and_back_again_median = np.median(dev_there_and_back_agains)
  dev_there_and_back_again_stddev = np.std(dev_there_and_back_agains)

  test_states = set([])
  test_state_transition_pairs = set()
  test_examples = set()
  test_there_and_back_agains = []
  test_length_statistics = defaultdict(int)
  for line in tqdm(open(test_dataset_path), smoothing=0):
    there_and_back_again = 0
    going_to_full = True
    state = []
    tokens = [x.strip() for x in line.strip().split()]
    test_length_statistics[len(tokens)] += 1
    test_examples.add(tuple(tokens))
    for token in tokens:
      test_state_transition_pairs.add((tuple(state), token))
      state = dyck_pfsa.update_state(state, token)
      test_states.add(tuple(state))
      if len(state) == args['language']['train_max_stack_depth'] and going_to_full:
        going_to_full = False
      elif len(state) == 0 and not going_to_full:
        going_to_full = True
        there_and_back_again += 1
    test_there_and_back_agains.append(there_and_back_again)
  test_there_and_back_again_median = np.median(test_there_and_back_agains)
  test_there_and_back_again_stddev = np.std(test_there_and_back_agains)

  training_states = set([])
  training_state_transition_pairs = set()
  training_examples = set()
  training_there_and_back_agains = []
  train_length_statistics = defaultdict(int)
  for index, line in tqdm(enumerate(open(train_dataset_path)), smoothing=0):
    there_and_back_again = 0
    going_to_full = True
    state = []
    tokens = [x.strip() for x in line.strip().split()]
    train_length_statistics[len(tokens)] += 1
    training_examples.add(tuple(tokens))
    for token in tokens:
      training_state_transition_pairs.add((tuple(state), token))
      state = dyck_pfsa.update_state(state, token)
      training_states.add(tuple(state))
      if len(state) == args['language']['train_max_stack_depth'] and going_to_full:
        going_to_full = False
      elif len(state) == 0 and not going_to_full:
        going_to_full = True
        there_and_back_again += 1
    training_there_and_back_agains.append(there_and_back_again)
  training_there_and_back_again_median = np.median(training_there_and_back_agains)
  training_there_and_back_again_stddev = np.std(training_there_and_back_agains)

  print('training length statistics', {x:c for x,c in sorted(train_length_statistics.items(), key=lambda x: x[0])})
  print('test length statistics', {x:c for x,c in sorted(test_length_statistics.items(), key=lambda x: x[0])})
  train_lengths = []
  for length in train_length_statistics:
    train_lengths = train_lengths + [length for _ in range(train_length_statistics[length])]
  print('Average train token count', np.mean(np.array(train_lengths)))

  print('training seen states: {} of max possible {}: {}'.format(len(training_states), max_possible_states, len(training_states)/max_possible_states))
  print('training seen state-transition tuples: {} of max possible {} : {}'.format(len(training_state_transition_pairs), max_possible_transitions, len(training_state_transition_pairs)/max_possible_transitions))
  print('training there_and_back_agains median: {} stddev: {}'.format(training_there_and_back_again_median, training_there_and_back_again_stddev))

  dev_states_in_training = dev_states.intersection(training_states)
  print('dev states seen in training: {} of total dev {} : {}'.format(len(dev_states_in_training), len(dev_states), len(dev_states_in_training)/len(dev_states)))

  dev_transitions_in_training = dev_state_transition_pairs.intersection(training_state_transition_pairs)
  print('dev transitions seen in training: {} of total dev {} : {}'.format(len(dev_transitions_in_training), len(dev_state_transition_pairs), len(dev_transitions_in_training)/len(dev_state_transition_pairs)))

  dev_samples_in_training = dev_examples.intersection(training_examples)
  print('dev examples seen in training: {} of max possible {} : {}'.format(len(dev_samples_in_training), len(dev_examples), len(dev_samples_in_training)/len(dev_examples)))
  print('dev there_and_back_agains median: {} stddev: {}'.format(dev_there_and_back_again_median, dev_there_and_back_again_stddev))

  test_states_in_training = test_states.intersection(training_states)
  print('test states seen in training: {} of total test {} : {}'.format(len(test_states_in_training), len(test_states), len(test_states_in_training)/len(test_states)))

  test_transitions_in_training = test_state_transition_pairs.intersection(training_state_transition_pairs)
  print('test transitions seen in training: {} of total test {} : {}'.format(len(test_transitions_in_training), len(test_state_transition_pairs), len(test_transitions_in_training)/len(test_state_transition_pairs)))

  test_samples_in_training = test_examples.intersection(training_examples)
  print('test examples seen in training: {} of max possible {} : {}'.format(len(test_samples_in_training), len(test_examples), len(test_samples_in_training)/len(test_examples)))

  print('test there_and_back_agains median: {} stddev: {}'.format(test_there_and_back_again_median, test_there_and_back_again_stddev))


if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('config')
  argp.add_argument('--evaluate', default=False)
  cli_args = argp.parse_args()
  args = yaml.load(open(cli_args.config))
  if cli_args.evaluate:
    evaluate_datasets(args)
    exit()
  dyck_pfsa = DyckPDFA(args, args['language']['train_max_stack_depth'], args['language']['bracket_types'])
  torch.manual_seed(1)
  write_dataset(args, dyck_pfsa,
      args['language']['train_min_length'],
      args['language']['train_max_length'],
      'train')
  torch.manual_seed(1000)
  dyck_pfsa = DyckPDFA(args, args['language']['dev_max_stack_depth'], args['language']['bracket_types'])
  write_dataset(args, dyck_pfsa,
      args['language']['dev_min_length'],
      args['language']['dev_max_length'],
      'dev')
  torch.manual_seed(2000)
  dyck_pfsa = DyckPDFA(args, args['language']['test_max_stack_depth'], args['language']['bracket_types'])
  write_dataset(args, dyck_pfsa,
      args['language']['test_min_length'],
      args['language']['test_max_length'],
      'test')
