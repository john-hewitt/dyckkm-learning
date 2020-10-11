"""
Utilities for determining paths to corpora, results, models
given config dictionaries describing an experiment, as well
as determining canonical vocabulary ordering
"""

import os
import string
import re
import copy

def get_identifier_iterator():
  """ Returns an iterator to provide unique ids to bracket types.
  """
  ids = iter(list(string.ascii_lowercase))
  k = 1
  while True:
    try:
      str_id = next(ids)
    except StopIteration:
      ids = iter(list(string.ascii_lowercase))
      k += 1
      str_id = next(ids)
    yield str_id*k

def get_vocab_of_bracket_types(bracket_types):
  """ Returns the vocabulary corresponding to the number of brackets.

  There are bracket_types open brackets, bracket_types close brackets,
  START, and END.
  Arguments:
    bracket_types: int (k in Dyck-(k,m))
  Returns:
    Dictionary mapping symbol string  s to int ids.
  """
  id_iterator = get_identifier_iterator()
  ids = [next(id_iterator) for x in range(bracket_types)]
  vocab = {x: c for c, x in enumerate(['(' + id_str for id_str in ids] + [id_str + ')' for id_str in ids] + ['START', 'END'])}
  return vocab, ids

def get_results_dir_of_args(args):
  """
  Takes a (likely yaml-defined) argument dictionary
  and returns the directory to which results of the
  experiment defined by the arguments will be saved
  """
  return args['reporting']['reporting_loc']

def get_corpus_paths_of_args(args):
  return {'train': args['corpus']['train_corpus_loc'],
          'dev': args['corpus']['dev_corpus_loc'],
          'test': args['corpus']['test_corpus_loc']}

def get_lm_path_of_args(args):
  results_dir = get_results_dir_of_args(args)
  return os.path.join(results_dir, args['lm']['save_path'])

def deprecated_get_results_dir_of_args(args):
  """ (Deprecated)
  Takes a (likely yaml-defined) argument dictionary
  and returns the directory to which results of the
  experiment defined by the arguments will be saved
  """
  if 'vocab_size' in args['language']:
    del args['language']['vocab_size']
  if (args['corpus']['train_override_path'] or 
      args['corpus']['dev_override_path'] or 
      args['corpus']['test_override_path']):
    language_specification_path = re.sub('train', '', args['corpus']['train_override_path'])
  else:
    language_specification_path = '_'.join([key+str(value) for key, value in args['language'].items()])

  path = os.path.join(
   args['reporting']['root'],
   '-'.join([
    '_'.join([key+str(value) for key, value in args['lm'].items()]),
    '_'.join([key+str(value) for key, value in args['training'].items()]),
    language_specification_path,
    ]))
  path = re.sub('max_stack_depth', 'msd', path)
  path = re.sub('learning_rate', 'lr', path)
  path = re.sub('sample_count', 'sc', path)
  path = re.sub('max_length', 'ml', path)
  path = re.sub('train', 'tr', path)
  path = re.sub('test', 'te', path)
  path = re.sub('num_layers', 'nl', path)
  path = re.sub('hidden_dim', 'hd', path)
  path = re.sub('embedding_dim', 'ed', path)
  path = re.sub('analytic_model', 'am', path)
  path = re.sub('max_epochs', 'me', path)
  path = re.sub('batch_size', 'bs', path)
  path = re.sub('min_length', 'ml', path)
  path = re.sub('min_state_filter_percent', 'fp', path)
  return path

def deprecated_get_corpus_paths_of_args(args):
  """ (Deprecated)
  Takes a (likely yaml-defined) argument dictionary
  and returns the paths of the train/dev/test
  corpora files.
  """
  args = copy.deepcopy(args)
  if 'vocab_size' in args['language']:
    del args['language']['vocab_size']
  if (args['corpus']['train_override_path'] or 
      args['corpus']['dev_override_path'] or 
      args['corpus']['test_override_path']):
    train_path = args['corpus']['train_override_path']
    dev_path = args['corpus']['dev_override_path']
    test_path = args['corpus']['test_override_path']
  else:
    path = os.path.join(
     args['corpus']['root'],
    '-'.join([
      '_'.join([key+str(value) for key, value in args['language'].items()])]))
    path = re.sub('max_stack_depth', 'msd', path)
    path = re.sub('learning_rate', 'lr', path)
    path = re.sub('sample_count', 'sc', path)
    path = re.sub('max_length', 'ml', path)
    path = re.sub('train', 'tr', path)
    path = re.sub('test', 'te', path)
    path = re.sub('num_layers', 'nl', path)
    path = re.sub('hidden_dim', 'hd', path)
    path = re.sub('embedding_dim', 'ed', path)
    path = re.sub('analytic_model', 'am', path)
    path = re.sub('max_epochs', 'me', path)
    path = re.sub('batch_size', 'bs', path)
    path = re.sub('min_length', 'ml', path)
    path = re.sub('min_state_filter_percent', 'fp', path)
    train_path = os.path.join(path, 'train.formal.txt')
    dev_path = os.path.join(path, 'dev.formal.txt')
    test_path = os.path.join(path, 'test.formal.txt')
      #path = re.sub('', 'te', path)
  return {'train':train_path, 'dev':dev_path, 'test':test_path}

