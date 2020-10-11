"""
Contains classes for reporting language modeling experiments
on the m-bounded Dyck-k languages.

Tests whether a language model has learned the constraints
of a dyck-k language
"""

import os
import yaml
import json
import sys
import torch
from collections import Counter
from tqdm import tqdm

import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from generate_mbounded_dyck import DyckPDFA
from dataset import Dataset
from lm import LanguageModel
import utils

P_THRESHOLD = .8

def eval_closing_bracket_constraint(prob_dist, gold_token, state, vocab, index_vec, current_index):
  """
  Determines whether, if there is an open bracket on the stack, the following holds:

    probability_mass_on_correct_closing_bracket / probability_mass_on_any_closing_bracket > 0.8
  
  and records the result of whether that property holds. (Does nothing if the stack is empty.)
  """
  result_dict = []
  if len(state) > 0:
    permitted_closing_bracket = state[-1] + ')'
    permitted_symbol_index = vocab[permitted_closing_bracket]
    permitted_close_bracket_mass = prob_dist[permitted_symbol_index]

    close_bracket_mass = 0
    for symbol in vocab:
      if ')' in symbol:
        close_bracket_mass += prob_dist[vocab[symbol]]

    # Record for the overall record
    result_dict.append(int(permitted_close_bracket_mass/close_bracket_mass > P_THRESHOLD))
    # Record for the number of tokens between the opening and closing brackets.
    result_dict.append('diff' + str(current_index - index_vec[-1])
        + '-' + str(int(permitted_close_bracket_mass/close_bracket_mass > P_THRESHOLD)))

  return result_dict

def eval_can_close_bracket_constraint(prob_dist, gold_token, state, vocab):
  """
  Determines whether, if the stack is empty, the sum of the probability mass assigned
  to all the closing brackets is less than 0.2

  and records the result of whether that property holds.
  (Does nothing if the stack is not empty.)
  """
  if len(state) == 0:
    close_bracket_mass = 0
    for symbol in vocab:
      if ')' in symbol:
        close_bracket_mass += prob_dist[vocab[symbol]]
    return [int(close_bracket_mass < 1 - P_THRESHOLD)]
  return []

def eval_can_open_bracket_constraint(prob_dist, gold_token, state, vocab, max_stack_depth):
  """
  Determines whether, if the stack is full, the sum of probability mass assigned
  to all the opening brackets is less than 0.2

  and records the result of whether that property holds.
  (Does nothing if the stack is not full.)
  """
  if len(state) == max_stack_depth:
    open_bracket_mass = 0
    for symbol in vocab:
      if '(' in symbol:
        open_bracket_mass += prob_dist[vocab[symbol]]
    return [int(open_bracket_mass < 1 - P_THRESHOLD)]
  return []

def eval_can_end_token_constraint(prob_dist, gold_token, state, vocab):
  """
  Determines whether, if the stack is not empty, the sum of probability mass assigned
  to the END token is less than 0.2

  and records the result of whether that property holds.
  (Does nothing if the stack is empty.)
  """
  if len(state) != 0:
    end_symbol_index = vocab['END']
    return [int(prob_dist[end_symbol_index] < 1 - P_THRESHOLD)]
  return []

def get_dyck_eval_dict(args, lm, dataset, dev_batches, split_name):
  """
  Collects statistics, over a set of examples from the m-bounded Dyck-k
  languages and the token-level probabilities assigned to them by a language model,
  of whether the the probabilities obey constraints specified by helper functions.
  """
  max_stack_depth = args['language']['{}_max_stack_depth'.format(split_name)]
  pdfa = DyckPDFA(args, max_stack_depth, args['language']['bracket_types'])

  results_dict = {
      'correct_closing_bracket_constraint': Counter(),
      'can_close_bracket_constraint': Counter(),
      'can_open_bracket_constraint': Counter(),
      'can_end_token_constraint': Counter(),
      }
  vocab = dataset.vocab
  inv_vocab = list(sorted(vocab, key = lambda x: vocab[x]))
  for observation_batch, label_batch, length_batch in tqdm(dev_batches, desc='[dev batch]'):
    logit_batch, _ = lm(observation_batch)
    batch_index = 0
    for batch_index, (observation, label, logit) in enumerate(zip(observation_batch, label_batch, logit_batch)):
      state_vec = []
      index_vec = []
      length = length_batch[batch_index]
      for symbol_index, (symbol, label, log_distrib, _) in enumerate(zip(observation, label, logit, range(length))):
        new_state_vec = pdfa.update_state(state_vec, inv_vocab[int(symbol)])
        if len(new_state_vec) > len(state_vec):
           index_vec.append(symbol_index)
        elif len(new_state_vec) < len(state_vec):
           index_vec.pop(-1)
        state_vec = new_state_vec
        results_dict['correct_closing_bracket_constraint'].update(eval_closing_bracket_constraint(
          torch.softmax(log_distrib, 0), symbol, state_vec, vocab, index_vec, symbol_index))
        results_dict['can_close_bracket_constraint'].update(eval_can_close_bracket_constraint(
          torch.softmax(log_distrib, 0), symbol, state_vec, vocab))
        results_dict['can_open_bracket_constraint'].update(eval_can_open_bracket_constraint(
          torch.softmax(log_distrib, 0), symbol, state_vec, vocab, max_stack_depth))
        results_dict['can_end_token_constraint'].update(eval_can_end_token_constraint(
          torch.softmax(log_distrib, 0), symbol, state_vec, vocab))
        batch_index += 1
  return results_dict

def report_results_dict(args, results, split_name):
  """ Aggregate statistics and write to disk.

  Arguments:
    results: string-key results dictionary from get_dyck_eval_dict
    split_name: string split name in train,dev,test.
  """
  # Report raw statistics
  output_dir =  utils.get_results_dir_of_args(args)
  output_path = os.path.join(output_dir, 'dyck-k-eval.json') 
  tqdm.write('Writing results to {}'.format(output_path))
  with open(output_path, 'w') as fout:
    json.dump(results, fout)

  # Report summary
  result_column = []
  indices = []
  for i in range(10000):
    key_correct = 'diff{}-1'.format(i)
    key_incorrect = 'diff{}-0'.format(i)
    correct_count = (results['correct_closing_bracket_constraint'][key_correct] 
        if key_correct in results['correct_closing_bracket_constraint'] else 0)
    incorrect_count = (results['correct_closing_bracket_constraint'][key_incorrect]
        if key_incorrect in results['correct_closing_bracket_constraint'] else 0)
    if correct_count + incorrect_count >= 1:
      result_column.append(correct_count  / (correct_count + incorrect_count) )
      indices.append(i)
  output_dir =  utils.get_results_dir_of_args(args)
  output_path = os.path.join(output_dir, 'summary-{}.json'.format(split_name)) 
  tqdm.write('Writing results to {}'.format(output_path))
  with open(output_path, 'w') as fout:
    json.dump(list(zip(result_column, indices)), fout)

def report_dyck_k_constraint_eval(args, lm, dataset, split_name):
  """ Get dataloader for reporting and run reporting
  """
  if split_name == 'dev': 
    dataloader = dataset.get_dev_dataloader()
  if split_name == 'train': 
    dataloader = dataset.get_train_dataloader()
  if split_name == 'test': 
    dataloader = dataset.get_test_dataloader()
  results_dict = get_dyck_eval_dict(args, lm, dataset, dataloader, split_name)
  report_results_dict(args, results_dict, split_name)

def make_plot(words, probabilities, vocab, path, title):
  """ Make a plot of probabilities assigned to tokens in a sequence 
  """
  fig = plt.gcf()
  fig.set_size_inches(22,16)
  plt.title(title)
  plt.ylabel("Vocabulary")
  plt.xlabel("Sequence")
  norm = mpl.colors.Normalize(vmin=0.,vmax=1.)
  im = plt.imshow(probabilities, norm=norm)
  cbar = plt.colorbar(im,fraction=0.010, pad=0.04, ticks=[0,0.25, 0.5, 0.75, 1])
  cbar.ax.tick_params(labelsize=7)
  for index, word in enumerate(words):
    plt.axvline(x=index)
    plt.annotate('\u25a0', (index-0.25, vocab[word]+.1),color='red',fontsize=4)
  plt.xticks(range(len(words)), labels=[x +','+ str(i+1) for i, x in enumerate(words)],rotation=65,fontsize=6)
  plt.yticks(range(len(vocab.keys())), labels=list(sorted(vocab.keys(), key=lambda x: vocab[x])),fontsize=6)
  plt.savefig(path,dpi=300,bbox_inches='tight')
  plt.clf()

def report_image_examples(args, lm, dataset, split_name):
  if split_name == 'dev': 
    dataloader = dataset.get_dev_dataloader()
  if split_name == 'train': 
    dataloader = dataset.get_train_dataloader()
  if split_name == 'test': 
    dataloader = dataset.get_test_dataloader()

  dataloader = dataset.get_dev_dataloader()
  vocab = dataset.vocab

  output_path = utils.get_results_dir_of_args(args)
  tqdm.write('Writing {} images to disk at {}'.format(split_name, output_path))

  index = 0
  inv_vocab = {v: k for k,v in vocab.items()}
  for batch, label_batch, length_batch in dataloader:
    logits, _ = lm(batch)
    for offset_index in range(logits.size()[0]):
      prob_data = logits[offset_index, 0:length_batch[offset_index],:]
      prob_data = prob_data.view(-1,*prob_data.size()).detach().cpu().numpy()
      _, seq_len, features = prob_data.shape

      make_plot([inv_vocab[int(x)] for x in 
        batch[offset_index][1:length_batch[offset_index]]] + ['END'] ,
        torch.softmax(torch.tensor(prob_data[0].T),0), vocab,
        output_path + '/example-'
        + str(index) +'.png', 'Likelihoods, {} Sentence {}'.format(split_name, index))
      if index == 20:
        return
      index += 1

def run_evals(args, lm, dataset, split_name):
  if 'constraints' in args['reporting']['reporting_methods']:
    report_dyck_k_constraint_eval(args, lm, dataset, split_name)
  if 'image_examples' in args['reporting']['reporting_methods']:
    report_image_examples(args, lm, dataset, split_name)

