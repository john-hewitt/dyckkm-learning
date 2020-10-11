""" Trains/runs a language model on data available tokenized sentence-per-line format.

The main interface to running experiments with this codebase.

Usage:
      python rnns_stacks/run_lm.py <config.yaml>
"""

import torch
import yaml
import os
from tqdm import tqdm
from argparse import ArgumentParser
from dataset import Dataset
import lm
from training_regimen import train
import utils
import rnn
import reporter

if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('config')
  args = argp.parse_args()
  args = yaml.load(open(args.config))

  # Determine whether CUDA is available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  args['device'] = device

  # Construct the language model and dataset objects
  dataset = Dataset(args)
  input_size = args['lm']['embedding_dim']
  hidden_size = args['lm']['hidden_dim']
  recurrent_model = rnn.PytorchRecurrentModel(args, input_size,
      hidden_size, args['lm']['num_layers'])
  lm_model = lm.TraditionalLanguageModel(args, recurrent_model)

  # Prepare to write results 
  output_dir =  utils.get_results_dir_of_args(args)
  tqdm.write('Writing results to {}'.format(output_dir))
  os.makedirs(utils.get_results_dir_of_args(args),exist_ok=True)

  # Train and load most recent parameters
  train(args, lm_model, dataset.get_train_dataloader(), dataset.get_dev_dataloader())
  lm_model.load_state_dict(torch.load(utils.get_lm_path_of_args(args)))

  # Evaluate language model
  reporter.run_evals(args, lm_model, dataset, 'dev')
  reporter.run_evals(args, lm_model, dataset, 'test')
