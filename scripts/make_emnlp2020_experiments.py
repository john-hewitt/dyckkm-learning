"""
This script generates a .yaml configuration file for each experiment
in the EMNLP 2020 paper 

     RNNs can generate bounded hierarchical languages with optimal memory

Usage:

      python scripts/make_emnlp2020_experiments.py
"""

import os
import math
import re

## A .yaml file is generated to construct the datasets, as well as run each experiment 
language_generation_template = 'configs/emnlp2020/templates/language_template.yaml'
language_generation_root = 'configs/emnlp2020/languages'
experiment_generation_template = 'configs/emnlp2020/templates/experiment_template.yaml'
experiment_generation_root = 'configs/emnlp2020/experiments'

os.makedirs(language_generation_root, exist_ok=True)
os.makedirs(experiment_generation_root, exist_ok=True)

# The dev and test counts are held fixed
dev_count = 20000
test_count = 300000

## Start with the language variants
# m is the maximum stack depth
for m in [3, 5]:
  # k is the number of types of brackets
  for k in [2, 8, 32, 128]:
    # train_count is the number of training samples
    #for train_count in [1000, 10000, 100000, 1000000, 10000000]:
    for train_count in [2e3, 2e4, 2e5, 2e6, 2e7]:
      # calculate the training min and max sequence lengths
      train_count = int(train_count)
      train_min = 1 #6*m*(m-2)+20
      train_max = 8*m*(m-2)+60
      # calculate the testing min and max sequence lengths
      test_min = train_max+1
      test_max = int(train_max*2)
      # determine the corpus paths
      data_root = 'data'
      train_path = os.path.join(data_root, 'k{}_m{}_tr{}.train'.format(k,m,train_count))
      dev_path = os.path.join(data_root, 'k{}_m{}_tr{}.dev'.format(k,m,train_count))
      test_path = os.path.join(data_root, 'k{}_m{}_tr{}.test'.format(k,m,train_count))
      # fill in the templates in the language generation config template
      language_generation_text = open(language_generation_template).read()
      language_generation_text = re.sub('__BRACKET_TYPES__', str(k), language_generation_text)
      language_generation_text = re.sub('__TRAIN_MAX_LEN__', str(train_max), language_generation_text)
      language_generation_text = re.sub('__TRAIN_MIN_LEN__', str(train_min), language_generation_text)
      language_generation_text = re.sub('__TEST_MAX_LEN__', str(test_max), language_generation_text)
      language_generation_text = re.sub('__TEST_MIN_LEN__', str(test_min), language_generation_text)
      language_generation_text = re.sub('__STACK_DEPTH__', str(m), language_generation_text)
      language_generation_text = re.sub('__DEV_COUNT__', str(dev_count), language_generation_text)
      language_generation_text = re.sub('__TEST_COUNT__', str(test_count), language_generation_text)
      language_generation_text = re.sub('__TRAIN_COUNT__', str(train_count), language_generation_text)
      language_generation_text = re.sub('__TRAIN_PATH__', train_path, language_generation_text)
      language_generation_text = re.sub('__DEV_PATH__', dev_path, language_generation_text)
      language_generation_text = re.sub('__TEST_PATH__', test_path, language_generation_text)
      output_filename = 'dyckkm_m{}_k{}_trainc{}.yaml'.format(m,k,train_count)
      with open(os.path.join(language_generation_root, output_filename), 'w') as fout:
        fout.write(language_generation_text.strip())

      # Now walk through the variation in the language models trained to generate
      # these languages
      embedding_dim = (2*k+10)
      #hidden_dim = round(m*(3*math.log(k,2)-2))+10 # old
      hidden_dim = round(m*(3*math.log(k,2)))-m # new
      print('k', k, 'm', m, 'embed', embedding_dim, 'hidden', hidden_dim, 'trainmin', train_min, 'trainmax', train_max)
      learning_rate = 0.001 if (train_count >= 10000000 or (train_count >= 1000000 and k==128)) else 0.01
      for rnn_type in ['LSTM']:
        for seed in [0, 1, 2]:
          experiment_generation_text = open(experiment_generation_template).read()
          experiment_generation_text = re.sub('__BRACKET_TYPES__', str(k), experiment_generation_text)
          experiment_generation_text = re.sub('__TRAIN_MAX_LEN__', str(train_max), experiment_generation_text)
          experiment_generation_text = re.sub('__TRAIN_MIN_LEN__', str(train_min), experiment_generation_text)
          experiment_generation_text = re.sub('__TEST_MAX_LEN__', str(test_max), experiment_generation_text)
          experiment_generation_text = re.sub('__TEST_MIN_LEN__', str(test_min), experiment_generation_text)
          experiment_generation_text = re.sub('__STACK_DEPTH__', str(m), experiment_generation_text)
          experiment_generation_text = re.sub('__DEV_COUNT__', str(dev_count), experiment_generation_text)
          experiment_generation_text = re.sub('__TEST_COUNT__', str(test_count), experiment_generation_text)
          experiment_generation_text = re.sub('__TRAIN_COUNT__', str(train_count), experiment_generation_text)
          experiment_generation_text = re.sub('__LEARNING_RATE__', str(learning_rate), experiment_generation_text)
          experiment_generation_text = re.sub('__TRAIN_PATH__', train_path, experiment_generation_text)
          experiment_generation_text = re.sub('__DEV_PATH__', dev_path, experiment_generation_text)
          experiment_generation_text = re.sub('__TEST_PATH__', test_path, experiment_generation_text)

          experiment_generation_text = re.sub('__EMBEDDING_DIM__', str(embedding_dim),
              experiment_generation_text)
          experiment_generation_text = re.sub('__HIDDEN_DIM__', str(hidden_dim),
              experiment_generation_text)
          experiment_generation_text = re.sub('__RNN_TYPE__', rnn_type,
              experiment_generation_text)
          experiment_generation_text = re.sub('__SEED__', str(seed),
              experiment_generation_text)
          output_filename = 'lm{}_seed{}_m{}_k{}_trainc{}.yaml'.format(rnn_type, seed, m,k,train_count)
          #reporting_path = output_filename # for the cluster
          reporting_path = '.' # for codalab
          experiment_generation_text = re.sub('__REPORTING_PATH__', reporting_path,
              experiment_generation_text)
          with open(os.path.join(experiment_generation_root, output_filename), 'w') as fout:
            fout.write(experiment_generation_text.strip())



