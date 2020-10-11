""" Training loop for LMs, with mostly hard-coded decisions.
"""
import sys
import math

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import utils


def train(args, lm, train_batches, dev_batches):
  """Trains the language model with Adam,

  Uses a learning rate annealing-on-plateau scheme,
  early stopping after 3 consecutive epochs bearing no improvement.

  Arguments:
    lm: a LanguageModel object
    train_batches: PyTorch DataLoader of training data from Dataset
    dev_batches: PyTorch DataLoader of dev data from Dataset
  """
  lm_params_path = utils.get_lm_path_of_args(args)
  optimizer = optim.Adam(lm.parameters(), args['training']['learning_rate'])
  scheduler_patience = 0
  max_epochs = args['training']['max_epochs']
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=scheduler_patience)
  steps_between_evals = len(train_batches)
  min_dev_loss = sys.maxsize
  min_dev_loss_epoch = -1
  loss = nn.CrossEntropyLoss()
  torch.save(lm.state_dict(), lm_params_path)
  total_gradient_steps = 0
  for epoch_index in tqdm(range(max_epochs), desc='[training]'):
    epoch_train_loss = 0
    epoch_dev_loss = 0
    train_batch_count = 0
    for observation_batch, label_batch, _ in tqdm(train_batches):
      # Compute forward, backward, and take gradient step
      train_batch_count+= 1
      lm.train()
      batch_size, seq_len = label_batch.size()
      logits, _ = lm(observation_batch)
      logits = logits.view(batch_size*seq_len, -1)
      label_batch = label_batch.view(batch_size*seq_len)
      batch_loss = loss(logits, label_batch)
      batch_loss.backward()
      optimizer.step()
      epoch_train_loss += batch_loss.detach().cpu().numpy()
      optimizer.zero_grad()
      total_gradient_steps += 1
      # Determine whether it's time to evaluate on dev data
      if total_gradient_steps % steps_between_evals == 0 and total_gradient_steps > 1:
        dev_batch_count = 0
        # Compute dev loss
        for observation_batch, label_batch, _ in tqdm(dev_batches, desc='[dev batch]', smoothing=0.01):
          dev_batch_count+= 1
          optimizer.zero_grad()
          lm.eval()
          batch_size, seq_len = label_batch.size()
          logits, _ = lm(observation_batch)
          logits = logits.view(batch_size*seq_len, -1)
          label_batch = label_batch.view(batch_size*seq_len)
          batch_loss = loss(logits, label_batch)
          epoch_dev_loss += batch_loss.detach().cpu().numpy()
        scheduler.step(epoch_dev_loss)
        epoch_dev_loss = epoch_dev_loss/ dev_batch_count
        epoch_train_loss = epoch_train_loss/ train_batch_count
        tqdm.write('[epoch {}] Train loss: {}, Dev loss: {}'.format(epoch_index,
            math.pow(epoch_train_loss,2), math.pow(epoch_dev_loss,2)))
        # If new best dev loss, save parameters.
        if epoch_dev_loss < min_dev_loss - 0.0001:
          torch.save(lm.state_dict(), lm_params_path)
          min_dev_loss = epoch_dev_loss
          min_dev_loss_epoch = epoch_index
          tqdm.write('Saving lm parameters')
        elif min_dev_loss_epoch < epoch_index - 2:
          tqdm.write('Early stopping')
          tqdm.write("Min dev loss: {}".format(math.pow(min_dev_loss,2)))
          return
    tqdm.write("Min dev loss: {}".format(math.pow(min_dev_loss,2)))
