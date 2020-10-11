"""
Constructs the plot in the EMNLP 2020 (main) paper,

    RNNs can generate bounded hierarchical languages with optimal memory

Usage:

    python make_emnlp_plot.py
"""

import yaml
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import utils
import seaborn as sns
sns.set_style("whitegrid")


ms = [3, 5]
ks = [2, 8, 32, 128]
train_line_counts = [2000, 20000, 200000, 2000000,20000000]
seeds = [0, 1, 2]

# Read results from dict
results_dict = {}
for elt in itertools.product(ms, ks, train_line_counts, seeds):
  m, k, train_count, seed = elt
  #output_filename = 'configs/emnlp2020/experiments/lmLSTM_seed{}_m{}_k{}_trainc{}.yaml'.format(seed, m,k, train_count)
  #results_dir = utils.get_results_dir_of_args(yaml.load(open(output_filename)))
  results_dir = 'compile-results/data/lmLSTM_seed{}_m{}_k{}_trainc{}.yaml'.format(seed, m,k, train_count)
  dev_result = [x[0] for x in json.load(open(os.path.join(results_dir, 'summary-dev.json')))]
  test_result = [x[0] for x in json.load(open(os.path.join(results_dir, 'summary-test.json')))]
  results_dict[(m,k,train_count,seed,'dev')] = dev_result
  results_dict[(m,k,train_count,seed,'test')] = test_result


# Aggregate across distances and seeds
#results_minima = {k: np.quantile(v,.0) for k,v in results_dict.items()}
results_minima = {k: np.mean(v) for k,v in results_dict.items()}
results_seed_averages = {(m, k, train_count, split): np.median(list(filter(lambda x: x is not None, [results_minima[(m,k,train_count,seed,split)] if 
 (m,k,train_count,seed,split) in results_minima else None for seed in seeds])))
  for (m,k,train_count,seed,split) in results_minima}

fig = plt.figure()
fig.set_figheight(3.5)
fig.set_figwidth(5)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# Make m3 plots
m3_xs = np.array(train_line_counts) #[train_token_counts[(3, x)] for x in train_line_counts]
for i, k in enumerate(ks):
  print('m=3,k={},tr=20m','median over seeds of mean over lengths:', results_seed_averages[(3,k,20000000, 'test')])
  m3_dev_ys = [results_seed_averages[(3, k, x, 'dev')] for x in train_line_counts]
  m3_test_ys = [results_seed_averages[(3, k, x, 'test')] for x in train_line_counts]
  line, = ax1.plot(m3_xs, m3_dev_ys, '--', label='k={},dev'.format(k), alpha=1, marker='d')
  ax1.plot(m3_xs, m3_test_ys, color=line.get_color(),alpha=1, marker='H',label='k={},test'.format(k))
  ax1.set_ylabel('m=3')
  ax1.set_xscale('log')

# Make m5 plots
m3_xs = np.array(train_line_counts)# [train_token_counts[(5, x)] for x in train_line_counts]
for i, k in enumerate(ks):
  print('m=5,k={},tr=20m','median over seeds of mean over lengths:', results_seed_averages[(5,k,20000000, 'test')])
  m3_dev_ys = [results_seed_averages[(5, k, x, 'dev')] for x in train_line_counts]
  m3_test_ys = [results_seed_averages[(5, k, x, 'test')] for x in train_line_counts]
  line, = ax2.plot(m3_xs, m3_dev_ys, '--',alpha=1,marker='d')#,  label='k={},m=5,dev'.format(k))
  ax2.plot(m3_xs, m3_test_ys, color=line.get_color(),alpha=1, marker='H')#,label='k={},m=3,test'.format(k))
  ax2.set_xscale('log')
  ax2.set_ylabel('m=5')

# Axis labels
text1 = fig.text(-0.03, 0.5, 'Average bracket-closing accuracy', va='center', rotation='vertical')
text2 = fig.text(0.5, 0.00, 'Training data (tokens)', ha='center')
lgd = fig.legend(fontsize=8, bbox_to_anchor=(1.05,0.5), loc='center')#loc='lower left')
text3 = fig.suptitle('Learning Dyck-$(k,m)$')
sns.despine(fig=fig)
fig.savefig('plt.pdf', bbox_extra_artists=(text1, text2, lgd, text3), bbox_inches='tight' )
fig.savefig('plt.png', bbox_extra_artists=(text1, text2, lgd, text3), bbox_inches='tight', dpi=300)
