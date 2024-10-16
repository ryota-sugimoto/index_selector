#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('index_reference', type=argparse.FileType('r'))
parser.add_argument('number_of_output_indices', type=int)
parser.add_argument('--used_indices', '-u', type=argparse.FileType('r'))
parser.add_argument('--number_of_seeds', '-n', type=int, default=8)
parser.add_argument('--number_of_replicates', '-r', type=int, default=1000)
parser.add_argument('--seed', type=int, default=123456)
args = parser.parse_args()

if args.number_of_output_indices <= args.number_of_seeds:
  args.number_of_seeds = args.number_of_output_indices

from Bio import SeqIO
ref = []
for record in SeqIO.parse(args.index_reference, 'fasta'):
  ref.append(list(str(record.seq)))

import numpy as np
ref_index = np.array(ref)

from collections import Counter
from scipy.stats import entropy
def sqdevsum(mat):
  N = mat.shape[0]
  res = []
  for col in range(mat.shape[1]):
    counts = Counter(list(mat[:,col]))
    freq = np.array(list(counts.get(nuc,0) for nuc in 'ATGC'))/N
    expected = np.array([0.25]*4)
    kl = entropy(freq, expected)
    res.append(kl)
  return sum(res)

if args.used_indices:
  l = []
  for record in SeqIO.parse(args.used_indices, 'fasta'):
    l.append(list(str(record.seq)))
  best = np.array(l)
  
  used_rows = []
  for row1 in best:
    for i,row2 in enumerate(ref_index):
      if np.array_equal(row1,row2):
        used_rows.append(i)
  ref_index = np.delete(ref_index, used_rows, axis=0)
else:
  best_score = float('inf')
  for i in range(args.number_of_replicates):
    rows = np.random.choice(ref_index.shape[0],
                            args.number_of_seeds, replace=True)
    
    score = sqdevsum(ref_index[rows])
    if score < best_score:
      best_score = score
      best_rows = rows
      best = ref_index[rows]
  
  ref_index = np.delete(ref_index, best_rows, axis=0)

np.random.seed(args.seed)

while best.shape[0] < args.number_of_output_indices and len(ref_index) > 0:
  best_score = float('inf')
  for row in range(ref_index.shape[0]):
    score = sqdevsum(np.vstack([best, ref_index[row]]))
    if score < best_score:
      best_score = score
      best_row = row
  best = np.vstack([best, ref_index[best_row]])
  ref_index = np.delete(ref_index, best_row, axis=0)

import sys
for col in range(best.shape[1]):
  count = Counter(best[:,col])
  N = best.shape[0]
  print('Position: {}, A: {:02.2f}%, T: {:02.2f}%, G: {:02.2f}%, C: {:02.2f}%'.format(col+1,
                                                    count.get('A',0)/N*100,
                                                    count.get('T',0)/N*100,
                                                    count.get('G',0)/N*100,
                                                    count.get('C',0)/N*100), file=sys.stderr)


for i,row in enumerate(best):
  print(">{}".format(i+1))
  print(''.join(row))

