#!/usr/bin/env python3

import argparse
from Bio import SeqIO
import numpy as np
from collections import Counter
from scipy.stats import entropy
import sys

# ---- Argument parsing ----
parser = argparse.ArgumentParser()
parser.add_argument('index_reference', type=argparse.FileType('r'))
parser.add_argument('number_of_output_indices', type=int)
parser.add_argument('--used_indices', '-u', type=argparse.FileType('r'))
parser.add_argument('--number_of_seeds', '-n', type=int, default=8)
parser.add_argument('--number_of_replicates', '-r', type=int, default=1000)
parser.add_argument('--seed', '-s', type=int, default=123456)
args = parser.parse_args()

if args.number_of_output_indices <= args.number_of_seeds:
    args.number_of_seeds = args.number_of_output_indices

# ---- Load reference sequences and IDs ----
seqs = []
ids  = []
for record in SeqIO.parse(args.index_reference, 'fasta'):
    seqs.append(list(str(record.seq)))
    ids.append(record.id)

ref_index = np.array(seqs)
id_index  = np.array(ids, dtype=object)

# ---- Entropy-based scoring function ----
def sqdevsum(mat):
    N = mat.shape[0]
    total = 0.0
    for col in range(mat.shape[1]):
        counts = Counter(mat[:, col])
        freq   = np.array([counts.get(nuc,0) for nuc in 'ATGC']) / N
        total += entropy(freq, np.array([0.25]*4))
    return total

np.random.seed(args.seed)

# ---- Pick initial “best” set ----
if args.used_indices:
    # read in already-used sequences
    used_seqs = [list(str(r.seq)) for r in SeqIO.parse(args.used_indices, 'fasta')]
    best = np.array(used_seqs)
    # find and record their IDs, then remove them from the pool
    used_rows = []
    best_ids  = []
    for u in used_seqs:
        for i, ref_row in enumerate(ref_index):
            if np.array_equal(u, ref_row):
                used_rows.append(i)
                best_ids.append(id_index[i])
                break
    ref_index = np.delete(ref_index, used_rows, axis=0)
    id_index  = np.delete(id_index, used_rows, axis=0)

else:
    best_score = float('inf')
    for _ in range(args.number_of_replicates):
        rows = np.random.choice(ref_index.shape[0],
                                args.number_of_seeds,
                                replace=False)
        score = sqdevsum(ref_index[rows])
        if score < best_score:
            best_score = score
            best_rows  = rows
    best     = ref_index[best_rows]
    best_ids = id_index[best_rows]
    # remove those from the pool
    ref_index = np.delete(ref_index, best_rows, axis=0)
    id_index  = np.delete(id_index, best_rows, axis=0)

# ---- Greedily add sequences until we have enough ----
while best.shape[0] < args.number_of_output_indices and len(ref_index) > 0:
    best_score = float('inf')
    best_row   = None
    for i in range(ref_index.shape[0]):
        score = sqdevsum(np.vstack([best, ref_index[i]]))
        if score < best_score:
            best_score = score
            best_row   = i
    best = np.vstack([best, ref_index[best_row]])
    best_ids = np.append(best_ids, id_index[best_row])
    ref_index = np.delete(ref_index, best_row, axis=0)
    id_index  = np.delete(id_index, best_row, axis=0)

# ---- Print entropy stats to stderr ----
for col in range(best.shape[1]):
    count = Counter(best[:, col])
    N     = best.shape[0]
    print(
        'Position: {}, A: {:02.2f}%, T: {:02.2f}%, G: {:02.2f}%, C: {:02.2f}%'.format(
            col+1,
            count.get('A',0)/N*100,
            count.get('T',0)/N*100,
            count.get('G',0)/N*100,
            count.get('C',0)/N*100,
        ),
        file=sys.stderr
    )

# ---- Print the selected sequences with their original IDs ----
for seq_id, row in zip(best_ids, best):
    print(f">{seq_id}")
    print(''.join(row))

