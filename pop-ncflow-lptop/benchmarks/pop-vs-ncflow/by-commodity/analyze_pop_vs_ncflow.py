#! /usr/bin/env python

import sys
import pandas as pd

fname = sys.argv[1]

df = pd.read_csv(fname)

pop_worse_df = df.query('pop_minus_ncflow < 0')
pop_better_df = df.query('pop_minus_ncflow > 0')

pop_worse_num_paths_mean = pop_worse_df['s_k_t_k_num_paths'].mean()
pop_worse_shortest_path_mean = pop_worse_df['s_k_t_k_shortest_path'].mean()
pop_worse_longest_path_mean = pop_worse_df['s_k_t_k_longest_path'].mean()

pop_better_num_paths_mean = pop_better_df['s_k_t_k_num_paths'].mean()
pop_better_shortest_path_mean = pop_better_df['s_k_t_k_shortest_path'].mean()
pop_better_longest_path_mean = pop_better_df['s_k_t_k_longest_path'].mean()

print('statistic\tPOP better\tNCFlow better')
for row in [
        ('count', len(pop_better_df), len(pop_worse_df)),
        ('longest path', pop_better_longest_path_mean, pop_worse_longest_path_mean),
        ('shortest path', pop_better_shortest_path_mean, pop_worse_shortest_path_mean),
        ('num paths', pop_better_num_paths_mean, pop_worse_num_paths_mean)
           ]:
    label, better, worse = row
    print('{}\t{}\t{}'.format(label, better, worse))
