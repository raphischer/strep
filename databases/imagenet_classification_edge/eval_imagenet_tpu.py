import os
import pandas as pd
import numpy as np

dfs = []

for idx in [1, 2, 3]:
    fname = os.path.join('databases', 'imagenet_classification_edge', f'lamarrws01_clean_classification_lauf_{idx}.pkl')
    df = pd.read_pickle(fname)
    df['total_parameters'] = df['total_parameters'].astype(int)
    df['trainable_parameters'] = df['trainable_parameters'].astype(int)
    df['non_trainable_parameters'] = df['non_trainable_parameters'].astype(int)
    df['configuration'] = df['task'] + ' - ' + df['dataset'] + ' - ' + df['model']
    df['environment'] = df['backend']
    df = df.drop(['log_name', 'backend'], axis=1)
    df = df.dropna()
    dfs.append(df)

num_cols = list(df.select_dtypes('number').columns)
non_num = list(df.select_dtypes(exclude='number').columns)

final = dfs[0].loc[:,non_num].copy()

for group, data in final.groupby(non_num):
    for col in num_cols:
        vals = []
        for df in dfs:
            row = df[(df.loc[:,non_num] == group).all(axis=1)]
            if row.shape[0] != 1:
                raise RuntimeError("row.shape[0] > 1")
            vals.append(row[col].values[0])
        final.loc[data.index,col] = np.mean(vals)

pd.to_pickle(final, os.path.join('databases', 'imagenet_classification_edge', 'database.pkl'))