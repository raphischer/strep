# git clone https://github.com/RobustBench/robustbench

import os
import json
import pandas as pd

model_root = os.path.join(os.path.dirname(__file__), 'robustbench', 'model_info')

def read_json(filepath):
    with open(filepath, 'r') as logf:
        return json.load(logf)

dfs = []
for dataset in os.listdir(model_root):
    for threat_model in os.listdir(os.path.join(model_root, dataset)):
        dirn = os.path.join(model_root, dataset, threat_model)
        json_file_names = [fname for fname in os.listdir(dirn) if 'json' in fname and 'Standard' not in fname]
        rows = [read_json(os.path.join(dirn, fname)) for fname in json_file_names]
        df = pd.DataFrame.from_records(rows)
        df['model'] = [fname.replace('.json', '') for fname in json_file_names]
        df['dataset'] = dataset
        df['threat_model'] = threat_model
        dfs.append(df)

dfs_merged = pd.concat(dfs)

# fill empty strings and null values
dfs_merged = dfs_merged.replace('null', pd.NA).convert_dtypes()
dfs_merged = dfs_merged.replace('', pd.NA).convert_dtypes()

# set config and env information
dfs_merged['environment'] = 'Tesla V100 - PyTorch 1.7.1'
dfs_merged['task'] = 'robust test'
dfs_merged['configuration'] = dfs_merged['model'] + ' - ' + dfs_merged['threat_model']

# convert numeric columns
dfs_merged['clean_acc'] = dfs_merged['clean_acc'].astype(float)
dfs_merged['autoattack_acc'] = dfs_merged['autoattack_acc'].astype(float)
dfs_merged['reported'] = dfs_merged['reported'].astype(float)
dfs_merged['corruptions_acc_3d'] = dfs_merged['corruptions_acc_3d'].astype(float)
dfs_merged['corruptions_acc'] = dfs_merged['corruptions_acc'].astype(float)

dfs_merged.reset_index(inplace=True)

dfs_merged.to_pickle(os.path.join(os.path.dirname(__file__), 'database.pkl'))
