# FIRST! (locally) git clone https://github.com/RobustBench/robustbench

import os
import json
import pandas as pd
import numpy as np

model_root = os.path.join(os.path.dirname(__file__), 'robustbench', 'model_info')

def read_json(filepath):
    with open(filepath, 'r') as logf:
        return json.load(logf)

model_meta = {}
dfs = []
for dataset in os.listdir(model_root):
    for threat_model in os.listdir(os.path.join(model_root, dataset)):
        dirn = os.path.join(model_root, dataset, threat_model)
        json_file_names = [fname for fname in os.listdir(dirn) if 'json' in fname and 'Standard' not in fname]
        rows = [read_json(os.path.join(dirn, fname)) for fname in json_file_names]
        df = pd.DataFrame.from_records(rows)
        df['model'] = [fname.replace('.json', '') for fname in json_file_names]
        df['dataset'] = dataset # f'{dataset} ({threat_model})'
        df = df.rename({'corruptions_acc': f'{threat_model}_acc'}, axis=1)
        df = df.rename({'reported': f'{threat_model}_reported'}, axis=1)
        dfs.append(df)

dfs_merged = pd.concat(dfs)

# fill empty strings and null values
dfs_merged = dfs_merged.replace('null', pd.NA).convert_dtypes()
dfs_merged = dfs_merged.replace('', pd.NA).convert_dtypes()

# aggregate results for multiple models, and extract meta information
# "deeprenewal": {
#         "name": "Deep Renewal Processes",
#         "short": "DRP",
#         "local/global": "Global",
#         "data": "Univariate",
#         "method": "RNN",
#         "implementation": "MXNet",
#         "paper": "T?rkmen et al. 2021",
#         "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764",
#         "module": "gluonts.mx.model.renewal._estimator",
#         "class": "DeepRenewalProcessEstimator"
model_meta = {}
dfs_agg = []
cols_to_delete = ['link', 'name', 'authors', 'additional_data', 'number_forward_passes', 'venue', 'architecture', 'footnote']
for (ds, model), data in dfs_merged.groupby(['dataset', 'model']):
    # make sure that all relevant columns store the same meta info
    unq_vals = [pd.unique(data[col]).size > 1 for col in cols_to_delete]
    if np.any(unq_vals):
        print(f'WARNING! Multiple errors encountered in {model} data for fields:', ', '.join(cols_to_delete[idx] for idx in np.where(unq_vals)[0]))
    agg = data.fillna(method='bfill').iloc[0]
    # unify the author notation
    if ' and ' in agg['authors']:
        old_auth = str(agg['authors'])
        if len(agg['authors'].split('and')) == 2: # only the last author is appended with an and
            agg['authors'] = agg['authors'].split('and')[0].strip() + ', ' + agg['authors'].split('and')[1].strip()
            agg['authors'] = agg['authors'].replace(', , ', ', ')
        else: # all authors concatenated with an and
            authors = agg['authors'].split(' and ')
            for idx, auth in enumerate(authors):
                if ',' in auth:
                    authors[idx] = ' '.join(reversed(auth.split(', ')))
            agg['authors'] = ', '.join(authors)
        print(f'reformatted {old_auth} to {str(agg["authors"])}')
    model_meta[model] = {
        'url': agg['link'],
        'paper': agg['name'],
        'architecture': agg['architecture'],
        'additional_data': str(agg['additional_data']),
        'number_forward_passes': int(agg['number_forward_passes']),
        'venue': agg['venue'],
        'authors': agg['authors'],
        'footnote': None if str(agg['footnote']) == '<NA>' else str(agg['footnote']),
        'name': f"{agg['authors'].split(',')[0].split()[1]} et al. {agg['venue']}",
        'short': f"{agg['authors'].split(',')[0].split()[1]}{agg['venue'][-2:]}"
    }    
    dfs_agg.append(agg)
dfs_merged = pd.concat(dfs_agg, axis=1).transpose()
dfs_merged = dfs_merged.drop(cols_to_delete + ['eps'], axis=1)
with open(os.path.join(os.path.dirname(__file__), 'meta_model.json'), 'w') as jf:
    json.dump(model_meta, jf, indent=4)

# set config and env information
dfs_merged['environment'] = 'Tesla V100 - PyTorch 1.7.1'
dfs_merged['task'] = 'Robustness Test'
dfs_merged['configuration'] = dfs_merged['model'] + ' - ' + dfs_merged['dataset']

# convert numeric columns
prop_meta = {}
for col in dfs_merged.columns:
    if 'acc' in col or 'corr' in col or 'report' in col:
        dfs_merged[col] = dfs_merged[col].fillna(value=np.nan).astype(float)
        prop_meta[col] = {
            "name": "Clean Accuracy",
            "shortname": "ACC",
            "weight": 0.2,
            "group": "Performance",
            "unit": "percent",
            "maximize": True
        }
        if 'report' in col:
            threat = 'L2' if 'L2' in col else ('Linf' if 'Linf' in col else 'Corruption')
            prop_meta[col]["name"], prop_meta[col]["shortname"] = f'Reported {threat} Robustness', f'{threat[:3]}Rep'
        elif 'autoattack' in col:
            prop_meta[col]["name"], prop_meta[col]["shortname"] = 'AutoAttack Robustness', 'AARob'
        elif 'corruptions' in col:
            version = ' '.join(col.split('_')[1:]).capitalize()
            prop_meta[col]["name"], prop_meta[col]["shortname"] = f'{version} Corruption Robustness', f'{version}Rob'
        if col != 'clean_acc':
            prop_meta[col]["icon"] = 'robustness'
with open(os.path.join(os.path.dirname(__file__), 'meta_properties.json'), 'w') as jf:
    json.dump(prop_meta, jf, indent=4)

dfs_merged.reset_index(inplace=True)
dfs_merged.to_pickle(os.path.join(os.path.dirname(__file__), 'database.pkl'))
