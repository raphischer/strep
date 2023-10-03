import os
import math
import json
import re

import numpy as np
from paperswithcode import PapersWithCodeClient
from tqdm import tqdm
import pandas as pd


def collect_evaluation_meta(client):
    evals_meta = {}
    init_list = client.evaluation_list()
    n_pages = math.ceil(init_list.count / len(init_list.results))
    for page in tqdm(range(n_pages), total=n_pages, desc='assessing evaluations'):
        results = client.evaluation_list(page=page + 1).results
        for res in results:
            evals_meta[res.id] = (res.dataset, res.task)
        # if page > 3:
        #     break
    return evals_meta


def uniform_metric(key):
    return key.lower().strip().replace('-', '_').replace(' ', '_')


def remove_digit_group_separator(value):
    if isinstance(value, str) and ('.' in value or ',' in value):
        # variant: 1.000.000,15
        match = re.match(r'([\d\.]+),?(\d*)', value)
        if match:
            return value.replace('.', '').replace(',', '.')
        # variant: 1 000 000,15
        match = re.match(r'([\d\s]+),?(\d*)', value)
        if match:
            return value.replace(' ', '').replace(',', '.')
        # variant: 1,000,000.15
        match = re.match(r'([\d\,]+).?(\d*)', value)
        if match:
            return value.replace(',', '')
    return value



def collect_eval_metrics(client, eval_id, eval_meta):
    init_list = client.evaluation_result_list(evaluation_id=eval_id)
    metrics = set(['methodology', 'paper', 'evaluated_on'])
    for res in init_list.results:
        metrics = metrics.union(set(list(res.metrics.keys())))
    other_stats = {'n_results': init_list.count, 'n_metrics': len(metrics)}

    eval_metrics = []
    if init_list.count < 1:
        return None, other_stats
    n_pages = math.ceil(init_list.count / len(init_list.results))
    for page in range(n_pages):
        if page > 0: # first page already loaded
            res_list = client.evaluation_result_list(evaluation_id=eval_id, page=page+1)
        else:
            res_list = init_list
        for res in res_list.results:
            if len(res.metrics) > 0:
                # uniformly rename metrics and remove , in values (sometimes used as digit group separator)
                metrics = {uniform_metric(key): remove_digit_group_separator(val) for key, val in res.metrics.items() if val}
                metrics['methodology'] = res.methodology
                metrics['paper'] = res.paper
                metrics['evaluated_on'] = res.evaluated_on
                eval_metrics.append(metrics)
    df = pd.DataFrame(eval_metrics)
    
    changed = True
    count = 0
    while changed: # drop until no more changes
        old_shape = df.shape
        df = df.loc[df.dropna(how='all').index, df.dropna(how='all',axis=1).columns] # drop complete nan rows and cols
        changed = df.shape != old_shape
        count += 1
    df['dataset'] = eval_meta[0]
    df['task'] = eval_meta[1]
    df = df.reset_index(drop=True)
    return df, other_stats


def filter_min_rows(df, min_rows=10):
    keep = []
    for _, data in tqdm(df.groupby(['dataset', 'task'])):
        if data.shape[0] > min_rows - 1:
            keep.extend(data.index.to_list())
    return keep


def filter_min_props(df, min_props=3, populated=0.0):
    keep = []
    for _, data in tqdm(df.groupby(['dataset', 'task'])):
        data = data.dropna(how='all', axis=1)
        if populated > 0:
            sparse_populated_cols = [col for col in data.columns if data[col].dropna().size / data.shape[0] < populated]
            data = data.drop(columns=sparse_populated_cols)
        props = []
        for col in data.columns:
            try:
                cf = data[col].astype(float)
                props.append(col)
            except Exception:
                pass

        if len(props) >= min_props:
            keep.extend(data.index.to_list())
    return keep


if __name__ == '__main__':
    FDIR = os.path.dirname(__file__)
    COMPLETE = os.path.join(FDIR, 'pwc_complete.pkl')
    FILTERED = os.path.join(FDIR, 'database.pkl')
    OTHER = os.path.join(FDIR, 'other_stats.json')
    PROPS = os.path.join(FDIR, 'meta_properties.json')
    FILTER_STATS = os.path.join(FDIR, 'filterstats.json')

    client = PapersWithCodeClient()

    # loading all evaluations
    if os.path.isfile(COMPLETE):
        merged = pd.read_pickle(COMPLETE)
        with open(OTHER, 'r') as jf:
            other_stats = json.load(jf)
    else:
        evals_meta = collect_evaluation_meta(client)
        evals_metrics, other_stats = {}, {}

        for eval_id, eval_meta in tqdm(evals_meta.items(), total=len(evals_meta), desc='assessing evaluation metrics'):
            df, other = collect_eval_metrics(client, eval_id, eval_meta)
            other_stats[eval_id] = other
            if df is not None:
                df.to_pickle(f'results/{eval_id}.pkl')
                evals_metrics[eval_id] = df
        with open(OTHER, 'w') as jf:
            json.dump(other_stats, jf)
        merged = pd.concat(list(evals_metrics.values())).reset_index(drop=True)
        merged = merged.rename(columns={'model': 'model_name'})
        merged['model'] = merged[['methodology', 'paper']].apply(lambda val: '{} from {}'.format(*val.astype(str)), axis=1, raw=True)
        merged['environment'] = 'unknown'
        merged['configuration'] = merged[['task', 'dataset', 'model']].apply(lambda val: '{} on {} via {}'.format(*val.astype(str)), axis=1, raw=True)
        merged = merged.astype(pd.SparseDtype("str", np.nan))
        merged.to_pickle(COMPLETE)

    # filtering for relevant evaluations
    filters = {
        'At least 10 results': filter_min_rows,
        'At least 3 properties': filter_min_props,
        'At least 50% populated properties': lambda df: filter_min_props(df, populated=0.5),
    }

    shapes = { 'Complete database': merged.shape }

    for filter, func in filters.items():
        keep = func(merged)
        merged = merged.loc[keep]
        merged = merged.dropna(how='all', axis=1) # drop all nan cols (might be redundant now)
        shapes[filter] = merged.shape
        print(filter, shapes[filter])

    merged.to_pickle(FILTERED)    
    
    # write filter stats
    with open(FILTER_STATS, 'w') as jf:
        json.dump(shapes, jf)

    # create properties json
    res_metrics = ['time', 'param', 'size', 'flops']
    properties = {}
    for task in tqdm(pd.unique(merged['task'])):
        eval_list = client.task_evaluation_list(task)
        for eval in eval_list.results:
            metrics = client.evaluation_metric_list(eval.id)
            for metr in metrics.results:
                metr_name = uniform_metric(metr.name)
                group = 'Resources' if any([key in metr_name for key in res_metrics]) else 'Performance'
                properties[metr_name] =  {
                    "name": metr.description if len(metr.description) > 0 else metr.name,
                    "shortname": metr.name,
                    "unit": "number",
                    "group": group,
                    "weight": 1,
                    "maximize": not metr.is_loss,
                }
    with open(PROPS, 'w') as jf:
        json.dump(properties, jf)
    
    for gr, data in merged.groupby(['dataset']):
        data = data.dropna(how='all', axis=1)
        n_tasks, n_papers, m_methods = pd.unique(data['task']).size, pd.unique(data['paper']).size, pd.unique(data['methodology']).size
        print(f"{gr:<45} {n_tasks:<2} tasks {n_papers:<3} papers {m_methods:<3} methods {str(data.shape):<10} results")
