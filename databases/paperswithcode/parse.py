import os
import math
import json

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


def collect_eval_metrics(client, eval_id, eval_meta):
    init_list = client.evaluation_result_list(evaluation_id=eval_id)
    metrics = set(['methodology', 'paper', 'evaluated_on'])
    for res in init_list.results:
        metrics = metrics.union(set(list(res.metrics.keys())))
    other_stats = {'n_results': init_list.count, 'n_metrics': len(metrics)}
    df = None

    if init_list.count > 9:
        eval_metrics = []
        n_pages = math.ceil(init_list.count / len(init_list.results))
        for page in range(n_pages):
            if page > 0: # first page already loaded
                res_list = client.evaluation_result_list(evaluation_id=eval_id, page=page+1)
            else:
                res_list = init_list
            for res in res_list.results:
                if len(res.metrics) > 0:
                    metrics = res.metrics
                    metrics['methodology'] = res.methodology
                    metrics['paper'] = res.paper
                    metrics['evaluated_on'] = res.evaluated_on
                    eval_metrics.append(metrics)
        df = pd.DataFrame(eval_metrics)

        possible_numerics = [col for col in df.columns if col not in ['methodology', 'paper', 'evaluated_on']]

        for col in possible_numerics:
            try:
                df[col] = df[col].map(lambda val: str(val).replace(',', '')).astype(float)
            except Exception:
                pass
        
        # filter for numeric cols with at least 20% of valid entries
        num_cols = [col for col in df.select_dtypes('number').columns if df[col].dropna().size / df[col].size > 0.2]
        df_num = df[num_cols]
        # check if there are at least 3 metrics and 10 results
        if df_num.columns.size > 2 and df_num.dropna(how='all').shape[0] > 9:
            df['dataset'] = eval_meta[0]
            df['task'] = eval_meta[1]
            if df_num.shape != df_num.dropna(how='all').shape:
                df = df.loc[df_num.dropna(how='all').index]
            df = df.rename(columns={col: col.lower().strip().replace('-', '_') for col in df})
            df = df.reset_index(drop=True)
        else:
            df = None
    return df, other_stats


if __name__ == '__main__':
    FDIR = os.path.dirname(__file__)
    MERGED = os.path.join(FDIR, 'database_new.pkl')
    OTHER = os.path.join(FDIR, 'other_stats.json')

    client = PapersWithCodeClient()

    if os.path.isfile(MERGED):
        merged = pd.read_pickle(MERGED)
    else:
        evals_meta = collect_evaluation_meta(client)
        evals_metrics, other_stats = {}, {}
        for eval_id, eval_meta in tqdm(evals_meta.items(), total=len(evals_meta), desc='assessing evaluation metrics'):
            df, other = collect_eval_metrics(client, eval_id, eval_meta)
            other_stats[eval_id] = other
            if df is not None:
                evals_metrics[eval_id] = df

        with open(OTHER, 'w') as jf:
            json.dump(other_stats, jf)
        merged = pd.concat(list(evals_metrics.values())).reset_index(drop=True)
        merged = merged.rename(columns={'model': 'model_name'})
        merged['model'] = merged[['methodology', 'paper']].apply(lambda val: '{} from {}'.format(*val.astype(str)), axis=1, raw=True)
        merged['environment'] = 'unknown'
        merged['configuration'] = merged[['task', 'dataset', 'model']].apply(lambda val: '{} on {} via {}'.format(*val.astype(str)), axis=1, raw=True)
        merged = merged.astype(pd.SparseDtype("str", np.nan))
        merged.to_pickle(MERGED)

    print(merged.shape)
    for gr, data in merged.groupby(['dataset']):
        data = data.dropna(how='all', axis=1)
        n_tasks, n_papers, m_methods = pd.unique(data['task']).size, pd.unique(data['paper']).size, pd.unique(data['methodology']).size
        print(f"{gr:<45} {n_tasks:<2} tasks {n_papers:<3} papers {m_methods:<3} methods {str(data.shape):<10} results")
