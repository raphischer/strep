import os
import math
import json
import re
import multiprocessing as mp
import sys
import time

import numpy as np
from paperswithcode import PapersWithCodeClient
from tqdm import tqdm
import pandas as pd


FDIR = os.path.dirname(__file__)
EVAL_SUBDIR = os.path.join(FDIR, 'evaluations')
STATISTICS = os.path.join(FDIR, 'statistics.csv')
COMPLETE = os.path.join(FDIR, 'database_complete.pkl')
TOP100 = os.path.join(FDIR, 'database.pkl')
PROPS = os.path.join(FDIR, 'meta_properties.json')


FORMATTER = {
    r'([\d\.]+)\,(\d*)': lambda v: v.replace('.', '').replace(',', '.'), # 1.000.000,15 => 1000000.15
    r'([\d\s]+),(\d*)': lambda v: v.replace(' ', '').replace(',', '.'), # 1 000 000,15 => 1000000.15
    r'([\d\,]+)\.(\d*)': lambda v: v.replace(',', ''), # 1,000,000.15 => 1000000.15
    r'([\d]+)\.\d[KkMBG]': lambda v: v.replace('.', '').replace('K', '00').replace('k', '00').replace('M', '00000').replace('B', '00000000').replace('G', '00000000'), # 3214.7M => 3214700000
    r'([\d]+)\.\d\d[KkMBG]': lambda v: v.replace('.', '').replace('K', '0').replace('k', '0').replace('M', '0000').replace('B', '0000000').replace('G', '0000000'), # 3214.7M => 3214700000
    r'([\d]+)\.\d\d\d[KkMBG]': lambda v: v.replace('.', '').replace('K', '').replace('k', '').replace('M', '000').replace('B', '000000').replace('G', '000000'), # 3214.7M => 3214700000
    r'([\d]+)[KkMBG]': lambda v: v.replace('K', '000').replace('k', '000').replace('M', '000000').replace('B', '000000000').replace('G', '000000000'), # 3214M => 3214000000
    r'([\d\.]+)m+': lambda v: v.replace('m', '') # remove length unit, e.g. 4.13mm
}


def convert_to_float(value):
    v = value
    if isinstance(value, str):
        v = v.strip()
        if value.lower() in ['_', '-', '--', '', '.', 'n/a', 'none', '/', 'na', 'n.a.', 'unknown']:
            return np.nan
        if value.lower() in ['yes', '✓', 'true', '√']:
            return True
        if value.lower() in ['no', '✘', 'false', '×']:
            return False
        v = v.replace('..', '.') # typos
        v = v.replace(' ', '') # remove redundant white space
        v = v.replace('%', '').replace('*', '') # remove percent info or annotations
        v = v.split('(')[0].split('±')[0].split('+/-')[0].split('+-')[0] # remove standard deviation info
    try:
        return float(v)
    except Exception:
        if isinstance(v, str) and ('.' in v or ',' in v or 'K' in v or 'M' in v or 'B' in v or 'k' in v or 'G' in v or 'm' in v):
            for regex, re_func in FORMATTER.items():
                match = re.match(regex, v)
                if match:
                    try:
                        a = float(re_func(v))
                        return a
                    except Exception:
                        continue
    return np.nan


def collect_evaluations(n_workers=-1, page_size=500):
    client = PapersWithCodeClient()
     
    # collect eval IDs
    eval_ids = []
    curr_page = client.evaluation_list(items_per_page=page_size)
    while curr_page is not None:
        solved = len(curr_page.results) * (curr_page.next_page - 1) if curr_page.next_page else curr_page.count
        print(f'  loading ids {solved/curr_page.count*100:4.1f}% ({solved:<5} / {curr_page.count})', end='\r')
        for eval in curr_page.results:
            eval_ids.append(eval.id)
        try:
            curr_page = client.evaluation_list(page=curr_page.next_page, items_per_page=page_size)
        except Exception:
            curr_page = None
    
    # collect results for all evals (with multiprocessing)
    stats = []
    ids_with_page_size = [(eval_id, page_size) for eval_id in eval_ids] # needs to be combined for multiprocessing
    if n_workers is None: # single core processing
        for eval_id in tqdm(ids_with_page_size):
            stats.append( collect_eval_metrics( eval_id) )
    else: # use all available CPUs
        if n_workers == -1:
            n_workers = mp.cpu_count()
        with mp.Pool(processes=n_workers) as pool:
            for eval_stats in tqdm(pool.imap(collect_eval_metrics, ids_with_page_size), total=len(ids_with_page_size), desc=f'assessing evaluation metrics pages with {n_workers} workers'):
                stats.append(eval_stats)
    
    # write statistics to file
    stats = pd.DataFrame(stats).sort_values('eval_id')
    stats.to_csv(STATISTICS, index=False)
    return stats


def collect_eval_metrics(args):
    eval_id, page_size, eval_df = args[0], args[1], 0
    while isinstance(eval_df, int) and eval_df < 100 and page_size > 1: # max retries
        try:
            client = PapersWithCodeClient()
            eval_metrics = []
            curr_page = client.evaluation_result_list(evaluation_id=eval_id, items_per_page=page_size)
            while curr_page is not None:
                for res in curr_page.results:
                    if len(res.metrics) > 0:
                        res.metrics['methodology_STREP$'] = res.methodology
                        res.metrics['paper_STREP$'] = res.paper
                        # res.metrics['evaluated_on_STREP$'] = res.evaluated_on
                        eval_metrics.append(res.metrics)
                try:
                    curr_page = client.evaluation_result_list(evaluation_id=eval_id, page=curr_page.next_page, items_per_page=page_size)
                except Exception:
                    curr_page = None
            # find evaluation meta info
            eval_info = client.evaluation_get(eval_id)
            idx = (eval_info.dataset, eval_info.task)
            eval_df = pd.DataFrame() if len(eval_metrics) == 0 else pd.DataFrame(eval_metrics, index=[idx]*len(eval_metrics))
        except Exception as e: # sometimes the http coonnection fails, so we retry
            time.sleep(1)
            eval_df += 1
            page_size //= 2
            continue
    results = {'eval_id': eval_id, 'assessed': time.strftime('%d %b %Y %H:%M:%S')}
    if isinstance(eval_df, int):
        print(f'Failed to retrieve evaluation {eval_id} (killed after {eval_df} retries)')
    if isinstance(eval_df, pd.DataFrame) and eval_df.size > 0:
        eval_df = eval_df.astype(pd.SparseDtype("str", np.nan))
        eval_df.to_pickle(os.path.join(EVAL_SUBDIR, f'{eval_id}.pkl') )
        results['n_results'] = eval_df.shape[0]
        results['n_metrics'] = eval_df.shape[1] - 2
        results['incompleteness'] = eval_df.drop(['methodology_STREP$', 'paper_STREP$'], axis=1).isna().sum().sum() / (results['n_results'] * results['n_metrics'])
    else:
        results.update({'n_results': 0, 'n_metrics': 0, 'incompleteness': 1})
    return results


def to_sparse(df):
    return df.astype(pd.SparseDtype("float", np.nan))


if __name__ == '__main__':

    # loading all evaluations - TODO build up as zip archive
    if os.path.isfile(STATISTICS):
        stats = pd.read_csv(STATISTICS)
    else:
        if not os.path.isdir(EVAL_SUBDIR):
            os.makedirs(EVAL_SUBDIR)
        stats = collect_evaluations()

    MIN_RESULTS, MIN_PROPS, MAX_INCOMPL, TOPK = 10, 3, 0.5, 20

    # filter for min evals, properties, and incompleteness
    stats = stats.groupby('eval_id').last()
    filtered = stats[(stats['n_results'] >= MIN_RESULTS) & (stats['n_metrics'] >= MIN_PROPS) & (stats['incompleteness'] < MAX_INCOMPL)]
    filtered['total_score'] = filtered['n_results'] * filtered['n_metrics'] * (1 - filtered['incompleteness'])
    filtered = filtered.sort_values('total_score', ascending=False)
    # load evaluations and merge them
    try:
        eval_data = [pd.read_pickle(os.path.join(EVAL_SUBDIR, f'{eval_id}.pkl')) for eval_id in tqdm(filtered.index, 'reading evaluations')]
    except Exception as e:
        print(f'Could not load all evaluations - please re-assemble by deleting {STATISTICS} and {EVAL_SUBDIR} and re-running the script')
    sparse = pd.concat(eval_data, axis=0)
    # transform index data into columns and reset index
    sparse['dataset_STREP$'] = sparse.index.map(lambda idx: idx[0])
    sparse['task_STREP$'] = sparse.index.map(lambda idx: idx[1])
    sparse['model_STREP$'] = sparse[['methodology_STREP$', 'paper_STREP$']].apply(lambda val: '{} from {}'.format(*val.astype(str)), axis=1, raw=True)
    sparse = sparse.reset_index(drop=True)
    meta_cols = [col for col in sparse.columns if '_STREP$' in col]

    # process values to positive floats wherever possible
    sparse_metrics_stacked = sparse.drop(meta_cols, axis=1).stack()
    sparse_metrics_proc = sparse_metrics_stacked.apply(convert_to_float)
    sparse_metrics_proc = sparse_metrics_proc.map(lambda v: np.nan if isinstance(v, float) and v < 0 else v) # purge values smaller than zero
    # check for amount of successful conversions
    equal = sparse_metrics_stacked[sparse_metrics_stacked == sparse_metrics_proc]
    unequal = sparse_metrics_stacked[sparse_metrics_stacked != sparse_metrics_proc]
    print(f'Could not convert {equal.size / sparse_metrics_proc.size * 100:.2f}% of PWC metric values to float ({pd.unique(equal).size} / {pd.unique(unequal).size} in total).')
    print('The following metric values could not be transformed:', pd.unique(equal).to_dense())
    sparse_metrics_proc = sparse_metrics_proc.unstack()
    # merge the processed metrics and meta info # TODO also remember and store the setup information (aggregated from non-conversible column values)
    sparse_new = pd.concat([sparse[meta_cols], to_sparse(sparse_metrics_proc)], axis=1)
    del equal, unequal, sparse_metrics_stacked, sparse_metrics_proc, sparse
    # check the individual values and sparsity level of each task and dataset
    ok_data_per_task_and_ds = []
    for (task, ds) in pd.unique(sparse_new[['task_STREP$', 'dataset_STREP$']].to_records(index=False)):
        # drop all values that extremely fall out of range (for example to account for 1.500.000 parameters vs 1.5 M parameters)
        sub_sparse = sparse_new[(sparse_new['task_STREP$'] == task) & (sparse_new['dataset_STREP$'] == ds)].dropna(how='all', axis=1)
        for col in sub_sparse.select_dtypes('number').columns:
            values = sub_sparse[col].astype(float).dropna()
            # TODO check if there is more than two values in the column, otherwise purge because uninformative
            if values.size > 0:
                # Calculate skewness of original and log-transformed values, use log if more appropriate
                c_original_skewness = pd.Series(values).skew()
                log_values = np.log1p(values)
                c_log_skewness = pd.Series(log_values).skew()
                if c_log_skewness < c_original_skewness:
                    values = log_values
                # Calculate acceptable range based on IQR on the chosen scale
                c_Q1, c_Q3 = np.percentile(values, [10, 90])
                c_IQR = c_Q3 - c_Q1
                c_lower_bound, c_upper_bound = c_Q1 - c_IQR * 2, c_Q3 + c_IQR * 2
                if c_log_skewness < c_original_skewness:
                    c_lower_bound, c_upper_bound = np.expm1(c_lower_bound), np.expm1(c_upper_bound)
                    c_Q1, c_Q3 = np.expm1(c_Q1), np.expm1(c_Q3)
                # Identify outliers in the original or log-transformed scale
                to_drop = ~(sub_sparse[col].between(c_lower_bound, c_upper_bound) | sub_sparse[col].isna())
                if to_drop.sum() > 0:
                    print(f'{col:<30} - Q: {c_Q1:15.1f} {c_Q3:15.1f} - drop {to_drop.sum():<2} / {values.size:<3} vals: {pd.unique(sub_sparse[col][to_drop]).tolist()}')
                    discard_func = lambda v: v if not np.isnan(v) and c_lower_bound <= v <= c_upper_bound else np.nan # TODO make more efficient
                    sub_sparse[col] = to_sparse(sub_sparse[col].astype(float).map(discard_func))
        # TODO add another check for the sub_sparse: how many numeric columns remain? what is the sparsity level? only add if ok!
        ok_data_per_task_and_ds.append(sub_sparse)

    # merge ok evaluations
    sparse_new = pd.concat(ok_data_per_task_and_ds, axis=0).dropna(how='all', axis=1)
    sparse_level = sparse_new.select_dtypes('number').stack().size / sparse_new.select_dtypes('number').size
    print(f'SPARSITY LEVEL: only {sparse_level*100:.2f}% of the DB is populated, with ({str(sparse_new.select_dtypes("number").shape)} numeric properties')
    assert np.all(sparse_new.index == sparse.index)

    # rename the meta columns (and check for duplicates!)
    renamed = {}
    for col in meta_cols:
        without_tag = col.replace('_STREP$', '')
        if without_tag in sparse_new.columns:
            sparse_new = sparse_new.rename({without_tag: f'{without_tag}_PWC$'}, axis=1)
            renamed[without_tag] = f'{without_tag}_PWC$'
        sparse_new = sparse_new.rename({col: without_tag}, axis=1)
    meta_cols = [col.replace('_STREP$', '') for col in meta_cols]
    print(':::::::::::::::: PURGED UNINFORMATIVE EVALUATIONS FOR SPARSELY REPORTED DATASETS AND TASKS FROM FULL DB', sparse_new.shape)

    # save evaluation databases to disk
    assert 'environment' not in sparse_new.columns
    sparse_new.to_pickle(COMPLETE)
    top_k = []
    for eval in eval_data[:TOPK]:
        ds, task = eval.index[0]
        proc_results = sparse_new[(sparse_new['task'] == task) & (sparse_new['dataset'] == ds)].dropna(how='all', axis=1)
        top_k.append( proc_results )
    best_evals = pd.concat(top_k, axis=0)
    assert best_evals.shape[0] == pd.unique(best_evals.index).size
    best_evals.reset_index(drop=True).to_pickle(TOP100)

    # assess the meta information of the properties
    res_metrics = ['time', 'param', 'size', 'flops', 'latenc', 'operation', 'emission']
    client = PapersWithCodeClient()
    group_res = {'Resources': [], 'Performance': []}
    properties, possible_errors = {}, {}
    sparse_ds = pd.unique(sparse_new['dataset'])
    # check metrics of each of the filtered evaluations
    missed_metrics = set()
    for eval_id in tqdm(filtered.index, 'assessing evaluation metrics'):
        metrics = client.evaluation_metric_list(eval_id)
        assert metrics.next_page == None
        # store meta info for every metric that is entailed in the processed (ie., discard non-numeric metrics)
        for metr in metrics.results:
            if metr.name in sparse_new.columns and metr.name not in meta_cols:
                # identify group based on metric name - TODO - can this be improved?
                unified = metr.name.lower().replace(' = ', '').replace(' ', '').replace('-', '').replace('_', '').replace('(', '').replace(')', '').replace('%', '').replace(',', '')
                group = 'Resources' if any([key in unified for key in res_metrics]) else 'Performance'
                m_name = metr.name if metr.name not in renamed else f'{metr.name}_PWC$'
                group_res[group].append(m_name)
                if m_name in properties: # multiple values found => store the is_loss info (might be contradictory!)
                    if m_name not in possible_errors:
                        possible_errors[m_name] = []
                    possible_errors[m_name].append(not metr.is_loss)
                else:
                    properties[m_name] =  {
                        "name": m_name, "shortname": unified[:6], "unit": "number",
                        "group": group, "weight": 1, "maximize": not metr.is_loss,
                    }
            else:
                missed_metrics.add(metr.name)
    print(f'the following metrics were not formalized into meta file, as they do not feature numeric data or are found in meta cols\n{missed_metrics}\n')

    print('RESOURCE METRICS:\n', set(group_res['Resources']), '\n\n PERFORMANCE METRICS:\n', set(group_res['Performance']), '\n\nPOSSIBLE_ERRORS ACROSS TASKS\n', possible_errors.keys())
    for metr, is_loss in possible_errors.items():
        # take the is_loss information that occurs most frequently
        properties[metr]['maximize'] = bool(np.median(is_loss))

    # override with some matched information
    for prop, vals in properties.items():
        p = prop.lower()
        if 'param' in p or 'flop' in p or 'time' in p or 'operation' in p or 'latenc' in p or 'emission' in p:
            if vals['maximize']:
                properties[prop]['maximize'] = False
                print('manually overrode "maximize" to "False" for', prop)
        if ('acc' in p or 'top' in p) and ('error' not in p and 'rate' not in p):
            if not vals['maximize']:
                properties[prop]['maximize'] = True
                properties[prop]['unit'] = 'percent'
                print('manually overrode "maximize" to "True" for', prop)


    with open(PROPS, 'w') as jf:
        json.dump(properties, jf)
    
    print('done', pd.__version__)
