import os
import math
import json
import re
import multiprocessing as mp
import time

import numpy as np
from paperswithcode import PapersWithCodeClient
from tqdm import tqdm
import pandas as pd


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
    return value


def collect_evaluations(n_workers=-1):
    client = PapersWithCodeClient()
    init_list = client.evaluation_list()
    n_pages = math.ceil(init_list.count / len(init_list.results))
    dfs = []
    if n_workers is None:
        for page in tqdm(range(n_pages)):
            dfs.append( collect_metrics_for_page(page) )
    else:
        if n_workers == -1:
            n_workers = mp.cpu_count()
        with mp.Pool(processes=n_workers) as pool:
            for page_dfs in tqdm(pool.imap(collect_metrics_for_page, np.arange(n_pages)), total=n_pages, desc='assessing evaluation metrics pages'):
                dfs.append(page_dfs)
    return pd.concat(dfs)


def collect_metrics_for_page(page):
    client = PapersWithCodeClient()
    eval_results = client.evaluation_list(page=page + 1).results
    metric_results = []
    for eval in eval_results:
        init_list = client.evaluation_result_list(evaluation_id=eval.id)
        if init_list.count < 1:
            continue
        eval_metrics = []
        n_eval_pages = math.ceil(init_list.count / len(init_list.results))
        for page in range(n_eval_pages):
            if page > 0: # first page already loaded
                res_list = client.evaluation_result_list(evaluation_id=eval.id, page=page+1)
            else:
                res_list = init_list
            for res in res_list.results:
                if len(res.metrics) > 0:
                    res.metrics['methodology_STREP$'] = res.methodology
                    res.metrics['paper_STREP$'] = res.paper
                    res.metrics['evaluated_on_STREP$'] = res.evaluated_on
                    eval_metrics.append(res.metrics)
        df = pd.DataFrame(eval_metrics, index=[eval.id]*len(eval_metrics))
        # store the evaluation meta info
        for col_name, meta in zip(['dataset', 'task'], [eval.dataset, eval.task]):
            df[f'{col_name}_STREP$'] = meta
        metric_results.append(df)
    if len(metric_results) == 0:
        return pd.DataFrame()
    return pd.concat(metric_results)


if __name__ == '__main__':
    FDIR = os.path.dirname(__file__)
    COMPLETE = os.path.join(FDIR, 'pwc_complete.pkl')
    FILTERED = os.path.join(FDIR, 'database.pkl')
    OTHER = os.path.join(FDIR, 'other_stats.json')
    PROPS = os.path.join(FDIR, 'meta_properties.json')
    FILTER_STATS = os.path.join(FDIR, 'filterstats.json')

    # loading all evaluations
    if os.path.isfile(FILTERED):
        sparse = pd.read_pickle(FILTERED)
    else:
        if os.path.isfile(COMPLETE):
            sparse = pd.read_pickle(COMPLETE)
        else:
            evals_metrics = collect_evaluations()
            sparse = evals_metrics.astype(pd.SparseDtype("str", np.nan))
            sparse.to_pickle(COMPLETE)

        sparse = sparse.reset_index(drop=True)
        sparse['model_STREP$'] = sparse[['methodology_STREP$', 'paper_STREP$']].apply(lambda val: '{} from {}'.format(*val.astype(str)), axis=1, raw=True)
        meta_cols = [col for col in sparse.columns if '_STREP$' in col]
        # TODO rename columns?
        # rename = {col: col.lower().replace(' = ', '').replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '') for col in sparse_metrics.columns}
        # ren_map = {}
        # for col, ren_col in rename.items():
        #     if ren_col not in ren_map:
        #         ren_map[ren_col] = []
        #     ren_map[ren_col].append(col)    
        # sparse_metrics = sparse_metrics.rename(rename, axis=1)
        # ren_cols = []
        # for idx, (unified, orig) in enumerate(ren_map.items()):
        #     t0 = time.time()
        #     if len(orig) > 1:
        #         ren_cols.append( sparse_metrics[unified].bfill(axis=1).iloc[:, 0].astype(pd.SparseDtype("str", np.nan)) )
        #         print(f'Renamed metrics {unified:<30} {idx:<4} / {len(ren_map)} - took {time.time() - t0:.3f}s with {len(orig)} merged metrics')
        #     else:
        #         ren_cols.append( sparse_metrics[unified] )
        # sparse_metrics = pd.concat(ren_cols, axis=1)
        
        # process values to positive floats wherever possible
        sparse_metrics_stacked = sparse.drop(meta_cols, axis=1).stack()
        sparse_metrics_proc = sparse_metrics_stacked.apply(convert_to_float)
        sparse_metrics_proc = sparse_metrics_proc.map(lambda v: np.nan if isinstance(v, float) and v < 0 else v)
        # check for amount of successful conversions
        equal = sparse_metrics_stacked[sparse_metrics_stacked == sparse_metrics_proc]
        unequal = sparse_metrics_stacked[sparse_metrics_stacked != sparse_metrics_proc]
        print(f'Could not convert {equal.size / sparse_metrics_proc.size * 100:.2f}% of PWC metric values to float ({pd.unique(equal).size} / {pd.unique(unequal).size} in total).\nThe following metric values could not be transformed:', pd.unique(equal).to_dense())
        sparse_metrics_proc = sparse_metrics_proc.unstack()
        print(f'Sparsity level: only {unequal.size / sparse_metrics_proc.size * 100:.2f}% of the DB is populated ({unequal.size} available metric values in total)')
        # merge the processed metrics and meta info
        sparse = pd.concat([sparse[meta_cols], sparse_metrics_proc], axis=1)
        del equal
        del unequal
        del sparse_metrics_stacked
        del sparse_metrics_proc

        # rename the meta columns (and check for duplicates!)
        for col in meta_cols:
            without_tag = col.replace('_STREP$', '')
            if without_tag in sparse.columns:
                sparse = sparse.rename({without_tag: f'{without_tag}_PWC$'}, axis=1)
            sparse = sparse.rename({col: without_tag}, axis=1)
        meta_cols = [col.replace('_STREP$', '') for col in meta_cols]
        print(':::::::::::::::: PURGING UNINFORMATIVE EVALUATIONS FOR SPARSELY REPORTED DATASETS AND TASKS FROM FULL DB', sparse.shape)

        # check number of evaluated metrics for every DS X TASK combo, and remove all that do not fit the filter
        assert 'environment' not in sparse.columns
        MIN_EVALS, MIN_PROPS, MIN_POP, TOPK = 10, 3, 0.5, 100
        eval_data = {}
        for eval_meta, data in tqdm(sparse.groupby(['dataset', 'task'])):
            # delete if there only very few evals
            try:
                assert data.shape[0] >= MIN_EVALS
                data = data.dropna(how='all', axis=1)
            except Exception:
                continue
            # delete if there are not enough populated metrics
            metrics = []
            for col in data.columns:
                if col not in meta_cols:
                    try:
                        num_vals = pd.to_numeric(data[col], errors='coerce').dropna()
                        assert num_vals.size >= data.shape[0] * MIN_POP
                        metrics.append(col)
                    except Exception:
                        continue
            # otherwise, keep these rows and well-populated metrics
            if len(metrics) >= MIN_PROPS:
                eval_size, eval_rows, eval_cols = data[metrics].stack().size, data[metrics].dropna(how='all').index.to_series(), pd.Series(metrics)
                eval_data[eval_meta] = [eval_size, eval_rows, eval_cols]

        # filter for top 100 most densely populated evaluations
        eval_meta, eval_data = zip(*eval_data.items())
        eval_sizes, eval_rows, eval_cols = zip(*eval_data)
        top_evals = np.argsort(eval_sizes)[-TOPK:]
        keep_rows, keep_cols = [eval_rows[i] for i in top_evals], [eval_cols[i] for i in top_evals]
        keep_cols = [pd.Series(meta_cols)] + keep_cols
        keep_rows, keep_cols = pd.concat(keep_rows), pd.concat(keep_cols).drop_duplicates()
        sparse = sparse.loc[keep_rows,keep_cols].reset_index(drop=True)
        # drop all non-float values
        for col in keep_cols:
            if col not in meta_cols:
                numeric = pd.to_numeric(sparse[col], errors='coerce')
                sparse.loc[:,col] = numeric.astype(pd.SparseDtype(numeric.dtype, np.nan))
        print(f':::::::::::::::: FINISH DB SHAPE:', sparse.shape)
        sparse.to_pickle(FILTERED)

    # create properties json by checking the metrics in the DF, and whether they want to by minimized or maximized
    res_metrics = ['time', 'param', 'size', 'flops']
    client = PapersWithCodeClient()
    group_res = {'Resources': [], 'Performance': []}
    properties, possible_errors = {}, {}
    for task, task_data in tqdm(sparse.groupby('task')):
        page, metrics_to_check = 1, set(task_data.dropna(how='all', axis=1).select_dtypes('number').columns.to_list())
        while len(metrics_to_check) > 0:
            eval_list = client.task_evaluation_list(task, page=page)
            for eval in eval_list.results:
                if len(metrics_to_check) == 0:
                    break
                metrics = client.evaluation_metric_list(eval.id)
                for metr in metrics.results:
                    if metr.name in metrics_to_check:
                        unified = metr.name.lower().replace(' = ', '').replace(' ', '').replace('-', '').replace('_', '').replace('(', '').replace(')', '').replace('%', '').replace(',', '')
                        group = 'Resources' if any([key in unified for key in res_metrics]) else 'Performance'
                        group_res[group].append(metr.name)
                        if metr.name in properties:
                            if metr.is_loss != properties[metr.name]['maximize']:
                                if metr.name not in possible_errors:
                                    possible_errors[metr.name] = []
                                possible_errors[metr.name].append(metr.is_loss)
                        else:
                            properties[metr.name] =  {
                                "name": metr.name, "shortname": unified[:6], "unit": "number",
                                "group": group, "weight": 1, "maximize": not metr.is_loss,
                            }
                        metrics_to_check.remove(metr.name)
                        if len(metrics_to_check) == 0:
                            break
            if eval_list.next_page is not None:
                page += 1
        if len(metrics_to_check) > 0:
            raise RuntimeError('No properties found for', metrics_to_check)
    print('RESOURCE METRICS:\n', set(group_res['Resources']), '\n\n PERFORMANCE METRICS:\n', set(group_res['Performance']), '\n\nPOSSIBLE_ERRORS ACROSS TASKS\n', possible_errors.keys())
    for metr, is_loss in possible_errors.items():
        # take the is_loss information that occurs more frequently
        properties[metr]['maximize'] = bool(np.median(is_loss))
    with open(PROPS, 'w') as jf:
        json.dump(properties, jf)
    
    print('done', pd.__version__)
    # for gr, data in merged.groupby(['dataset']):
    #     data = data.dropna(how='all', axis=1)
    #     n_tasks, n_papers, m_methods = pd.unique(data['task']).size, pd.unique(data['paper']).size, pd.unique(data['methodology']).size
    #     print(f"{gr:<45} {n_tasks:<2} tasks {n_papers:<3} papers {m_methods:<3} methods {str(data.shape):<10} results")



############# FILTER FOR RELEVANT:
# print('    search relevant metrics')
#     all_metrics = {}
#     x_default, y_default = {}, {}
#     to_delete = []
#     properties_meta = identify_property_meta(meta, database)
#     for ds in pd.unique(database['dataset']):
#         subds = find_sub_db(database, ds)
#         for task in pd.unique(database[database['dataset'] == ds]['task']):
#             lookup = (ds, task)
#             subd = find_sub_db(subds, ds, task)
#             metrics = {}
#             for col, meta in properties_meta.items():
#                 if col in subd.columns or ('independent_of_task' in meta and meta['independent_of_task'] and col in subds.columns):
#                     val = properties_meta[col]
#                     metrics[col] = (val['weight'], val['group']) # weight is used for identifying the axis defaults
#             if len(metrics) < 2:
#                 to_delete.append(lookup)
#             else:
#                 # TODO later add this, such that it can be visualized
#                 # metrics['resource_index'] = {sum([weight for (weight, group) in metrics.values() if group != 'Performance']), 'Resource'}
#                 # metrics['quality_index'] = {sum([weight for (weight, group) in metrics.values() if group == 'Performance']), 'Performance'}
#                 # metrics['compound_index'] = {1.0, 'n.a.'}
#                 weights, groups = zip(*list(metrics.values()))

#                 argsort = np.argsort(weights)
#                 groups = np.array(groups)[argsort]
#                 metrics = np.array(list(metrics.keys()))[argsort]
#                 # use most influential Performance property on y-axis
#                 if 'Performance' not in groups:
#                     raise RuntimeError(f'Could not find quality property for {lookup}!')
#                 y_default[lookup] = metrics[groups == 'Performance'][-1]
#                 if 'Resources' in groups: # use the most influential resource property on x-axis
#                     x_default[lookup] = metrics[groups == 'Resources'][-1]
#                 elif 'Complexity' in groups: # use most influential complexity
#                     x_default[lookup] = metrics[groups == 'Complexity'][-1]
#                 else:
#                     try:
#                         x_default[lookup] = metrics[groups == 'Performance'][-2]
#                     except IndexError:
#                         print(f'No second Performance property and no Resources or Complexity properties were found for {lookup}!')
#                         to_delete.append(lookup)
#                 all_metrics[lookup] = metrics
#     drop_rows = []
#     for (ds, task) in to_delete:
#         print(f'Not enough numerical properties found for {task} on {ds}!')
#         try:
#             del(all_metrics[(ds, task)])
#             del(x_default[(ds, task)])
#             del(y_default[(ds, task)])
#         except KeyError:
#             pass
#         drop_rows.extend( find_sub_db(database, ds, task).index.to_list() )
#     database = database.drop(drop_rows)
#     database = database.reset_index(drop=True)
#     return database, all_metrics, x_default, y_default