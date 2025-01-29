import itertools

from strep.util import prop_dict_to_val

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

def calc_correlation(data, scale='index'):
    # init correlation table
    props = prop_dict_to_val(data, scale)
    if scale == 'index':
        props = props.rename(lambda col: col.replace('_index', ''), axis=1)
    corrs = np.full((props.shape[1], props.shape[1]), fill_value=np.nan)
    np.fill_diagonal(corrs, 1)
    
    # assess correlation between properties
    for idx_a, idx_b in itertools.combinations(np.arange(props.shape[1]), 2):
        cols = props.iloc[:, [idx_a, idx_b]].dropna().values
        if cols.size > 4:
            corrs[idx_a, idx_b] = pearsonr(cols[:,0], cols[:,1])[0]
            corrs[idx_b, idx_a] = corrs[idx_a, idx_b]
    return pd.DataFrame(corrs, index=props.columns, columns=props.columns)

def calc_all_correlations(db, scale='index'):
    corr = {}
    for lookup, data in db.groupby(['task', 'dataset', 'environment']):
        corr[lookup] = calc_correlation(data, scale)
    return corr