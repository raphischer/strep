import pandas as pd
import numpy as np

from strep.util import lookup_meta


def _identify_property_meta(input, given_meta=None):
    properties_meta = {}
    if given_meta is not None and isinstance(given_meta, dict) and 'properties' in given_meta: # assess columns defined in meta
        cols_to_rate = [key for key in given_meta['properties'] if (key in input.columns) and (input[key].dropna(how='all').size > 0)]
    else: # assess all numeric columns
        cols_to_rate = input.select_dtypes('number').dropna(how='all')
    if len(cols_to_rate) < 1:
        raise RuntimeError('No rateable properties found!')
    for col in cols_to_rate:
        meta = lookup_meta(given_meta, col, '', 'properties')
        if not isinstance(meta, dict):
            # TODO improve by looking up group from properly characterized popular metrics in PWC and OpenML
            meta = { "name": col, "shortname": col[:4], "unit": "number", "group": "Quality", "weight": 1.0 / len(cols_to_rate) }
        properties_meta[col] = meta
    return properties_meta

def _value_to_index(value, ref, higher_better=True):
    res = value / ref if higher_better else ref / value
    res[np.isinf(value)] = 1 if higher_better else np.nan
    res[np.isnan(value)] = np.nan
    return res

def _index_to_rating(index, scale):
    for i, (upper, lower) in enumerate(scale):
        if index <= upper and index > lower:
            return i
    return 4 # worst rating if index does not fall in boundaries

def _index_to_value(index, ref, higher_better=True):
    raise NotImplementedError
    assert ref > 0, f'Invalid reference value {ref} (must be larger than zero)'
    if isinstance(index, float) and index == 0:
        index = 10e-4
    if isinstance(index, pd.Series) and index[index==0].size > 0:
        index = index.copy()
        index[index == 0] = 10e-4
    #      v = i * r                            OR         v = r / i
    return index * ref  if higher_better else ref / index

def _index_scale_reference(input, properties_meta, reference):
    raise NotImplementedError

def _index_scale_best(input, properties_meta, ___unused=None):
    results = {}
    for prop, meta in properties_meta.items():
        assert not np.any(input[prop] < 0), f"Found negative values in {prop}, please scale to only positive values!"
        if 'maximize' in meta and meta['maximize']:
            results[prop] = _value_to_index(input[prop], input[prop].max())
        else:
            results[prop] = _value_to_index(input[prop], input[prop].min(), higher_better=False)
    # rating and compound scoring return boundaries, so return placeholder dict instead
    return pd.DataFrame(results), {prop: None for prop in results.keys()}

def _rate(input, ___unused=None, boundaries=None):
    assert(isinstance(boundaries, dict))
    results = {prop: np.digitize(input[prop], bins) for prop, bins in boundaries.items()}
    return pd.DataFrame(results, index=input.index), boundaries

def _compound_single(input, properties_meta):
    props = list(properties_meta.keys())
    weights = np.array([meta['weight'] for meta in properties_meta.values()])
    assert np.all(weights >= 0)
    weights_norm = weights / weights.sum()
    return np.matmul(input[props].fillna(0), weights_norm)

def _compound(input, properties_meta, boundaries):
    index_vals = {}
    index_vals['compound_index'] = _compound_single(input, properties_meta)
    index_vals['resource_index'] = _compound_single(input, {p: v for p, v in properties_meta.items() if v['group'] != 'Performance'})
    index_vals['quality_index'] = _compound_single(input, {p: v for p, v in properties_meta.items() if v['group'] == 'Performance'})
    index_vals = pd.DataFrame(index_vals, index=input.index)
    boundaries = prepare_boundaries(index_vals, boundaries) # either provided from outside, or calculated based on quantiles
    rated, _ = _rate(index_vals, boundaries=boundaries)
    rated = rated.rename(columns={col: col.replace('index', 'rating') for col in index_vals.columns})
    return pd.concat([index_vals, rated], axis=1), boundaries

def _reverse_index_scale(input):
    raise NotImplementedError

def _check_for_splitting(input):
    return [field for field in ['environment', 'task', 'dataset'] if field in input.columns]

def _prepare_for_scale(input):
    if isinstance(input, pd.DataFrame):
        pass
    else:
        raise NotImplementedError('Please pass a pandas dataframe as input!')
    split_by = _check_for_splitting(input)
    return input, split_by

def prepare_boundaries(input, boundaries=None):
    assert (input.shape[0] > 0) and (input.shape[1] > 0)
    assert not np.any(np.logical_or(input > 1, input < 0)), 'Found values outside of the interval [0, 1] - please index-scale your results first and remove all unimportant columns!'
    if isinstance(boundaries, dict):
        assert sorted(list(boundaries.keys())) == sorted(input.columns)
        return boundaries
    elif isinstance(boundaries, np.ndarray):
        quantiles = boundaries
    elif boundaries is None:
        quantiles = [0.8, 0.6, 0.4, 0.2]
    else:
        raise NotImplementedError('Please pass boundaries / reference as a dict (list of boundaries for each property), np.ndarray (quantiles) or None (for standard [0.8, 0.6, 0.4, 0.2] quantiles).')
    boundaries = {prop: np.quantile(input[prop], quantiles) for prop in input.columns}
    return boundaries

def _scale_single(input, scale_m, meta, reference, mode):
    sub_meta = _identify_property_meta(input, meta)
    reference_input = reference
    if mode == 'rating':
        assert not np.any(np.logical_or(input[sub_meta.keys()] > 1, input[sub_meta.keys()] < 0)), f'Found values outside of the interval (0, 1] - please properly index-scale your results first!'
        reference_input = prepare_boundaries(input[sub_meta.keys()], reference) # reference now controls the rating boundaries per property
    if mode == 'compound':
        assert not np.any(np.logical_or(input[sub_meta.keys()] > 1, input[sub_meta.keys()] < 0)), f'Found values outside of the interval (0, 1] - please properly index-scale your results first!'
    results, boundaries = scale_m(input, sub_meta, reference_input)
    return results, boundaries, sub_meta.keys()

def scale(input, meta=None, reference=None, mode='index', verbose=True):
    input, split_by = _prepare_for_scale(input)
    assert mode in ['rating', 'index', 'compound']
    mode_str = 'relative index scaling'
    scale_m = _index_scale_best if reference is None else _index_scale_reference
    if mode == 'rating':
        scale_m, mode_str = _rate, 'discrete rating'
    if mode == 'compound':
        scale_m, mode_str = _compound, 'compound scoring'

    if len(split_by) > 0:
        if verbose:
            print(f'Performing {mode_str} for every separate combination of {str(split_by)}')
        sub_results = {}
        for sub_config, sub_input in input.groupby(split_by): 
            sub_results[sub_config] = _scale_single(sub_input, scale_m, meta, reference, mode)
            if verbose:
                print('   - scaled results for', sub_config)
        # unzip and merge results
        results, boundaries, sub_properties = zip(*sub_results.values())
        results = pd.concat(results).sort_index()
        boundaries = {config: bounds for config, bounds in zip(sub_results.keys(), boundaries)}
        all_props = list(set([col for sub_cols in sub_properties for col in sub_cols]))
    else:
        if verbose:
            print(f'Performing {mode_str} for the complete data frame, without any splitting. If you want internal splitting, please provide information on respective "environment", "task" or "dataset".')
        results, boundaries, all_props = _scale_single(input, scale_m, meta, reference, mode)
    final = pd.concat([input[input.columns.drop(all_props)], results], axis=1)
    # make some properties available across all tasks
    if 'task' in split_by and meta is not None and mode == 'index':
        independent_props = [prop for prop, vals in meta['properties'].items() if 'independent_of_task' in vals and vals['independent_of_task']]
        fixed_fields = ['model'] + list(set(['dataset', 'environment']).intersection(split_by))
        for group_field_vals, data in final.groupby(fixed_fields):
            for prop in independent_props:
                valid = data[prop].dropna()
                if valid.shape[0] != 1:
                    print(f'{valid.shape[0]} not-NA values found for {prop} across all tasks on {group_field_vals}!')
                if valid.shape[0] > 0:
                    final.loc[data.index,prop] = [valid.values[0]] * data.shape[0]
    if mode in ['rating', 'compound']:
        return final, boundaries
    return final

def scale_and_rate(input, meta, reference=None, boundaries=None, verbose=False):
    scaled = scale(input, meta, reference=reference, verbose=verbose)
    rated, prop_boundaries = scale(scaled, meta, reference=boundaries, mode='rating', verbose=verbose)
    compound, comp_boundaries = scale(scaled, meta, reference=boundaries, mode='compound', verbose=verbose)
    config_cols = [col for col in compound.columns if col in input.columns]
    all_res = [
        input,
        scaled.drop(config_cols, axis=1).rename(columns=lambda col: f'{col}_index'),
        rated.drop(config_cols, axis=1).rename(columns=lambda col: f'{col}_rating'),
        compound.drop(config_cols, axis=1)
    ]
    for config, boundaries in prop_boundaries.items():
        comp_boundaries[config].update(boundaries)
    return pd.concat(all_res, axis=1), comp_boundaries
