import pandas as pd
import numpy as np
from tqdm import tqdm

from strep.util import weighted_median, load_meta

def _identify_property_meta(input, given_meta=None):
    properties_meta = {}
    if given_meta is not None and isinstance(given_meta, dict) and len(given_meta) > 0: # assess columns defined in meta
        cols_to_rate = [key for key in given_meta if (key in input.columns) and (input[key].dropna(how='all').size > 0)]
    else: # assess all numeric columns
        cols_to_rate = input.dropna(how='all', axis=1).select_dtypes('number').columns.to_list()
    if len(cols_to_rate) < 1:
        raise RuntimeError('No rateable properties found!')
    for col in cols_to_rate:
        try:
            meta = given_meta[col]
        except (TypeError, IndexError, KeyError):
            # TODO improve by looking up group and unit from properly characterized popular metrics in PWC and OpenML
            meta = { "name": col, "shortname": col[:4], "unit": "number", "group": "Quality", "weight": 1.0 / len(cols_to_rate) }
        properties_meta[col] = meta
    return properties_meta

def _value_to_index(value, ref, higher_better=True):
    res = value / ref if higher_better else ref / value
    res[np.isinf(value)] = 1 if higher_better else np.nan
    return res

def _index_to_value(index, ref, higher_better=True):
    res = index * ref  if higher_better else ref / index
    res[np.isnan(index)] = 0 if higher_better else np.inf
    return res

def _index_scale_reference(input, properties_meta, reference):
    raise NotImplementedError

def _index_scale_best(input, properties_meta, ___unused=None):
    results = {}
    for prop, meta in properties_meta.items():
        # values = input[prop].astype(input[prop].dtype.subtype) if isinstance(input[prop].dtype, pd.SparseDtype) else input[prop]
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

def _compound_single(input, mode, properties_meta):
    props = list(properties_meta.keys())
    weights = np.array([meta['weight'] for meta in properties_meta.values()])
    assert np.all(weights >= 0)
    weights_norm = weights / weights.sum()
    values = input[props].fillna(0)
    if mode == 'mean':
        return np.matmul(values, weights_norm)
    if mode == 'median':
        return np.array([weighted_median(row, weights_norm) for row in values.values])
        # TODO implement this more efficiently
    if mode == 'min':
        return values.min()
    if mode == 'max':
        return values.max()
    raise NotImplementedError(f'Mode {mode} not known for compound scoring!')

def _compound(input, properties_meta, boundaries, mode):
    index_vals = {}
    index_vals['compound_index'] = _compound_single(input, mode, properties_meta)
    index_vals['resource_index'] = _compound_single(input, mode, {p: v for p, v in properties_meta.items() if v['group'] != 'Quality'})
    index_vals['quality_index'] = _compound_single(input, mode, {p: v for p, v in properties_meta.items() if v['group'] == 'Quality'})
    index_vals = pd.DataFrame(index_vals, index=input.index)
    boundaries = _prepare_boundaries(index_vals, boundaries) # either provided from outside, or calculated based on quantiles
    rated, _ = _rate(index_vals, boundaries=boundaries)
    rated = rated.rename(columns={col: col.replace('index', 'rating') for col in index_vals.columns})
    return pd.concat([index_vals, rated], axis=1), boundaries

def _real_boundaries_and_defaults(input, boundaries, meta, reference=None):
    split_by = ['task', 'dataset', 'environment']
    real_bounds = {}
    defaults = { 'x': {}, 'y': {} }
    if len(split_by) == 0:
        raise NotImplementedError
    else:
        for sub_config, sub_data in input.groupby(split_by):
            assert sub_config in boundaries
            # calculate the real-valued bounds, based on the index-value bounds
            if sub_config not in real_bounds:
                real_bounds[sub_config] = {}
            sub_props = [prop for prop in boundaries[sub_config].keys() if prop not in ['compound_index', 'resource_index', 'quality_index']]
            for prop in sub_props:
                prop_bounds = boundaries[sub_config][prop]
                if reference is not None:
                    raise NotImplementedError
                elif 'maximize' in meta[prop] and meta[prop]['maximize']:
                    real_bounds[sub_config][prop] = _index_to_value(prop_bounds, sub_data[prop].max())
                else:
                    real_bounds[sub_config][prop] = _index_to_value(prop_bounds, sub_data[prop].min(), higher_better=False)
            # find best properties for default visualization
            task_ds = tuple([val for conf, val in zip(split_by, sub_config) if conf != 'environment'])
            if task_ds in defaults['x']: # not necessary for each individual data set!
                continue
            sub_props = list(reversed(sub_props)) # makes sure that equally weighted properties stay in right order (top down in json)
            weights, groups = zip(*[(meta[prop]['weight'], meta[prop]['group']) for prop in sub_props])            
            argsort = np.argsort(weights)
            groups = np.array(groups)[argsort]
            metrics = np.array(sub_props)[argsort]
            # use most influential Performance property on y-axis
            if 'Quality' not in groups:
                raise RuntimeError(f'Could not find quality property for {task_ds}!')
            defaults['y'][task_ds] = metrics[groups == 'Quality'][0]
            if 'Resources' in groups: # use the most influential resource property on x-axis
                defaults['x'][task_ds] = metrics[groups == 'Resources'][0]
            elif 'Complexity' in groups: # use most influential complexity
                defaults['x'][task_ds] = metrics[groups == 'Complexity'][0]
            else:
                try:
                    defaults['x'][task_ds] = metrics[groups == 'Quality'][1]
                except IndexError:
                    raise RuntimeError(f'No second Performance property and no Resources or Complexity properties were found for {task_ds}!')
    return real_bounds, defaults

def _prepare_boundaries(input, boundaries=None):
    assert (input.shape[0] > 0) and (input.shape[1] > 0)
    assert not np.any(np.logical_or(input > 1, input < 0)), 'Found values outside of the interval [0, 1] - please index-scale your results first and remove all unimportant columns!'    
    if isinstance(boundaries, np.ndarray):
        quantiles, boundaries = boundaries, {}
    else:
        quantiles = [0.8, 0.6, 0.4, 0.2]
        if boundaries is None:
            boundaries = {}
        if not isinstance(boundaries, dict):
            raise NotImplementedError('Please pass boundaries / reference as a dict (list of boundaries for each property), np.ndarray (quantiles) or None (for standard [0.8, 0.6, 0.4, 0.2] quantiles).')
    new_boundaries = {prop: boundaries[prop] if prop in boundaries else np.quantile(input[prop].dropna(), quantiles) for prop in input.columns}
    return new_boundaries

def _scale_single(input, scale_m, meta, reference, mode):
    sub_meta = _identify_property_meta(input, meta)
    reference_input = reference
    if mode == 'rating':
        assert not np.any(np.logical_or(input[sub_meta.keys()] > 1, input[sub_meta.keys()] < 0)), f'Found values outside of the interval (0, 1] - please properly index-scale your results first!'
        reference_input = _prepare_boundaries(input[sub_meta.keys()], reference) # reference now controls the rating boundaries per property
    if mode == 'compound':
        assert not np.any(np.logical_or(input[sub_meta.keys()] > 1, input[sub_meta.keys()] < 0)), f'Found values outside of the interval (0, 1] - please properly index-scale your results first!'
    results, boundaries = scale_m(input, sub_meta, reference_input)
    return results, boundaries, sub_meta

def _scaled_cols(input):
    return [col for col in input.columns if '_index' in col or '_rating' in col]

def load_database(fname):
    try:
        database = pd.read_pickle(fname)
    except Exception:
        raise RuntimeError(f'Could not load database "{fname}"\nPlease check the given file path and make sure to pass a pickled pandas dataframe!')
    meta = load_meta(fname)
    for field in ['environment', 'task', 'dataset']:
        if field not in database.columns:
            database[field] = 'unknown'
    return database, meta

def scale(input, meta=None, reference=None, mode='index', verbose=True):
    assert mode in ['rating', 'index', 'compound_mean', 'compound_median', 'compound_max', 'compound_min']
    scale_m = _index_scale_best if reference is None else _index_scale_reference # default is index scaling
    if isinstance(meta, dict) and 'properties' in meta: # other meta info irrelevant for scaling
        meta = meta['properties']
    if mode == 'index' and len(_scaled_cols(input)) > 0:
        raise RuntimeError('Found columns with "_scaling" or "_rating" information, which could result in runtime problems. Please rename these columns.')
    if mode == 'rating':
        scale_m = _rate
    if 'compound' in mode:
        scale_m = lambda inp, sm, ref: _compound(inp, sm, ref, mode.split('_')[1]) # pass compound mode [mean, median, min max] to compound function

    # process each individual combination of environment, task and dataset
    split_by = ['task', 'dataset', 'environment']
    # if mode != 'index':
    #     split_by.remove('environment') # environment only need to be considered for index scaling
    if verbose:
        print(f'Performing {mode} scaling for every separate combination of {str(split_by)}')
    sub_results = {}
    for sub_config, sub_input in tqdm(input.groupby(split_by), f'calculating {mode}'):
        sub_ref = reference
        if reference is not None and mode != 'index':
            try:
                sub_ref = reference[sub_config]
            except (ValueError, KeyError):
                raise RuntimeError(f'Could not properly process boundary information {reference} for {sub_config}')
        sub_results[sub_config] = _scale_single(sub_input, scale_m, meta, sub_ref, mode)
        if verbose:
            print('   - scaled results for', sub_config)
    # merge results results of each combination
    results, boundaries, sub_properties = zip(*sub_results.values())
    results = pd.concat(results).sort_index()
    all_props = {k: v for sub in sub_properties for k, v in sub.items()}
    final = pd.concat([input[input.columns.drop(all_props.keys())], results], axis=1)
    if mode == 'index': # no boundary information, instead return identified meta info
        return final, all_props
    boundaries = {config: bounds for config, bounds in zip(sub_results.keys(), boundaries)}
    # make some properties available across all tasks
    # if meta is not None and mode == 'index':
    #     independent_props = [prop for prop, vals in meta.items() if 'independent_of_task' in vals and vals['independent_of_task']]
    #     fixed_fields = ['model'] + list(set(['dataset', 'environment']).intersection(split_by))
    #     for group_field_vals, data in final.groupby(fixed_fields):
    #         for prop in independent_props:
    #             valid = data[prop].dropna()
    #             if valid.shape[0] != 1:
    #                 print(f'{valid.shape[0]} not-NA values found for {prop} across all tasks on {group_field_vals}!')
    #             if valid.shape[0] > 0:
    #                 final.loc[data.index,prop] = [valid.values[0]] * data.shape[0]
    return final, boundaries

def scale_and_rate(input, meta, reference=None, boundaries=None, compound_mode='mean', verbose=False):
    # make some properties available across all tasks if "independent_of_task" in meta
    if meta is not None and len(meta['properties']) >= 0:
        is_indep = lambda p, df: meta['properties'][p].get('independent_of_task') and df[p].dropna().size > 0
        for ds, ds_data in input.groupby('dataset'):
            fixed_fields = ['model', 'environment']
            independent_props = [prop for prop in meta['properties'].keys() if is_indep(prop, ds_data)]
            for group_field_vals, data in ds_data.groupby(fixed_fields):
                for prop in independent_props:
                    valid = data[prop].dropna()
                    if valid.shape[0] != 1:
                        print(f'{valid.shape[0]} not-NA values found for {prop} across all tasks on {ds} {group_field_vals}!')
                    if valid.shape[0] > 0:
                        input.loc[data.index,prop] = [valid.values[0]] * data.shape[0]
    # already processed db as input, so go back to original values
    if 'compound_rating' in input.columns:
        input = input.drop(columns=_scaled_cols(input))
    # compute index values
    scaled, meta_ret = scale(input, meta, reference=reference, verbose=verbose)
    if len(meta['properties']) == 0:
        meta['properties'] = meta_ret
    # compute ratings and compounds
    rated, prop_boundaries = scale(scaled, meta, reference=boundaries, mode='rating', verbose=verbose)
    compound, comp_boundaries = scale(scaled, meta, reference=boundaries, mode=f'compound_{compound_mode}', verbose=verbose)
    # concat all results
    config_cols = [col for col in compound.columns if col in input.columns]
    all_res = pd.concat([
            input,
            scaled.drop(config_cols, axis=1).rename(columns=lambda col: f'{col}_index'),
            rated.drop(config_cols, axis=1).rename(columns=lambda col: f'{col}_rating'),
            compound.drop(config_cols, axis=1)
        ], axis=1)
    # compute boundaries
    for config, boundaries in prop_boundaries.items():
        comp_boundaries[config].update(boundaries)
    real_boundaries, defaults = _real_boundaries_and_defaults(input, comp_boundaries, meta['properties'], reference)
    return all_res, meta, defaults, comp_boundaries, real_boundaries, reference
