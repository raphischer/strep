import os
import json

import numpy as np
import pandas as pd

from strep.unit_reformatting import CustomUnitReformater
from strep.load_experiment_logs import find_sub_db
from strep.util import lookup_meta, drop_na_properties, prop_dict_to_val


def filter_properties(data, groups, inverse=False):
    if not isinstance(groups, list):
        groups = [groups]
    sel = [ prop for prop in data.index if isinstance(data[prop], dict) and ((data[prop]['group'] in groups) != inverse) ]
    if len(sel) == 0:
        raise RuntimeError('Filtering properties results in no properties!')
    return data[sel]


def score_performances(database, mode='optimistic mean', quantiles=None):
    if quantiles is None:
        quantiles = [0.8, 0.6, 0.4, 0.2]
    results = {
        'compound_index': [],
        'resource_index': [],
        'quality_index': []
    }
    for i, (_, log) in enumerate(database.iterrows()):
        results['compound_index'].append( calculate_single_compound_rating(log, mode) )
        try:
            results['quality_index'].append( calculate_single_compound_rating(filter_properties(log, 'Performance'), mode) )
        except RuntimeError:
            results['quality_index'].append(0)
        try:
            results['resource_index'].append( calculate_single_compound_rating(filter_properties(log, 'Performance', True), mode) )
        except RuntimeError:
            results['resource_index'].append(0)

    ratings = {}
    for key, values in results.items():
        rating_bounds = load_boundaries({'tmp': np.quantile(values, quantiles)})['tmp'] # TODO implement this nicer
        ratings[key.replace('index', 'rating')] = [index_to_rating(val, rating_bounds) for val in values]
    results.update(ratings)

    return results


def weighted_median(values, weights):
    assert np.isclose(weights.sum(), 1), "Weights for weighted median should sum up to one"
    cumw = np.cumsum(weights)
    for i, (cw, v) in enumerate(zip(cumw, values)):
        if cw == 0.5:
            return np.average([v, values[i + 1]])
        if cw > 0.5 or (cw < 0.5 and cumw[i + 1] > 0.5):
            return v
    raise RuntimeError


def calculate_single_compound_rating(input, mode='optimistic mean'):
    # extract lists of values
    if isinstance(input, pd.Series):
        input = input.to_dict()
    if isinstance(input, dict): # model summary given instead of list of ratings
        weights, index_vals = [], []
        for val in input.values():
            if isinstance(val, dict) and 'weight' in val and val['weight'] > 0:
                weights.append(val['weight'])
                index_vals.append(val['index'])
    elif isinstance(input, list):
        weights = [1] * len(input)
        index_vals = input
    else:
        raise NotImplementedError()
    if len(weights) == 0:
        return 0
    weights = [w / sum(weights) for w in weights] # normalize so that weights sum up to one
    asort = np.flip(np.argsort(index_vals))
    weights = np.array(weights)[asort]
    values = np.array(index_vals)[asort]
    if mode == 'best':
        return values[0]
    if mode == 'worst':
        return values[-1]
    if 'median' in mode:
        # TODO FIX weighted median rating / index error
        return weighted_median(values, weights)
    if 'mean' in mode:
        return np.average(values, weights=weights)
    raise NotImplementedError('Rating Mode not implemented!', mode)


def value_to_index(value, ref, higher_better):
    if np.isinf(value):
        return 1 if higher_better else 0
    if np.isnan(value):
        return 0
    #      i = v / r                     OR                i = r / v
    try:
        return value / ref if higher_better else ref / value
    except:
        return 0


def index_to_value(index, ref, higher_better):
    if index == 0:
        index = 10e-4
    #      v = i * r                            OR         v = r / i
    return index * ref  if higher_better else ref / index


def index_to_rating(index, scale):
    for i, (upper, lower) in enumerate(scale):
        if index <= upper and index > lower:
            return i
    return 4 # worst rating if index does not fall in boundaries


def process_property(value, reference_value, meta, unit_fmt):
    if isinstance(value, dict): # re-indexing
        returned_dict = value
    else:
        returned_dict = meta.copy()
        returned_dict['value'] = value
        if pd.isna(value):
            fmt_val, fmt_unit = 'N.A.', returned_dict['unit']
        else:
            fmt_val, fmt_unit = unit_fmt.reformat_value(value, returned_dict['unit'])
        returned_dict.update({'fmt_val': fmt_val, 'fmt_unit': fmt_unit})
    if 'weight' in returned_dict:
        higher_better = 'maximize' in returned_dict and returned_dict['maximize']
        val = returned_dict['value']
        if not higher_better and reference_value < 0: # shift both value to positive range for calculating the correct index
            val, reference_value = reference_value * -1 + 1 + val, 1
        returned_dict['index'] = value_to_index(val, reference_value, higher_better)
        if returned_dict['index'] < 0 or returned_dict['index'] > 1:
            print(f'Invalid index occured! {returned_dict["index"]} for value {val}, reference {reference_value}')
    return returned_dict


def find_optimal_reference(database, pre_rating_use_meta=None):
    model_names = database['model'].values
    metric_values = {}
    if pre_rating_use_meta is not None:
        metrics = [col for col in pre_rating_use_meta.keys() if col in database and any([not np.isnan(entry) for entry in database[col]])]
    else:
        metrics = [col for col in database.columns if any([isinstance(entry, dict) for entry in database[col]])]
    # aggregate index values for each metric
    for metric in metrics:
        if pre_rating_use_meta is not None:
            meta = pre_rating_use_meta[metric]
            higher_better = 'maximize' in meta and meta['maximize']
            weight = meta['weight']
            values = {model: val for _, (model, val) in database[['model', metric]].iterrows()}
        else:
            weight, values = 0, {}
            for idx, entry in enumerate(database[metric]):
                if isinstance(entry, dict):
                    higher_better = 'maximize' in entry and entry['maximize']
                    weight = max([entry['weight'], weight])
                    values[model_names[idx]] = entry['value']
        # assess the reference for each individual metric
        ref = np.median(list(values.values())) # TODO allow to change the rating mode
        values = {name: value_to_index(val, ref, higher_better) for name, val in values.items()}
        metric_values[metric] = values, weight
    # calculate model-specific scores based on metrix index values
    scores = {}
    for model in model_names:
        scores[model] = 0
        for values, weight in metric_values.values():
            if model in values:
                scores[model] += values[model] * weight
    # take the most average scored model
    ref_model_idx = np.argsort(list(scores.values()))[len(scores)//2]
    return model_names[ref_model_idx]


def calculate_optimal_boundaries(database, quantiles=None):
    if quantiles is None:
        quantiles = [0.8, 0.6, 0.4, 0.2]
    boundaries = {'default': [0.9, 0.8, 0.7, 0.6]}
    for col in database.columns:
        index_values = [ val['index'] for val in database[col] if isinstance(val, dict) and 'index' in val ]
        if len(index_values) > 0:
            try:
                boundaries[col] = np.quantile(index_values, quantiles)
            except Exception as e:
                print(e)
    return load_boundaries(boundaries)


def load_boundaries(content=None):
    if isinstance(content, dict):
        if isinstance(list(content.values())[0][0], list):
            # this is already the boundary dict with interval format
            return content
    elif content is None:
        content = {'default': [0.9, 0.8, 0.7, 0.6]}
    elif isinstance(content, str) and os.path.isfile(content):
        with open(content, "r") as file:
            content = json.load(file)
    else:
        raise RuntimeError('Invalid boundary input', content)
    
    boundary_intervals = {}
    for key, boundaries in content.items():
        boundary_intervals[key] = convert_boundaries_to_intervals(boundaries)
    return boundary_intervals


def convert_boundaries_to_intervals(boundaries, min_value=-100, max_value=100000):
    # Convert boundaries to dictionary
    intervals = [[max_value, boundaries[0]]]
    for i in range(len(boundaries)-1):
        intervals.append([boundaries[i], boundaries[i+1]])
    intervals.append([boundaries[-1], min_value])
    return intervals


def save_boundaries(boundary_intervals, output="boundaries.json"):
    scale = {}
    for key in boundary_intervals.keys():
        scale[key] = [sc[0] for sc in boundary_intervals[key][1:]]
    if output is not None:
        with open(output, 'w') as out:
            json.dump(scale, out, indent=4)
    
    return json.dumps(scale, indent=4)


def save_weights(database, output_fname=None):
    weights = {}
    for col in database.columns:
        any_result = database[col].dropna().iloc[0]
        if isinstance(any_result, dict) and 'weight' in any_result:
            weights[col] = any_result['weight']
    # directly save to file or return string
    if output_fname is not None:
        with open(output_fname, 'w') as out:
            json.dump(weights, out, indent=4)
    return json.dumps(weights, indent=4)


def update_weights(database, weights):
    update_db = False
    for key, weight in weights.items():
        axis_data_entries = database[key]
        for data in axis_data_entries:
            if isinstance(data, dict):
                if data['weight'] != weight:
                    update_db = True
                    data['weight'] = weight
    return update_db


def identify_property_meta(given_meta, database):
    properties_meta = {}
    if 'properties' in given_meta: # assess columns defined in meta
        cols_to_rate = [ key for key in given_meta['properties'] if key in database.select_dtypes('number').columns ]
    else: # assess all numeric columns
        cols_to_rate = database.select_dtypes('number').columns
    if len(cols_to_rate) < 1:
        raise RuntimeError('No rateable properties found!')
    for col in cols_to_rate:
        meta = lookup_meta(given_meta, col, '', 'properties')
        if not isinstance(meta, dict):
            meta = { "name": col, "shortname": col[:4], "unit": "number", "group": "Performance", "weight": 1.0 }
        properties_meta[col] = meta
    return properties_meta


def rate_database(database, given_meta, boundaries=None, indexmode='best', references=None, unit_fmt=None, rating_mode='optimistic mean'):
    assert pd.unique(database.index).size == database.shape[0], f"ERROR! Database shaped {database.shape} has only {pd.unique(database.index).size} unique indices"
    unit_fmt = unit_fmt or CustomUnitReformater()
    database = prop_dict_to_val(database) # necessary if already was processed
    properties_meta = identify_property_meta(given_meta, database)

    # group each dataset, task and environment combo
    fixed_fields = ['dataset', 'task', 'environment']

    # assess index values
    print('    assessing index values')
    for group_field_vals, data in database.groupby(fixed_fields):
        data = drop_na_properties(data)
        ds = group_field_vals[fixed_fields.index('dataset')]
        # process per metric
        for prop, meta in properties_meta.items():
            if prop in data.columns:
                higher_better = 'maximize' in meta and meta['maximize']
                # one central reference model receives index 1, everything else in relation
                if indexmode == 'centered':
                    if references is None:
                        references = {}
                    reference_name = references[ds] if ds in references else find_optimal_reference(data, properties_meta)
                    references[ds] = reference_name # if using optimal, store this info for later use
                    reference = data[data['model'] == reference_name]
                    assert reference.shape[0] == 1, f'Found multiple results for reference {reference_name} in {group_field_vals} results!'
                    ref_val = reference[prop].values[0]
                # the best perfoming model receives index 1, everything else in relation
                elif indexmode == 'best':
                    all_values = data[prop].dropna().values
                    if len(all_values) == 0:
                        ref_val = data[prop].iloc[0]
                    else:
                        ref_val = max(all_values) if higher_better else min(all_values)
                else:
                    raise RuntimeError(f'Invalid indexmode {indexmode}!')
                # extract meta, project on index values and rate
                database.loc[data.index,prop + '_dict'] = data.loc[:,prop].map( lambda value: process_property(value, ref_val, meta, unit_fmt) )
    # override the float values with dicts
    for prop in properties_meta.keys():
        if prop + '_dict' in database.columns:
            database.loc[:,prop] = database.loc[:,prop + '_dict']
            database = database.drop(labels=prop + '_dict', axis=1)

    print('    assessing ratings')
    # assess ratings & boundaries
    boundaries = calculate_optimal_boundaries(database) if boundaries is None else load_boundaries(boundaries)
    real_boundaries = {}
    for group_field_vals, data in database.groupby(fixed_fields):
        data = drop_na_properties(data)
        real_boundaries[group_field_vals] = {}
        # process per metric
        for prop, meta in properties_meta.items():
            if prop in data.columns:
                higher_better = 'maximize' in meta and meta['maximize']
                # extract rating boundaries per metric
                pr_bounds = boundaries[prop] if prop in boundaries else boundaries['default']
                boundaries[prop] = [bound.copy() for bound in pr_bounds] # copies not references! otherwise changing a boundary affects multiple metrics
                # calculate rating and real boundary values
                data[prop].map( lambda pr: pr.update({'rating': index_to_rating(pr['index'], pr_bounds)}) if isinstance(pr, dict) else pr )
                if indexmode == 'centered':
                    raise NotImplementedError() # but should be the value with index val 1
                else: # indexmode == 'best'
                    ref_val = sorted([(entry['index'], entry['value']) for entry in data[prop]])[-1][1]
                real_boundaries[group_field_vals][prop] = [(index_to_value(start, ref_val, higher_better), index_to_value(stop, ref_val, higher_better)) for (start, stop) in pr_bounds]
        database.loc[data.index,data.columns] = data
    
    # make certain model metrics available across all tasks
    for prop, meta in properties_meta.items():
        if 'independent_of_task' in meta and meta['independent_of_task']:
            fixed_fields = ['dataset', 'model']
            if pd.unique(database['environment']).size > 1:
                fixed_fields.append('environment')
            grouped_by = database.groupby(fixed_fields)
            for group_field_vals, data in grouped_by:
                data = drop_na_properties(data)
                if prop in data.columns:
                    valid = data.loc[prop_dict_to_val(data)[prop].dropna().index,prop]
                    # check if there even are nan rows in the database (otherwise metrics maybe have been made available already)
                    if valid.size != data[prop].size:
                        if valid.shape[0] != 1:
                            print(f'{valid.shape[0]} not-NA values found for {prop} across all tasks on {group_field_vals}!')
                        if valid.shape[0] > 0:
                            # replicate the available data and place in each row
                            database.loc[data.index,prop] = [valid.values[0]] * data.shape[0]
    
    print('    calculating compound ratings')
    # calculate compound index and ratings for each DS X TASK X ENVIRONMENT combo
    for group_field_vals, data in database.groupby(['dataset', 'task', 'environment']):
        # drop nan columns
        scoring = score_performances(drop_na_properties(data), rating_mode)
        for key, values in scoring.items():
            database.loc[data.index,key] = values
    for key in [col for col in database.columns if '_index' in col]:
        boundaries[key] = convert_boundaries_to_intervals(np.quantile(database[key], np.array([0.8, 0.6, 0.4, 0.2]))) # TODO improve the boundary handling
    database['compound_rating'] = database['compound_rating'].astype(int)
    database['resource_rating'] = database['resource_rating'].astype(int)
    database['quality_rating'] = database['quality_rating'].astype(int)
    return database, boundaries, real_boundaries, references


def find_relevant_metrics(database, meta):
    print('    search relevant metrics')
    all_metrics = {}
    x_default, y_default = {}, {}
    to_delete = []
    properties_meta = identify_property_meta(meta, database)
    for ds in pd.unique(database['dataset']):
        subds = find_sub_db(database, ds)
        for task in pd.unique(database[database['dataset'] == ds]['task']):
            lookup = (ds, task)
            subd = find_sub_db(subds, ds, task)
            metrics = {}
            for col, meta in properties_meta.items():
                if col in subd.columns or ('independent_of_task' in meta and meta['independent_of_task'] and col in subds.columns):
                    val = properties_meta[col]
                    metrics[col] = (val['weight'], val['group']) # weight is used for identifying the axis defaults
            if len(metrics) < 2:
                to_delete.append(lookup)
            else:
                # TODO later add this, such that it can be visualized
                # metrics['resource_index'] = {sum([weight for (weight, group) in metrics.values() if group != 'Performance']), 'Resource'}
                # metrics['quality_index'] = {sum([weight for (weight, group) in metrics.values() if group == 'Performance']), 'Performance'}
                # metrics['compound_index'] = {1.0, 'n.a.'}
                weights, groups = zip(*list(metrics.values()))

                argsort = np.argsort(weights)
                groups = np.array(groups)[argsort]
                metrics = np.array(list(metrics.keys()))[argsort]
                # use most influential Performance property on y-axis
                if 'Performance' not in groups:
                    raise RuntimeError(f'Could not find quality property for {lookup}!')
                y_default[lookup] = metrics[groups == 'Performance'][-1]
                if 'Resources' in groups: # use the most influential resource property on x-axis
                    x_default[lookup] = metrics[groups == 'Resources'][-1]
                elif 'Complexity' in groups: # use most influential complexity
                    x_default[lookup] = metrics[groups == 'Complexity'][-1]
                else:
                    try:
                        x_default[lookup] = metrics[groups == 'Performance'][-2]
                    except IndexError:
                        print(f'No second Performance property and no Resources or Complexity properties were found for {lookup}!')
                        to_delete.append(lookup)
                all_metrics[lookup] = metrics
    drop_rows = []
    for (ds, task) in to_delete:
        print(f'Not enough numerical properties found for {task} on {ds}!')
        try:
            del(all_metrics[(ds, task)])
            del(x_default[(ds, task)])
            del(y_default[(ds, task)])
        except KeyError:
            pass
        drop_rows.extend( find_sub_db(database, ds, task).index.to_list() )
    database = database.drop(drop_rows)
    database = database.reset_index(drop=True)
    return database, all_metrics, x_default, y_default


def load_database(fname):
    database = pd.read_pickle(fname)
    if hasattr(database, 'sparse'): # convert sparse databases to regular ones
        old_shape = database.shape
        database = database.sparse.to_dense()
        assert old_shape == database.shape
        for col in database.columns:
            try:
                fl = database[col].astype(float)
                database[col] = fl
            except Exception as e:
                pass
        database['environment'] = 'unknown'
    return database
