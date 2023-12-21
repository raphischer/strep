import base64
import json

import numpy as np
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
from dash import dcc
import dash_bootstrap_components as dbc

from strep.index_and_rate import rate_database, load_boundaries, save_boundaries, calculate_optimal_boundaries, save_weights, find_optimal_reference, update_weights
from strep.elex.pages import create_page
from strep.elex.util import summary_to_html_tables, toggle_element_visibility, fill_meta
from strep.elex.graphs import assemble_scatter_data, create_scatter_graph, create_bar_graph, add_rating_background, create_star_plot
from strep.labels.label_generation import PropertyLabel
from strep.unit_reformatting import CustomUnitReformater
from strep.load_experiment_logs import find_sub_db
from strep.util import lookup_meta, PatchedJSONEncoder


class Visualization(dash.Dash):

    def __init__(self, databases, index_mode='best', dark_mode=True, **kwargs):
        self.dark_mode = dark_mode
        if dark_mode:
            kwargs['external_stylesheets'] = [dbc.themes.DARKLY]
        super().__init__(__name__, **kwargs)

        self.databases = databases
        self.unit_fmt = CustomUnitReformater()

        self.state = {
            'db': None,
            'ds': None,
            'task': None,
            'sub_database': None,
            'indexmode': index_mode,
            'update_on_change': False,
            'rating_mode': 'optimistic mean',
            'model': None,
            'label': None
        }
        
        # setup page and create callbacks
        self.layout = create_page(self.databases, index_mode, self.state['rating_mode'])
        self.callback(
            [Output('x-weight', 'value'), Output('y-weight', 'value')],
            [Input('xaxis', 'value'), Input('yaxis', 'value'), Input('weights-upload', 'contents')]
        ) (self.update_metric_fields)
        # changing database, dataset or task
        self.callback(
            [Output('ds-switch', 'options'), Output('ds-switch', 'value')],
            Input('db-switch', 'value')
        ) (self.db_selected)
        self.callback(
            [Output('task-switch', 'options'), Output('task-switch', 'value')],
            Input('ds-switch', 'value')
        ) (self.ds_selected)
        self.callback(
            [Output('environments', 'options'), Output('environments', 'value'), Output('xaxis', 'options'), Output('xaxis', 'value'), Output('yaxis', 'options'),  Output('yaxis', 'value'), Output('select-reference', 'options'), Output('select-reference', 'value')],
            [Input('task-switch', 'value'), Input('btn-optimize-reference', 'n_clicks')]
        ) (self.task_selected)
        # updating boundaries and graphs
        self.callback(
            [Output(sl_id, prop) for sl_id in ['boundary-slider-x', 'boundary-slider-y'] for prop in ['min', 'max', 'value', 'marks']],
            [Input('xaxis', 'value'), Input('yaxis', 'value'), Input('boundaries-upload', 'contents'), Input('btn-calc-boundaries', 'n_clicks'), Input('select-reference', 'value')]
        ) (self.update_boundary_sliders)
        self.callback(
            [Output('graph-scatter', 'figure'), Output('select-reference', 'disabled'), Output('btn-optimize-reference', 'disabled')],
            [Input('environments', 'value'), Input('scale-switch', 'value'), Input('indexmode-switch', 'value'), Input('rating', 'value'), Input('x-weight', 'value'), Input('y-weight', 'value'), Input('boundary-slider-x', 'value'), Input('boundary-slider-y', 'value')]
        ) (self.update_scatter_graph)
        self.callback(
            Output('graph-bars', 'figure'),
            Input('graph-scatter', 'figure')
        ) (self.update_bars_graph)
        self.callback(
            [Output('model-table', 'children'), Output('metric-table', 'children'), Output('model-label', "src"), Output('label-modal-img', "src"), Output('btn-open-paper', "href"), Output('info-hover', 'is_open')],
            Input('graph-scatter', 'hoverData'), State('environments', 'value'), State('rating', 'value')
        ) (self.display_model)
        # buttons for saving and loading
        self.callback(Output('save-label', 'data'), [Input('btn-save-label', 'n_clicks'), Input('btn-save-label2', 'n_clicks'), Input('btn-save-summary', 'n_clicks'), Input('btn-save-logs', 'n_clicks')]) (self.save_label)
        self.callback(Output('save-boundaries', 'data'), Input('btn-save-boundaries', 'n_clicks')) (self.save_boundaries)
        self.callback(Output('save-weights', 'data'), Input('btn-save-weights', 'n_clicks')) (self.save_weights)
        # offcanvas and modals
        self.callback(Output("exp-config", "is_open"), Input("btn-open-exp-config", "n_clicks"), State("exp-config", "is_open")) (toggle_element_visibility)
        self.callback(Output("graph-config", "is_open"), Input("btn-open-graph-config", "n_clicks"), State("graph-config", "is_open")) (toggle_element_visibility)
        self.callback(Output('label-modal', 'is_open'), Input('model-label', "n_clicks"), State('label-modal', 'is_open')) (toggle_element_visibility)


    def update_scatter_graph(self, env_names=None, scale_switch=None, indexmode_switch=None, rating_mode=None, xweight=None, yweight=None, *slider_args):
        update_db = False
        triggered_prop = self.triggered_graph_prop or dash.callback_context.triggered[0]['prop_id']
        self.triggered_graph_prop = None
        # most input triggers affect one of the two axis
        if 'x' in triggered_prop:
            axis, slider_values, weight = self.state['xaxis'],  slider_args[0],  xweight
        else:
            axis, slider_values, weight = self.state['yaxis'],  slider_args[1],  yweight
        # check if axis weight were updated
        if any([xweight, yweight]) and 'weight' in triggered_prop:
            update_db = update_db or update_weights(self.database, {axis: weight})
        # check if sliders were updated
        if any(slider_args) and 'slider' in triggered_prop:
            for sl_idx, sl_val in enumerate(slider_values):
                self.boundaries[axis][4 - sl_idx][0] = sl_val
                self.boundaries[axis][3 - sl_idx][1] = sl_val
            update_db = True
         # check if rating mode was changed
        if rating_mode != self.state['rating_mode']:
            self.state['rating_mode'] = rating_mode
            update_db = True
        # check if indexmode was changed
        if indexmode_switch != self.state['indexmode']:
            self.state['indexmode'] = indexmode_switch
            update_db = True
        reference_select_disabled = self.state['indexmode'] != 'centered'
        # update database if necessary
        if update_db:
            self.update_database(only_current=False)
        # assemble data for plotting
        scale = scale_switch or 'index'
        db, xaxis, yaxis = self.state['sub_database'], self.state['xaxis'], self.state['yaxis']
        bounds = self.boundaries if scale == 'index' else self.boundaries_real[(self.state['ds'], self.state['ds_task'][1], env_names[0])]
        self.plot_data, axis_names, rating_pos = assemble_scatter_data(env_names, db, scale, xaxis, yaxis, self.meta, bounds)
        scatter = create_scatter_graph(self.plot_data, axis_names, dark_mode=self.dark_mode)
        add_rating_background(scatter, rating_pos, self.state['rating_mode'], dark_mode=self.dark_mode)
        return scatter, reference_select_disabled, reference_select_disabled
    
    def update_database(self, only_current=True):
        if not only_current: # remark for making a full update when task / data set is changed
            self.state['update_on_change'] = True
        # update the data currently displayed to user
        self.state['sub_database'], self.boundaries, self.boundaries_real, self.references = rate_database(self.state['sub_database'], self.meta, self.boundaries, self.state['indexmode'], self.references, self.unit_fmt, self.state['rating_mode'])
        self.database.loc[self.state['sub_database'].index] = self.state['sub_database']

    def update_bars_graph(self, scatter_graph=None, discard_y_axis=False):
        bars = create_bar_graph(self.plot_data, self.dark_mode, discard_y_axis)
        return bars

    def update_boundary_sliders(self, xaxis=None, yaxis=None, uploaded_boundaries=None, calculated_boundaries=None, reference=None):
        self.triggered_graph_prop = dash.callback_context.triggered[0]['prop_id']
        if uploaded_boundaries is not None:
            boundaries_dict = json.loads(base64.b64decode(uploaded_boundaries.split(',')[-1]))
            self.boundaries = load_boundaries(boundaries_dict)
            self.update_database(only_current=False)
        if calculated_boundaries is not None and 'calc' in dash.callback_context.triggered[0]['prop_id']:
            if self.state['update_on_change']: # if the indexmode was changed, it is first necessary to update all index values
                self.database, self.boundaries, self.boundaries_real, self.references = rate_database(self.database, self.meta, self.boundaries, self.state['indexmode'], self.references, self.unit_fmt, self.state['rating_mode'])
                self.state['update_on_change'] = False
            self.boundaries = calculate_optimal_boundaries(self.database, [0.8, 0.6, 0.4, 0.2])
        if self.references is not None and reference != self.references[self.state['ds']]:
            # reference changed, so re-index the current sub database
            self.references[self.state['ds']] = reference
            self.update_database()
        self.state['xaxis'] = xaxis or self.state['xaxis']
        self.state['yaxis'] = yaxis or self.state['yaxis']
        values = []
        for axis in [self.state['xaxis'], self.state['yaxis']]:
            all_ratings = [metric['index'] for metric in self.state['sub_database'][axis].dropna()]
            min_v = min(all_ratings)
            max_v = max(all_ratings)
            value = [entry[0] for entry in reversed(self.boundaries[axis][1:])]
            marks = { val: {'label': str(val)} for val in np.round(np.linspace(min_v, max_v, 20), 3)}
            values.extend([min_v, max_v, value, marks])
        return values
    
    def db_selected(self, db=None):
        self.state['db'] = db or self.state['db']
        self.database, self.meta, self.metrics, self.xaxis_default, self.yaxis_default, self.boundaries, self.boundaries_real, self.references = self.databases[self.state['db']]
        options = [ {'label': lookup_meta(self.meta, ds, subdict='dataset'), 'value': ds} for ds in pd.unique(self.database['dataset']) ]
        return options, options[0]['value']

    def ds_selected(self, ds=None):
        self.state['ds'] = ds or self.state['ds']
        tasks = [ {"label": task.capitalize(), "value": task} for task in pd.unique(find_sub_db(self.database, self.state['ds'])['task']) ]
        return tasks, tasks[0]['value']

    def task_selected(self, task=None, find_optimal_ref=None):
        if find_optimal_ref is not None:
            self.references[self.state['ds']] = find_optimal_reference(self.state['sub_database'])
            self.update_database()
        if self.state['update_on_change']:
            self.database, self.boundaries, self.boundaries_real, self.references = rate_database(self.database, self.meta, self.boundaries, self.state['indexmode'], self.references, self.unit_fmt, self.state['rating_mode'])
            self.state['update_on_change'] = False
        self.state['ds_task'] = (self.state['ds'], task) or (self.state['ds'], self.state['task'])

        avail_envs = [ {"label": env, "value": env} for env in pd.unique(find_sub_db(self.database, self.state['ds'], self.state['ds_task'][1])['environment']) ]
        axis_options = [{'label': lookup_meta(self.meta, metr, subdict='properties'), 'value': metr} for metr in self.metrics[self.state['ds_task']]]
        self.state['xaxis'] = self.xaxis_default[self.state['ds_task']]
        self.state['yaxis'] = self.yaxis_default[self.state['ds_task']]
        self.state['sub_database'] = find_sub_db(self.database, self.state['ds'], self.state['ds_task'][1])
        models = self.state['sub_database']['model'].values
        ref_options = [{'label': mod, 'value': mod} for mod in models]
        curr_ref = self.references[self.state['ds']] if self.references is not None and self.state['ds'] in self.references else models[0]
        return avail_envs, [avail_envs[0]['value']], axis_options, self.state['xaxis'], axis_options, self.state['yaxis'], ref_options, curr_ref

    def display_model(self, hover_data=None, env_names=None, rating_mode=None):
        if hover_data is None:
            self.state['model'] = None
            self.state['label'] = None
            model_table, metric_table,  enc_label, link, open = None, None, None, "/", True
        else:
            point = hover_data['points'][0]
            env_name = env_names[point['curveNumber']] if len(env_names) < 2 else env_names[point['curveNumber'] - 1]
            model = find_sub_db(self.state['sub_database'], environment=env_name).iloc[point['pointNumber']].to_dict()
            self.state['model'] = fill_meta(model, self.meta)
            if isinstance(self.state['model']['model'], str): 
                # make sure that model is always a dict with name field
                self.state['model']['model'] = {'name': self.state['model']['model']}
            self.state['label'] = PropertyLabel(self.state['model'])
            model_table, metric_table = summary_to_html_tables(self.state['model'], self.metrics[self.state['ds_task']])
            starplot = create_star_plot(self.state['model'], self.metrics[self.state['ds_task']])
            enc_label = self.state['label'].to_encoded_image()
            try:
                link = self.state['model']['model']['url']
            except (IndexError, KeyError, TypeError):
                link = 'https://github.com/raphischer/strep'
            open = False
        return model_table, metric_table,  enc_label, enc_label, link, open

    def save_boundaries(self, save_labels_clicks=None):
        if save_labels_clicks is not None:
            return dict(content=save_boundaries(self.boundaries, None), filename='boundaries.json')

    def update_metric_fields(self, xaxis=None, yaxis=None, upload=None):
        if upload is not None:
            weights = json.loads(base64.b64decode(upload.split(',')[-1]))
            update_db = update_weights(self.database, weights)
            if update_db:
                self.update_database(only_current=False)
        self.state['xaxis'] = xaxis or self.state['xaxis']
        self.state['yaxis'] = yaxis or self.state['yaxis']
        for _, row in self.state['sub_database'].iterrows():
            try:
                return row[self.state['xaxis']]['weight'], row[self.state['yaxis']]['weight']
            except TypeError:
                pass
        raise RuntimeError('No weights found for props ')

    def save_weights(self, save_weights_clicks=None):
        if save_weights_clicks is not None:
            return dict(content=save_weights(self.database), filename='weights.json')

    def save_label(self, lbl_clicks=None, lbl_clicks2=None, sum_clicks=None, log_clicks=None):
        if (lbl_clicks is None and lbl_clicks2 is None and sum_clicks is None and log_clicks is None) or self.state['model'] is None:
            return # callback init
        f_id = f'{self.state["model"]["model"]["name"]}_{self.state["model"]["environment"]}'.replace(' ', '_')
        if 'label' in dash.callback_context.triggered[0]['prop_id']:
            return dcc.send_bytes(self.state['label'].write(), filename=f'energy_label_{f_id}.pdf')
        elif 'sum' in dash.callback_context.triggered[0]['prop_id']:
            return dict(content=json.dumps(self.state['model'], indent=4, cls=PatchedJSONEncoder), filename=f'energy_summary_{f_id}.json')
        else: # full logs
            # TODO load logs
            raise NotImplementedError
