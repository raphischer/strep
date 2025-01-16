import base64
import json

import numpy as np
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from strep.elex.pages import create_page
from strep.elex.util import summary_to_html_tables, toggle_element_visibility
from strep.elex.graphs import assemble_scatter_data, create_scatter_graph, create_bar_graph, add_rating_background, create_star_plot
from strep.labels.label_generation import PropertyLabel
from strep.index_scale import scale_and_rate, _extract_weights
from strep.unit_reformatting import CustomUnitReformater
from strep.util import lookup_meta, PatchedJSONEncoder, fill_meta, find_sub_db


class Visualization(dash.Dash):

    def __init__(self, databases, index_mode='best', dark_mode=True, **kwargs):
        self.dark_mode = dark_mode
        if dark_mode:
            kwargs['external_stylesheets'] = [dbc.themes.DARKLY]
        super().__init__(__name__, use_pages=True, pages_folder='', **kwargs)
        if not isinstance(databases, dict):
            databases = {'CUSTOM': databases}
        self.databases = databases
        self.unit_fmt = CustomUnitReformater()
        self.state = {
            'db': None,
            'ds': None,
            'task': None,
            'sub_database': None,
            'indexmode': index_mode,
            'update_on_change': False,
            'compound_mode': 'mean',
            'model': None,
            'label': None
        }
        
        # setup page and create callbacks
        page_creator = lambda **kwargs: create_page(self.databases, index_mode, self.state['compound_mode'], **kwargs)
        dash.register_page("home", layout=page_creator, path="/")
        self.layout = dash.html.Div(dash.page_container)
        self.callback(
            [Output('x-weight', 'value'), Output('y-weight', 'value')],
            [Input('xaxis', 'value'), Input('yaxis', 'value'), Input('weights-upload', 'contents')]
        ) (self.update_metric_fields)
        # changing database, dataset or task
        self.callback(
            [Output('ds-switch', 'options'), Output('ds-switch', 'value'), Output("url", "search"),],
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
            [Input('environments', 'value'), Input('scale-switch', 'value'), Input('indexmode-switch', 'value'), Input('compound_mode', 'value'), Input('x-weight', 'value'), Input('y-weight', 'value'), Input('boundary-slider-x', 'value'), Input('boundary-slider-y', 'value')]
        ) (self.update_scatter_graph)
        self.callback(
            Output('graph-bars', 'figure'),
            Input('graph-scatter', 'figure')
        ) (self.update_bars_graph)
        self.callback(
            [Output('model-table', 'children'), Output('metric-table', 'children'), Output('model-label', "src"), Output('label-modal-img', "src"), Output('btn-open-paper', "href"), Output('info-hover', 'is_open')],
            Input('graph-scatter', 'hoverData'), State('environments', 'value'), State('compound_mode', 'value')
        ) (self.display_model)
        # buttons for saving and loading
        self.callback(Output('save-label', 'data'), [Input('btn-save-label', 'n_clicks'), Input('btn-save-label2', 'n_clicks'), Input('btn-save-summary', 'n_clicks'), Input('btn-save-logs', 'n_clicks')]) (self.save_label)
        self.callback(Output('save-boundaries', 'data'), Input('btn-save-boundaries', 'n_clicks')) (self.save_boundaries)
        self.callback(Output('save-weights', 'data'), Input('btn-save-weights', 'n_clicks')) (self.save_weights)
        # offcanvas and modals
        self.callback(Output("exp-config", "is_open"), Input("btn-open-exp-config", "n_clicks"), State("exp-config", "is_open")) (toggle_element_visibility)
        self.callback(Output("graph-config", "is_open"), Input("btn-open-graph-config", "n_clicks"), State("graph-config", "is_open")) (toggle_element_visibility)
        self.callback(Output('label-modal', 'is_open'), Input('model-label', "n_clicks"), State('label-modal', 'is_open')) (toggle_element_visibility)

    def update_scatter_graph(self, env_names=None, scale_switch=None, indexmode_switch=None, compound_mode=None, xweight=None, yweight=None, *slider_args):
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
        if compound_mode != self.state['compound_mode']:
            self.state['compound_mode'] = compound_mode
            for bounds in self.boundaries.values(): # remove old boundaries, they have to be renewed after the update
                for key in ['compound_index', 'quality_index', 'resource_index']:
                    del bounds[key]
            update_db = True
        # check if indexmode was changed
        if indexmode_switch != self.state['indexmode']:
            self.state['indexmode'] = indexmode_switch
            update_db = True
        reference_select_disabled = self.state['indexmode'] != 'centered'
        # update database if necessary
        if update_db:
            self.update_database()
        # assemble data for plotting and create scatter figure
        scale = scale_switch or 'index'
        db, xaxis, yaxis = self.state['sub_database'], self.state['xaxis'], self.state['yaxis']
        self.plot_data, axis_names = assemble_scatter_data(env_names, db, scale, xaxis, yaxis, self.meta, self.unit_fmt)
        scatter = create_scatter_graph(self.plot_data, axis_names, dark_mode=self.dark_mode)
        # identify boundaries for rating background
        bounds = self.boundaries if scale == 'index' else self.boundaries_real
        bounds = bounds[(self.state['task'], self.state['ds'], env_names[0])]
        bounds = [bounds[xaxis].tolist(), bounds[yaxis].tolist()]
        add_rating_background(scatter, bounds, use_grad=len(env_names)>1, mode=self.state['compound_mode'], dark_mode=self.dark_mode)
        return scatter, reference_select_disabled, reference_select_disabled
    
    def update_database(self):
        # re-scale everything!
        self.database, self.boundaries, self.real_boundaries, self.defaults = scale_and_rate(self.database, self.meta['properties'], self.references, self.boundaries, self.state['compound_mode'], False)
        self.state['sub_database'] = self.database.loc[self.state['sub_database'].index]

    def update_bars_graph(self, scatter_graph=None, discard_y_axis=False):
        bars = create_bar_graph(self.plot_data, self.dark_mode, discard_y_axis)
        return bars

    def update_boundary_sliders(self, xaxis=None, yaxis=None, uploaded_boundaries=None, calculated_boundaries=None, reference=None):
        self.triggered_graph_prop = dash.callback_context.triggered[0]['prop_id']
        if uploaded_boundaries is not None:
            boundaries_dict = json.loads(base64.b64decode(uploaded_boundaries.split(',')[-1]))
            raise NotImplementedError
            self.update_database()
        if calculated_boundaries is not None and 'calc' in dash.callback_context.triggered[0]['prop_id']:
            if self.state['update_on_change']: # if the indexmode was changed, it is first necessary to update all index values
                self.database, self.boundaries, self.boundaries_real, self.references = rate_database(self.database, self.meta, self.boundaries, self.state['indexmode'], self.references, self.unit_fmt, self.state['compound_mode'])
                self.state['update_on_change'] = False
            self.boundaries = calculate_optimal_boundaries(self.database, [0.8, 0.6, 0.4, 0.2])
        if self.references is not None and reference != self.references[self.state['ds']]:
            # reference changed, so re-index the current sub database
            self.references[self.state['ds']] = reference
            self.update_database()
        self.state['xaxis'] = xaxis or self.state['xaxis']
        self.state['yaxis'] = yaxis or self.state['yaxis']
        # TODO implement again
        # values = []
        # for axis in [self.state['xaxis'], self.state['yaxis']]:
        #     all_vals = self.state['sub_database'][f'{axis}_index'].dropna()
        #     min_v, max_v = all_vals.min(), all_vals.max()
        #     marks = { val: {'label': str(val)} for val in np.round(np.linspace(min_v, max_v, 20), 3)}
        #     values.extend([min_v, max_v, self.boundaries[self.state['task_ds']][axis].tolist(), marks])
        return [0, 1, np.linspace(0, 1, 10), np.round(np.linspace(0, 1, 10), 3)] * 2
    
    def db_selected(self, db=None):
        self.state['db'] = db or self.state['db']
        self.database, self.meta, self.defaults, self.boundaries, self.boundaries_real, self.references = self.databases[self.state['db']]
        options = [ {'label': lookup_meta(self.meta, ds, subdict='dataset'), 'value': ds} for ds in pd.unique(self.database['dataset']) ]
        return options, options[0]['value'], f"?database={self.state['db']}"

    def ds_selected(self, ds=None):
        self.state['ds'] = ds or self.state['ds']
        tasks = [ {"label": task.capitalize(), "value": task} for task in pd.unique(find_sub_db(self.database, self.state['ds'])['task']) ]
        return tasks, tasks[0]['value']

    def task_selected(self, task=None, find_optimal_ref=None):
        if find_optimal_ref is not None:
            self.references[self.state['ds']] = find_optimal_reference(self.state['sub_database'])
            self.update_database()
        if self.state['update_on_change']:
            self.database, self.boundaries, self.boundaries_real, self.references = rate_database(self.database, self.meta, self.boundaries, self.state['indexmode'], self.references, self.unit_fmt, self.state['compound_mode'])
            self.state['update_on_change'] = False
        # update task, ds, environment and metrics
        self.state['task'] = task or self.state['task']
        self.state['task_ds'] = (self.state['task'], self.state['ds'])
        avail_envs = [ {"label": e, "value": e} for e in pd.unique(find_sub_db(self.database, self.state['ds'], self.state['task'])['environment']) ]
        sel_env = [avail_envs[0]['value']]
        avail_metrics = self.boundaries_real[(self.state['task'], self.state['ds'], sel_env[0])].keys()
        self.state['metrics'] = {prop: self.meta['properties'][prop] for prop in avail_metrics}
        axis_options = [{'label': lookup_meta(self.meta, metr, subdict='properties'), 'value': metr} for metr in self.state['metrics']]
        self.state['xaxis'] = self.defaults['x'][self.state['task_ds']]
        self.state['yaxis'] = self.defaults['y'][self.state['task_ds']]
        if 'weight' not in self.state['metrics'][self.state['xaxis']]: # safe defaults
            weights = _extract_weights(self.state['metrics'])
            for prop, weight in zip(self.state['metrics'], weights):
                self.state['metrics'][prop]['weight'] = weight
        # find corresponding sub database and models
        self.state['sub_database'] = find_sub_db(self.database, self.state['ds'], self.state['task'])
        models = self.state['sub_database']['model'].values
        ref_options = [{'label': mod, 'value': mod} for mod in models]
        curr_ref = self.references[self.state['ds']] if self.references is not None and self.state['ds'] in self.references else models[0]
        return avail_envs, sel_env, axis_options, self.state['xaxis'], axis_options, self.state['yaxis'], ref_options, curr_ref

    def display_model(self, hover_data=None, env_names=None, compound_mode=None):
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
            self.state['label'] = PropertyLabel(self.state['model'], self.state['metrics'], self.unit_fmt, custom=self.meta['meta_dir'])
            # TODO here, pass only the prop meta info for the properties in state!
            model_table, metric_table = summary_to_html_tables(self.state['model'], self.state['metrics'], self.unit_fmt)
            starplot = create_star_plot(self.state['model'], self.state['metrics']) # TODO display!
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
                self.update_database()
        self.state['xaxis'] = xaxis or self.state['xaxis']
        self.state['yaxis'] = yaxis or self.state['yaxis']
        return self.state['metrics'][self.state['xaxis']]['weight'], self.state['metrics'][self.state['yaxis']]['weight']

    def save_weights(self, save_weights_clicks=None):
        if save_weights_clicks is not None:
            return dict(content=save_weights(self.database), filename='weights.json')

    def save_label(self, lbl_clicks=None, lbl_clicks2=None, sum_clicks=None, log_clicks=None):
        if (lbl_clicks is None and lbl_clicks2 is None and sum_clicks is None and log_clicks is None) or self.state['model'] is None:
            return # callback init
        f_id = f'{self.state["model"]["model"]["name"]}_{self.state["model"]["environment"]}'.replace(' ', '_')
        if 'label' in dash.callback_context.triggered[0]['prop_id']:
            return dash.dcc.send_bytes(self.state['label'].write(), filename=f'energy_label_{f_id}.pdf')
        elif 'sum' in dash.callback_context.triggered[0]['prop_id']:
            return dict(content=json.dumps(self.state['model'], indent=4, cls=PatchedJSONEncoder), filename=f'energy_summary_{f_id}.json')
        else: # full logs
            # TODO load logs
            raise NotImplementedError
