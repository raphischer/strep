import os
import time
import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats.stats import pearsonr

from mlprops.util import read_json, lookup_meta
from mlprops.index_and_rate import prop_dict_to_val
from mlprops.load_experiment_logs import find_sub_db
from mlprops.elex.util import ENV_SYMBOLS, RATING_COLORS
from mlprops.elex.graphs import create_scatter_graph, add_rating_background


PLOT_WIDTH = 1000
PLOT_HEIGHT = PLOT_WIDTH // 3
COLORS = ['#009ee3', '#983082', '#ffbc29', '#35cdb4', '#e82e82', '#59bdf7', '#ec6469', '#706f6f', '#4a4ad8', '#0c122b', '#ffffff']




# import spacy
# nlp = spacy.load('en_core_web_lg')
# words = ' '.join(database.select_dtypes('number').columns)
# tokens = nlp(words)
# sim_matr = np.ones((database.select_dtypes('number').shape[0], database.select_dtypes('number').shape[0]))
# for x, token1 in enumerate(tokens):
#     for y, token2 in enumerate(tokens):
#         sim_matr[x,y] = token1.similarity(token2)



def create_all(databases):
    filterstats = read_json('databases/paperswithcode/filterstats.json')
    pwc_stats = read_json('databases/paperswithcode/other_stats.json')
    os.chdir('paper_results')

    ####### DUMMY OUTPUT #######
    # for setting up pdf export of plotly
    fig=px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    time.sleep(0.5)
    os.remove("dummy.pdf")



    # PWC filtering
    fig = go.Figure(layout={
        'width': PLOT_WIDTH / 2, 'height': PLOT_HEIGHT, 'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
        'xaxis':{'title': 'Number of evaluations'}, 'yaxis':{'title': 'Number of properties'}}
    )
    pos = [ 'bottom left', 'top right', 'middle right', 'bottom right' ]
    for idx, (key, shape) in enumerate(filterstats.items()):
        color = RATING_COLORS[idx]
        x, y = shape[0], shape[1]
        fig.add_shape(type="rect", fillcolor=color, layer='below', x0=0, x1=x, y0=0, y1=y, opacity=.8, name=key)
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text", marker={'color': color}, text=[key],
            textposition=pos[idx], showlegend=False,
        ))
    fig.write_image("db_filtered.pdf")

    pwc_stats = pd.DataFrame(pwc_stats).transpose()
    for _, data in pwc_stats.groupby(['n_results', 'n_metrics']):
        pwc_stats.loc[data.index,'count'] = data.shape[0]
    pwc_stats['log_count'] = np.log(pwc_stats['count'])
    pwc_stats['log_n_results'] = np.log(pwc_stats['n_results'])
    pwc_stats.loc[(pwc_stats['n_results'] == 0).index,'log_n_results'] = 0
    fig = px.scatter(data_frame=pwc_stats, x='log_n_results', y='n_metrics', color='log_count')
    fig.update_layout(
        width=PLOT_WIDTH / 2, height=PLOT_HEIGHT,
        coloraxis_colorbar_tickvals=np.linspace(0, pwc_stats['log_count'].max(), 5), coloraxis_colorbar_ticktext=np.quantile(pwc_stats['count'], [0, .25, .50, .75, 1]),
        xaxis = dict(tickmode='array', tickvals=np.linspace(0, pwc_stats['log_n_results'].max(), 5), ticktext=np.quantile(pwc_stats['n_results'], [0, .25, .50, .75, 1]))
    )
    fig.write_image(f'pwc_stats.pdf')

    # papers with code property correlations
    fig = go.Figure()
    for key, db in databases.items():
        pwc_db, pwc_metrics = db[0], db[2]
        corr = []
        for (ds, task), data in pwc_db.groupby(['dataset', 'task']):
            num_cols = pwc_metrics[(ds, task)]
            props = prop_dict_to_val(data[num_cols]).dropna()
            if props.size > 1:
                for col_a, col_b in itertools.combinations(props.columns, 2):
                    pr = pearsonr(props.loc[:, col_a], props.loc[:, col_b])[0]
                    if not np.isnan(pr):
                        corr.append(pr)
        fig.add_trace(go.Violin(
            y=corr,
            name=key,
            box_visible=True,
            meanline_visible=True)
        )
    fig.update_layout(width=PLOT_WIDTH / 2, height=PLOT_HEIGHT, xaxis={'visible': False, 'showticklabels': False})
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5) )
    fig.write_image(f'prop_corr.pdf')

    # imagenet results
    db, meta, metrics, xdef, ydef, bounds, _, _ = databases['ImageNet Classification']

    plot_data = {}
    env = db['environment'].iloc[0]
    ds, task = 'imagenet', 'infer'
    env_data = { 'ratings': [], 'x': [], 'y': [], 'index': [] }
    xaxis, yaxis = xdef[(ds, task)], ydef[(ds, task)]
    for _, log in find_sub_db(db, dataset=ds, task=task, environment=env).iterrows():
        env_data['ratings'].append(log['compound_rating'])
        env_data['index'].append(log['compound_index'])
        for xy_axis, metric in zip(['x', 'y'], [xaxis, yaxis]):
            if isinstance(log[metric], dict): # either take the value or the index of the metric
                env_data[xy_axis].append(log[metric]['index'])
            else: # error during value aggregation
                env_data[xy_axis].append(0)
    plot_data[env] = env_data
    axis_names = [lookup_meta(meta, ax, subdict='properties') for ax in [xaxis, yaxis]] # TODO pretty print, use name of axis?
    rating_pos = [bounds[xaxis], bounds[yaxis]]
    rating_pos[0][0][0] = scatter.layout.xaxis.range[1]
    rating_pos[1][0][0] = scatter.layout.yaxis.range[1]
    axis_names = [name.split('[')[0].strip() + ' Index' for name in axis_names]
    scatter = create_scatter_graph(plot_data, axis_names, dark_mode=False)
    add_rating_background(scatter, rating_pos, 'optimistic mean', dark_mode=False)
    fig.write_image("scatter.pdf")





    # imagenet env trades
    envs = sorted([env for env in pd.unique(db['environment']) if 'Xeon' not in env])
    models = sorted(pd.unique(db['model']).tolist())
    traces = {}
    for env in envs:
        subdb = db[(db['environment'] == env) & (db['task'] == 'infer')]
        avail_models = set(subdb['model'].tolist())
        traces[env] = [subdb[subdb['model'] == mod]['compound_index'].iloc[0] if mod in avail_models else None for mod in models]
        
    fig = go.Figure(
        layout={'width': PLOT_WIDTH, 'height': PLOT_HEIGHT, 'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
                'xaxis':{'title': 'Model'}, 'yaxis':{'title': 'Compound score'}},
        data=[
            go.Scatter(x=models, y=vals, name=env, mode='markers',
            marker=dict(
                color=COLORS[i],
                symbol=ENV_SYMBOLS[i]
            ),) for i, (env, vals) in enumerate(traces.items())
        ])

    fig.update_layout( legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5) )
    fig.write_image(f'environment_changes.pdf')
    fig.show()



if __name__ == '__main__':
    pass