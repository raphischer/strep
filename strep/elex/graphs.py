import os

import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
from PIL import Image

from strep.util import lookup_meta
from strep.index_and_rate import calculate_single_compound_rating, find_sub_db
from strep.elex.util import RATING_COLORS, ENV_SYMBOLS, PATTERNS, RATING_COLOR_SCALE

GRAD = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'grad.png'))


def assemble_scatter_data(env_names, db, scale_switch, xaxis, yaxis, meta, boundaries):
    plot_data = {}
    if scale_switch != 'index':
        boundaries = boundaries[(db['dataset'].iloc[0], db['task'].iloc[0], env_names[0])]
    for env in env_names:
        env_data = { 'ratings': [], 'x': [], 'y': [], 'index': [], 'names': [] }
        for _, log in find_sub_db(db, environment=env).iterrows():
            env_data['ratings'].append(log['compound_rating'])
            env_data['index'].append(log['compound_index'])
            env_data['names'].append(lookup_meta(meta, log['model'], key='short', subdict='model'))
            for xy_axis, metric in zip(['x', 'y'], [xaxis, yaxis]):
                if isinstance(log[metric], dict): # either take the value or the index of the metric
                    env_data[xy_axis].append(log[metric][scale_switch])
                elif isinstance(log[metric], float):
                    if scale_switch != 'index':
                        print(f'WARNING: Only index values found for displaying {metric}!')
                    env_data[xy_axis].append(log[metric])
                else: # error during value aggregation
                    env_data[xy_axis].append(0)
        plot_data[env] = env_data
    axis_names = [lookup_meta(meta, ax, subdict='properties') for ax in [xaxis, yaxis]]
    if scale_switch == 'index':
        axis_names = [name.split('[')[0].strip() + ' Index' if 'Index' not in name else name for name in axis_names]
    return plot_data, axis_names, [boundaries[xaxis], boundaries[yaxis]]


def add_rating_background(fig, rating_pos, mode=None, dark_mode=None, col=None):
    xaxis, yaxis = fig.layout[f'xaxis{col if col is not None and col > 1 else ""}'], fig.layout[f'yaxis{col if col is not None and col > 1 else ""}']
    min_x, max_x, min_y, max_y = xaxis.range[0], xaxis.range[1], yaxis.range[0], yaxis.range[1]
    if rating_pos is None:
        grad = GRAD if mode is None else GRAD.transpose(getattr(Image, mode))
        fig.add_layout_image(dict(source=grad, xref="x", yref="y", x=min_x, y=max_y, sizex=max_x-min_x, sizey=max_y-min_y, sizing="stretch", opacity=0.75, layer="below"))
    else:
        for xi, (x0, x1) in enumerate(rating_pos[0]):
            x0 = max_x if xi == 0 and x0 > x1 else (min_x if xi == 0 else x0)
            x1 = min_x if xi == len(rating_pos[0]) - 1 and x0 > x1 else (max_x if xi == len(rating_pos[0]) - 1 else x1)
            for yi, (y0, y1) in enumerate(rating_pos[1]):
                y0 = max_y if yi == 0 and y0 > y1 else (min_y if yi == 0 else y0)
                y1 = min_y if yi == len(rating_pos[1]) - 1 and y0 > y1 else (max_y if yi == len(rating_pos[1]) - 1 else y1)
                color = RATING_COLORS[int(calculate_single_compound_rating([xi, yi], mode))]
                add_args = {}
                if dark_mode:
                    add_args['line'] = dict(color='#0c122b')
                if col is not None:
                    add_args['row'], add_args['col'] = 1, col
                fig.add_shape(type="rect", layer='below', fillcolor=color, x0=x0, x1=x1, y0=y0, y1=y1, opacity=.8, **add_args)


def create_scatter_graph(plot_data, axis_title, dark_mode, ax_border=0.1, marker_width=15, norm_colors=True, display_text=True, return_traces=False):
    traces = []
    i_min, i_max = min([min(vals['index']) for vals in plot_data.values()]), max([max(vals['index']) for vals in plot_data.values()])
     # link model scatter points across multiple environment
    if len(plot_data) > 1:
        models = set.union(*[set(data['names']) for data in plot_data.values()])
        x, y, text = [], [], []
        for model in models:
            avail = 0
            for _, data in enumerate(plot_data.values()):
                try:
                    idx = data['names'].index(model)
                    avail += 1
                    x.append(data['x'][idx])
                    y.append(data['y'][idx])
                except ValueError:
                    pass
            text = text + ['' if i != (avail - 1) // 2 or not display_text else model for i in range(avail + 1)] # place text near most middle node
            x.append(None)
            y.append(None)
        traces.append(go.Scatter(x=x, y=y, text=text, mode='lines+text', line={'color': 'black', 'width': marker_width / 5}, showlegend=False))
    for env_i, (env_name, data) in enumerate(plot_data.items()):
        # scale to vals between 0 and 1?
        index_vals = (np.array(data['index']) - i_min) / (i_max - i_min) if norm_colors else data['index']
        node_col = sample_colorscale(RATING_COLOR_SCALE, [1-val for val in index_vals])
        text = [''] * len(data['x']) if (not display_text) or ('names' not in data) or (len(plot_data) > 1) else data['names']
        traces.append(go.Scatter(
            x=data['x'], y=data['y'], name=env_name, text=text,
            mode='markers+text', marker_symbol=ENV_SYMBOLS[env_i],
            legendgroup=env_name, marker=dict(color=node_col, size=marker_width),
            marker_line=dict(width=marker_width / 5, color='black'))
        )
    if return_traces:
        return traces
    fig = go.Figure(traces)
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title=axis_title[0], yaxis_title=axis_title[1])
    fig.update_layout(legend=dict(x=.5, y=1, orientation="h", xanchor="center", yanchor="bottom",))
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    if dark_mode:
        fig.update_layout(template='plotly_dark', paper_bgcolor="#0c122b", plot_bgcolor="#0c122b")
    min_x, max_x = np.min([min(data['x']) for data in plot_data.values()]), np.max([max(data['x']) for data in plot_data.values()])
    min_y, max_y = np.min([min(data['y']) for data in plot_data.values()]), np.max([max(data['y']) for data in plot_data.values()])
    diff_x, diff_y = max_x - min_x, max_y - min_y
    fig.update_layout(
        xaxis_range=[min_x - ax_border * diff_x, max_x + ax_border * diff_x],
        yaxis_range=[min_y - ax_border * diff_y, max_y + ax_border * diff_y],
        margin={'l': 10, 'r': 10, 'b': 10, 't': 10}
    )
    return fig

def create_bar_graph(plot_data, dark_mode, discard_y_axis):
    fig = go.Figure()
    for env_i, (env_name, data) in enumerate(plot_data.items()):
        counts = np.zeros(len(RATING_COLORS), dtype=int)
        unq, cnt = np.unique(data['ratings'], return_counts=True)
        for u, c in zip(unq, cnt):
            counts[u] = c
        fig.add_trace(go.Bar(
            name=env_name, x=['A', 'B', 'C', 'D', 'E', 'N.A.'], y=counts, legendgroup=env_name,
            marker_pattern_shape=PATTERNS[env_i], marker_color=RATING_COLORS, showlegend=False)
        )
    fig.update_layout(barmode='stack')
    fig.update_layout(margin={'l': 10, 'r': 10, 'b': 10, 't': 10})
    fig.update_layout(xaxis_title='Final Rating')
    if not discard_y_axis:
        fig.update_layout(yaxis_title='Number of Ratings')
    if dark_mode:
        fig.update_layout(template='plotly_dark', paper_bgcolor="#0c122b", plot_bgcolor="#0c122b")
    return fig


def create_star_plot(summary, metrics):
    pass # TODO
    