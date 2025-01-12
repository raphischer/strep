import os
from itertools import pairwise

import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
from PIL import Image
import base64

from strep.util import lookup_meta, find_sub_db
from strep.elex.util import RATING_COLORS, ENV_SYMBOLS, PATTERNS, RATING_COLOR_SCALE

GRAD = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'grad.png'))


def assemble_scatter_data(env_names, db, scale_switch, xaxis, yaxis, meta, unit_fmt):
    plot_data, substr = {}, '_index' if scale_switch == 'index' else ''
    for env in env_names:
        sub_db = find_sub_db(db, environment=env)
        env_data = {
            'ratings': sub_db['compound_rating'].values,
            'index': sub_db['compound_index'].values,
            'x': sub_db[f'{xaxis}{substr}'].values,
            'y': sub_db[f'{yaxis}{substr}'].values,
            'names': sub_db['model'].map(lambda mod: lookup_meta(meta, mod, key='short', subdict='model') ).tolist()
        }
        plot_data[env] = env_data
    axis_names = [lookup_meta(meta, ax, subdict='properties') for ax in [xaxis, yaxis]]
    if scale_switch == 'index':
        axis_names = [name.split('[')[0].strip() + ' Index' if 'Index' not in name else name for name in axis_names]
    else:
        for idx, ax in enumerate([xaxis, yaxis]):
            unit = unit_fmt.reformat_value(1, lookup_meta(meta, ax, key='unit', subdict='properties'))[1]
            axis_names[idx] = f'{axis_names[idx]} {unit}'
    return plot_data, axis_names


def add_rating_background(fig, rating_pos, use_grad=False, mode=None, dark_mode=None, col=None):
    xaxis, yaxis = fig.layout[f'xaxis{col if col is not None and col > 1 else ""}'], fig.layout[f'yaxis{col if col is not None and col > 1 else ""}']
    min_x, max_x, min_y, max_y = xaxis.range[0], xaxis.range[1], yaxis.range[0], yaxis.range[1]
    add_args = {}
    if dark_mode:
        add_args['line'] = dict(color='#0c122b')
    if col is not None:
        add_args['row'], add_args['col'] = 1, col
    axis_sorted = [np.all(vals[:-1] <= vals[1:]) for vals in rating_pos]
    if use_grad: # use gradient background
        if axis_sorted[0]:
            transp_mode = 'TRANSPOSE' if axis_sorted[1] else 'FLIP_LEFT_RIGHT'
        else:
            transp_mode = 'FLIP_TOP_BOTTOM' if axis_sorted[1] else None
        grad = GRAD if transp_mode is None else GRAD.transpose(getattr(Image, transp_mode))
        if col is None: # TODO improve and use kwargs instead of code copy?
            fig.add_layout_image(source=grad, xref="x domain", yref="y domain", x=1, y=1, xanchor="right", yanchor="top", sizex=1.0, sizey=1.0, sizing="stretch", opacity=0.75, layer="below")
        else:
            fig.add_layout_image(row=1, col=col, source=grad, xref="x domain", yref="y domain", x=1, y=1, xanchor="right", yanchor="top", sizex=1.0, sizey=1.0, sizing="stretch", opacity=0.75, layer="below")
    else:
        # add values for bordering rectangles
        for idx, (vals, min_v, max_v) in enumerate(zip(rating_pos, [min_x, min_y], [max_x, max_y])):
            rating_pos[idx] = [min_v] + vals + [max_v] if axis_sorted[idx] else [max_v] + vals + [min_v]            
        # plot each rectangle
        for xi, (x0, x1) in enumerate(pairwise(rating_pos[0])):
            for yi, (y0, y1) in enumerate(pairwise(rating_pos[1])):
                color = RATING_COLORS[int(getattr(np, mode)([xi, yi]))]
                fig.add_shape(type="rect", layer='below', fillcolor=color, x0=x0, x1=x1, y0=y0, y1=y1, opacity=.8, **add_args)


def create_scatter_graph(plot_data, axis_title, dark_mode, ax_border=0.1, marker_width=15, norm_colors=True, display_text=False, return_traces=False):
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
    