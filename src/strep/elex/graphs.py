import os
from itertools import pairwise

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
from PIL import Image

from strep.util import lookup_meta, find_sub_db
from strep.elex.util import RATING_COLORS, ENV_SYMBOLS, PATTERNS, RATING_COLOR_SCALE, rgb_to_rgba
from strep.index_scale import calc_all_correlations

GRAD = Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'grad.png'))


def assemble_scatter_data(env_names, db, scale_switch, xaxis, yaxis, meta, unit_fmt):
    plot_data, substr = {}, '_index' if scale_switch == 'index' else ''
    for env in env_names:
        sub_db = find_sub_db(db, environment=env)
        dropped_na = sub_db.loc[~((sub_db[f'{xaxis}{substr}'].isna()) | (sub_db[f'{yaxis}{substr}'].isna()))]
        env_data = {
            'ratings': dropped_na['compound_rating'].values,
            'index': dropped_na['compound_index'].values,
            'x': dropped_na[f'{xaxis}{substr}'].values,
            'y': dropped_na[f'{yaxis}{substr}'].values,
            'names': dropped_na['model'].map(lambda mod: lookup_meta(meta, mod, key='short', subdict='model') ).tolist()
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


def add_rating_background(fig, rating_pos, use_grad=False, mode='mean', dark_mode=None, rowcol=None):
    xaxis_name, yaxis_name, add_args = "", "", {}
    if rowcol is not None:
        add_args['row'], add_args['col'] = rowcol[0], rowcol[1]
        plot_idx = rowcol[2] + 1
        if plot_idx > 1:
            xaxis_name, yaxis_name = str(plot_idx), str(plot_idx)
    xaxis, yaxis = fig.layout[f'xaxis{xaxis_name}'], fig.layout[f'yaxis{yaxis_name}']
    min_x, max_x, min_y, max_y = xaxis.range[0], xaxis.range[1], yaxis.range[0], yaxis.range[1]
    if dark_mode:
        add_args['line'] = dict(color='#0c122b')
    axis_sorted = [np.all(vals[:-1] <= vals[1:]) for vals in rating_pos]
    if use_grad: # use gradient background
        if axis_sorted[0]:
            transp_mode = 'TRANSPOSE' if axis_sorted[1] else 'FLIP_LEFT_RIGHT'
        else:
            transp_mode = 'FLIP_TOP_BOTTOM' if axis_sorted[1] else None
        grad = GRAD if transp_mode is None else GRAD.transpose(getattr(Image, transp_mode))
        if rowcol is None: # TODO improve and use kwargs instead of code copy?
            fig.add_layout_image(source=grad, xref="x domain", yref="y domain", x=1, y=1, xanchor="right", yanchor="top", sizex=1.0, sizey=1.0, sizing="stretch", opacity=0.75, layer="below")
        else:
            fig.add_layout_image(row=rowcol[0], col=rowcol[1], source=grad, xref="x domain", yref="y domain", x=1, y=1, xanchor="right", yanchor="top", sizex=1.0, sizey=1.0, sizing="stretch", opacity=0.75, layer="below")
    else:
        for idx, (vals, min_v, max_v) in enumerate(zip(rating_pos, [min_x, min_y], [max_x, max_y])):
            if len(rating_pos[idx]) == 6: # remove old border values
                vals = vals[1:5]
            # add values for bordering rectangles
            rating_pos[idx] = [min_v] + vals + [max_v] if axis_sorted[idx] else [max_v] + vals + [min_v]
        # plot each rectangle
        for xi, (x0, x1) in enumerate(pairwise(rating_pos[0])):
            for yi, (y0, y1) in enumerate(pairwise(rating_pos[1])):
                color = RATING_COLORS[int(getattr(np, mode)([xi, yi]))]
                fig.add_shape(type="rect", layer='below', fillcolor=color, x0=x0, x1=x1, y0=y0, y1=y1, opacity=.75, **add_args)


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

def create_correlation_graph(db, metrics, dark_mode):
    corr = calc_all_correlations(db)
    prop_names = [lookup_meta(metrics, prop, "shortname") for prop in list(corr.values())[0].columns]
    # take the average correlation across the different environments ( no effect if len(corr) == 1 )
    corr = pd.DataFrame(np.array(list(corr.values())).mean(axis=0), index=prop_names, columns=prop_names)
    fig = go.Figure(go.Heatmap(z=corr, x=prop_names, y=prop_names, coloraxis="coloraxis"))
    fig.update_layout(coloraxis={'colorscale': RATING_COLOR_SCALE, 'colorbar': {'title': 'Correlation'}},
                      margin={'l': 10, 'r': 10, 'b': 10, 't': 10})
    if dark_mode:
        fig.update_layout(template='plotly_dark', paper_bgcolor="#0c122b", plot_bgcolor="#0c122b")
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


def create_star_plot(summary, meta, scale='index', name=None, color=None, showlegend=True, return_trace=False):
    if scale == 'index':
        scale = '_index'
    elif scale == 'value':
        scale = ''
    else:
        raise NotImplementedError(f'Unsupported scale {scale}')
    name = name or summary['model']['name']
    color = color or RATING_COLORS[summary['compound_rating']]
    star_cols = [key for key in meta.keys() if key in summary]
    star_cols.append(star_cols[0])
    star_cols_short = [lookup_meta(meta, col, 'shortname') for col in star_cols]
    trace = go.Scatterpolar(
        r=[summary[f'{col}{scale}'] for col in star_cols], theta=star_cols_short,
        line={'color': color}, fillcolor=rgb_to_rgba(color, 0.1), fill='toself', name=name, showlegend=showlegend
    )
    if return_trace:
        return trace
    return go.Figure(trace)
    