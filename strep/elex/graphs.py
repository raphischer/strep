import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

from strep.util import lookup_meta
from strep.index_and_rate import calculate_single_compound_rating, find_sub_db
from strep.elex.util import RATING_COLORS, ENV_SYMBOLS, PATTERNS, RATING_COLOR_SCALE


def assemble_scatter_data(env_names, db, scale_switch, xaxis, yaxis, meta, boundaries):
    plot_data = {}
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
    rating_pos = [boundaries[xaxis], boundaries[yaxis]]
    return plot_data, axis_names, rating_pos


def add_rating_background(fig, rating_pos, mode, dark_mode):
    for xi, (x0, x1) in enumerate(rating_pos[0]):
        if xi == 0:
            x0 = fig.layout.xaxis.range[1]
        if xi == len(rating_pos[0]) - 1:
            x1 = fig.layout.xaxis.range[0]
        for yi, (y0, y1) in enumerate(rating_pos[1]):
            if yi == 0:
                y0 = fig.layout.yaxis.range[1]
            if yi == len(rating_pos[1]) - 1:
                y1 = fig.layout.yaxis.range[0]
            color = RATING_COLORS[int(calculate_single_compound_rating([xi, yi], mode))]
            if dark_mode:
                fig.add_shape(type="rect", layer='below', line=dict(color='#0c122b'), fillcolor=color, x0=x0, x1=x1, y0=y0, y1=y1, opacity=.8)
            else:
                fig.add_shape(type="rect", layer='below', fillcolor=color, x0=x0, x1=x1, y0=y0, y1=y1, opacity=.8)


def create_scatter_graph(plot_data, axis_title, dark_mode, ax_border=0.1, marker_width=15, norm_colors=True):
    fig = go.Figure()
    i_min, i_max = min([min(vals['index']) for vals in plot_data.values()]), max([max(vals['index']) for vals in plot_data.values()])
     # link model scatter points across multiple environment
    if len(plot_data) > 1:
        models = set.union(*[set(data['names']) for data in plot_data.values()])
        x, y, text = [], [], []
        for model in models:
            avail = 0
            for d_idx, data in enumerate(plot_data.values()):
                try:
                    idx = data['names'].index(model)
                    avail += 1
                    x.append(data['x'][idx])
                    y.append(data['y'][idx])
                except ValueError:
                    pass
            model_text = ['' if i != (avail - 1) // 2 else model for i in range(avail + 1)]
            text = text + model_text # place text near most middle node
            x.append(None)
            y.append(None)
        fig.add_trace(go.Scatter(x=x, y=y, text=text, mode='lines+text', line={'color': 'black'}, showlegend=False))
    for env_i, (env_name, data) in enumerate(plot_data.items()):
        # scale to vals between 0 and 1?
        index_vals = (np.array(data['index']) - i_min) / (i_max - i_min) if norm_colors else data['index']
        node_col = sample_colorscale(RATING_COLOR_SCALE, [1-val for val in index_vals])
        text = [''] * len(data['x']) if 'names' not in data or len(plot_data) > 1 else data['names']
        fig.add_trace(go.Scatter(
            x=data['x'], y=data['y'], name=env_name, text=text,
            mode='markers+text', marker_symbol=ENV_SYMBOLS[env_i],
            legendgroup=env_name, marker=dict(color=node_col, size=marker_width),
            marker_line=dict(width=marker_width / 5, color='black'))
        )
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
    