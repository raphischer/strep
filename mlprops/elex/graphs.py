import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

from mlprops.util import lookup_meta
from mlprops.index_and_rate import calculate_compound_rating, find_sub_db
from mlprops.elex.util import RATING_COLORS, ENV_SYMBOLS, PATTERNS, RATING_COLOR_SCALE


def assemble_scatter_data(env_names, db, scale_switch, xaxis, yaxis, meta, boundaries):
    plot_data = {}
    for env in env_names:
        env_data = { 'ratings': [], 'x': [], 'y': [], 'index': [] }
        for _, log in find_sub_db(db, environment=env).iterrows():
            env_data['ratings'].append(log['compound_rating'])
            env_data['index'].append(log['compound_index'])
            for xy_axis, metric in zip(['x', 'y'], [xaxis, yaxis]):
                if isinstance(log[metric], dict): # either take the value or the index of the metric
                    env_data[xy_axis].append(log[metric][scale_switch])
                else: # error during value aggregation
                    env_data[xy_axis].append(0)
        plot_data[env] = env_data
    axis_names = [lookup_meta(meta, ax, subdict='properties') for ax in [xaxis, yaxis]]
    if scale_switch == 'index':
        axis_names = [name.split('[')[0].strip() + ' Index' for name in axis_names]
    rating_pos = [boundaries[xaxis], boundaries[yaxis]]
    return plot_data, axis_names, rating_pos


def add_rating_background(fig, rating_pos, mode, dark_mode):
    for xi, (x0, x1) in enumerate(rating_pos[0]):
        for yi, (y0, y1) in enumerate(rating_pos[1]):
            color = RATING_COLORS[int(calculate_compound_rating([xi, yi], mode))]
            if dark_mode:
                fig.add_shape(type="rect", layer='below', line=dict(color='#0c122b'), fillcolor=color, x0=x0, x1=x1, y0=y0, y1=y1, opacity=.8)
            else:
                fig.add_shape(type="rect", layer='below', fillcolor=color, x0=x0, x1=x1, y0=y0, y1=y1, opacity=.8)


def create_scatter_graph(plot_data, axis_title, dark_mode, ax_border=0.1, marker_width=15, norm_colors=True):
    fig = go.Figure()
    for env, data in plot_data.items():
        if 'names' not in data:
            plot_data[env]['names'] = ['' for _ in range(len(data['x']))]
    i_min, i_max = min([min(vals['index']) for vals in plot_data.values()]), max([max(vals['index']) for vals in plot_data.values()])
    for env_i, (env_name, data) in enumerate(plot_data.items()):
        # scale to vals between 0 and 1?
        index_vals = (np.array(data['index']) - i_min) / (i_max - i_min) if norm_colors else data['index']
        node_col = sample_colorscale(RATING_COLOR_SCALE, [1-val for val in index_vals])

        fig.add_trace(go.Scatter(
            x=data['x'], y=data['y'], name=env_name, text=data['names'],
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
    print(1)
    