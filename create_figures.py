import argparse
import importlib
import os

from main import DATABASES, load_database, scale_and_rate
from strep.util import lookup_meta, find_sub_db
from strep.elex.graphs import assemble_scatter_data, create_scatter_graph, add_rating_background
from strep.unit_reformatting import CustomUnitReformater

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

PLOT_WIDTH = 1000
PLOT_HEIGHT = PLOT_WIDTH // 3
COLORS = ['#009ee3', '#983082', '#ffbc29', '#35cdb4', '#e82e82', '#59bdf7', '#ec6469', '#706f6f', '#4a4ad8', '#0c122b', '#ffffff']
UNIT_FMT = CustomUnitReformater()

DISS_MATERIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'materials', 'dissertation', 'scripts_and_data')
DISS_FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'materials', 'dissertation', 'figures')

def load_db(db):
    database, meta = load_database(db)
    return scale_and_rate(database, meta)


def chapter1():
    pass


def chapter2(show=True):
    
    ##### ch2_ml_areas_activity
    # load dblp data
    dblp_df = pd.read_csv(os.path.join(DISS_MATERIAL, "ch2_parse_dblp_data.csv"), sep=';', index_col=1)
    dblp_df = dblp_df[(dblp_df.index > 2012) & (dblp_df.index < 2025)]
    importlib.util.find_spec("ch2_parse_dblp")
    spec = importlib.util.spec_from_file_location("ch2_parse_dblp", os.path.join(DISS_MATERIAL, 'ch2_parse_dblp.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # count keyword occurrences
    for kw, term in module.KEYWORDS.items():
        if term not in dblp_df.columns:
            dblp_df[term] = 0
        dblp_df[term] += dblp_df['title'].map(lambda t: 1 if kw in t.lower() else 0)
    terms = [col for col in dblp_df.columns if col not in ['author', 'title']]
    for term in terms:
        dblp_df[term] = dblp_df[term].astype(bool)
    dblp_df['overlapping fields'] = dblp_df[terms].sum(axis=1) > 1
    to_plot = dblp_df.groupby('year').sum().drop(['title', 'author'], axis=1)
    # plot
    to_plot[to_plot == 0] = 1
    fig = px.line(to_plot, markers=True, log_y=True)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                      legend=dict(title="Title mentioning AI and", yanchor="top", y=0.95, xanchor="left", x=0.02),
                      xaxis_title="Year", yaxis_title="Number of publications"
    )
    # fig = px.bar(to_plot, orientation='h')
    # fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    #                   legend=dict(title="Title mentioning AI and", yanchor="bottom", y=0.1, xanchor="right", x=0.9),
    #                   xaxis_title="Number of publications", yaxis_title="Year"
    # )
    if show:
        fig.show()
    fig.write_image("ch2_ml_areas_activity.pdf")


def chapter3():
    db, meta, defaults, idx_bounds, val_bounds, _ = load_db(DATABASES['ImageNetEff'])
    # value vs index scatter plot
    ds, task = 'imagenet', 'infer'
    xaxis, yaxis = defaults['x'][(task, ds)], defaults['y'][(task, ds)]
    bounds = {s: [b[(task, ds)][xaxis].tolist(), b[(task, ds)][yaxis].tolist()] for s, b in zip(['value', 'index'], [val_bounds, idx_bounds])}
    db = find_sub_db(db, dataset=ds, task=task)
    scatter = make_subplots(rows=1, cols=2, horizontal_spacing=.05, subplot_titles=['Real measurements', 'Index scaled values'])
    for idx, scale in enumerate(['value', 'index']):
        plot_data, axis_names = assemble_scatter_data(pd.unique(db['environment'])[:2], db, scale, xaxis, yaxis, meta, UNIT_FMT)
        traces = create_scatter_graph(plot_data, axis_names, dark_mode=False, display_text=False, marker_width=8, return_traces=True)
        scatter.add_traces(traces, rows=[1]*len(traces), cols=[idx+1]*len(traces))
        min_x, max_x = np.min([min(data['x']) for data in plot_data.values()]), np.max([max(data['x']) for data in plot_data.values()])
        min_y, max_y = np.min([min(data['y']) for data in plot_data.values()]), np.max([max(data['y']) for data in plot_data.values()])
        diff_x, diff_y = max_x - min_x, max_y - min_y
        scatter.update_xaxes(range=[min_x-0.1*diff_x, max_x+0.1*diff_x], showgrid=False, title=axis_names[0], row=1, col=idx+1)
        scatter.update_yaxes(range=[min_y-0.1*diff_y, max_y+0.1*diff_y], showgrid=False, title=axis_names[1], row=1, col=idx+1)
        add_rating_background(scatter, bounds[scale], 'mean', dark_mode=False, col=(idx+1))  
    for idx in [1, 2]:
        scatter.data[idx]['showlegend'] = False
    scatter.update_yaxes(side='right', row=1, col=2)
    scatter.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 25},
                            legend=dict(x=.5, y=0.05, orientation="h", xanchor="center", yanchor="bottom"))
    scatter.write_image(f"ch_.pdf")


def chapter4():
    pass


def chapter5():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", default=2)
    args = parser.parse_args()
    chapters = [chapter1, chapter2, chapter3, chapter4, chapter5]
    if args.chapter < 1 or args.chapter > 5:
        raise ValueError("Chapter number must be between 1 and 5")
    chapters[args.chapter-1]()
