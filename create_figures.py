import argparse
import importlib
import os
import time

# STREP imports
from main import DATABASES, load_database, scale_and_rate
from strep.util import lookup_meta, find_sub_db, fill_meta
from strep.elex.graphs import assemble_scatter_data, create_scatter_graph, add_rating_background, create_star_plot
from strep.unit_reformatting import CustomUnitReformater
from strep.labels.label_generation import PropertyLabel

# external libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale, make_colorscale

PLOT_WIDTH = 800
PLOT_HEIGHT = PLOT_WIDTH // 3

COLORS = ['#009ee3', '#983082', '#ffbc29', '#35cdb4', '#e82e82', '#59bdf7', '#ec6469', '#706f6f', '#4a4ad8', '#0c122b', '#ffffff']
LAMARR_COLORS = [
    '#009ee3', # aqua
    '#983082', # fresh violet
    '#ffbc29', # sunshine
    '#35cdb4', # carribean
    '#e82e82', # fuchsia
    '#59bdf7', # sky blue
    '#ec6469', # indian red
    '#706f6f', # gray
    '#4a4ad8', # corn flower
    '#0c122b', # dark corn flower
    '#ffffff'
]
LAM_COL_SCALE = make_colorscale([LAMARR_COLORS[0], LAMARR_COLORS[2], LAMARR_COLORS[4]])
LAM_COL_FIVE = sample_colorscale(LAM_COL_SCALE, np.linspace(0, 1, 5))
LAM_COL_TEN = sample_colorscale(LAM_COL_SCALE, np.linspace(0, 1, 10))

UNIT_FMT = CustomUnitReformater()

DISS_MATERIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'materials', 'dissertation', 'scripts_and_data')
DISS_FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'materials', 'dissertation', 'figures')


def load_db(db):
    database, meta = load_database(db)
    return scale_and_rate(database, meta)


def tex(text):
    return r'$\text{' + text + r'}$'


def print_init(fname):
    print(f'                 - -- ---  {fname:<20}  --- -- -                 ')
    return fname


def finalize(fig, fname, show):
    if show:
        fig.show()
    fig.write_image(os.path.join(DISS_FIGURES, f"{fname}.pdf"))


def chapter1(show):
    pass


def chapter2(show):
    fname = print_init('ch2_ml_areas_activity') ###############################################################################
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
    fig = px.line(to_plot, color_discrete_sequence=LAMARR_COLORS, markers=True, log_y=True)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 26},
                      title='Paper titles mentioning AI and...',
                      xaxis_title="Year", yaxis_title="Number of publications", yaxis_range=[-0.1, 4],
                      legend=dict(title='', itemwidth=80, bgcolor='rgba(0,0,0,0)', orientation='h', yanchor="top", y=0.98, xanchor="center", x=0.5)
    )
    # fig = px.bar(to_plot, orientation='h')
    # fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    #                   legend=dict(title="Title mentioning AI and", yanchor="bottom", y=0.1, xanchor="right", x=0.9),
    #                   xaxis_title="Number of publications", yaxis_title="Year"
    # )
    finalize(fig, fname, show)


def chapter3(show):

    ############################### EdgeAcc Results (Section 3.3.2) #######################################################
    db, meta, defaults, idx_bounds, val_bounds, _ = load_db(DATABASES['ImageNetEff22'])
    


    ############################### ImageNet Results (Section 3.3.1) #######################################################
    db, meta, defaults, idx_bounds, val_bounds, _ = load_db(DATABASES['ImageNetEff22'])
    ds, task, bounds = 'imagenet', 'infer', {'value': val_bounds, 'index': idx_bounds}
    MOD_SEL = 'EfficientNetB2', 'VGG16', 'MobileNetV3Small'
    ENV_SEL = pd.unique(db['environment'])[:2]
    task_props = {prop: meta for prop, meta in meta['properties'].items() if prop in idx_bounds[task, ds, ENV_SEL[0]]}

    fname = print_init('ch3_imagenet_stars') ###############################################################################
    fig = make_subplots(rows=1, cols=len(MOD_SEL), specs=[[{'type': 'polar'}] * len(MOD_SEL)], subplot_titles=MOD_SEL)
    for idx, mod in enumerate(MOD_SEL):
        for e_idx, env in enumerate(ENV_SEL):
            model = find_sub_db(db, ds, task, env, mod).iloc[0].to_dict()
            summary = fill_meta(model, meta)
            trace = create_star_plot(summary, task_props, name=env, color=LAM_COL_FIVE[e_idx], showlegend=idx==0, return_trace=True)
            fig.add_trace(trace, row=1, col=idx+1)
            if e_idx == 0:
                label = PropertyLabel(summary, task_props, UNIT_FMT)
                label.save(os.path.join(DISS_FIGURES, f'ch3_label_{mod}.pdf'))
    fig.update_annotations(yshift=20)
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)), width=PLOT_WIDTH, height=PLOT_HEIGHT,
        legend=dict( yanchor="top", y=-0.1, xanchor="center", x=0.5, orientation='h'), margin={'l': 10, 'r': 10, 'b': 0, 't': 40}
    )
    finalize(fig, fname, show)

    fname = print_init('ch3_imagenet_tradeoffs') ###############################################################################
    env = pd.unique(db['environment'])[0]
    scatter = make_subplots(rows=2, cols=2, shared_yaxes=True, horizontal_spacing=.02, vertical_spacing=.1)
    for idx, (xaxis, yaxis, t) in enumerate([['power_draw', 'top-1_val', 'infer'], ['train_power_draw', 'top-1_val', 'train'], ['running_time', 'parameters', 'infer'], ['fsize', 'parameters', 'train']]):
        row, col = (idx // 2) + 1, (idx % 2) + 1
        ax_bounds = [val_bounds[(t, ds, env)][xaxis].tolist(), val_bounds[(t, ds, env)][yaxis].tolist()]
        plot_data, axis_names = assemble_scatter_data([env], db, 'value', xaxis, yaxis, meta, UNIT_FMT)
        traces = create_scatter_graph(plot_data, axis_names, dark_mode=False, display_text=True, marker_width=8, return_traces=True)
        scatter.add_traces(traces, rows=[row]*len(traces), cols=[col]*len(traces))
        min_x, max_x = np.min([min(data['x']) for data in plot_data.values()]), np.max([max(data['x']) for data in plot_data.values()])
        min_y, max_y = np.min([min(data['y']) for data in plot_data.values()]), np.max([max(data['y']) for data in plot_data.values()])
        diff_x, diff_y = max_x - min_x, max_y - min_y
        scatter.update_xaxes(range=[min_x-0.1*diff_x, max_x+0.1*diff_x], showgrid=False, title=tex(axis_names[0]), row=row, col=col)
        scatter.update_yaxes(range=[min_y-0.1*diff_y, max_y+0.1*diff_y], showgrid=False, row=row, col=col)
        if col == 1:
            scatter.update_yaxes(title=tex(axis_names[1]), row=row, col=col)
        add_rating_background(scatter, ax_bounds, rowcol=(row, col, idx))
        scatter.data[idx]['showlegend'] = False
    # scatter.update_traces(textposition='top center')
    scatter.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*2, margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                          legend=dict(x=.5, y=0.05, orientation="h", xanchor="center", yanchor="bottom"))
    finalize(scatter, fname, show)

    fname = print_init('ch3_index_scaling') ###############################################################################
    xaxis, yaxis = defaults['x'][(task, ds)], defaults['y'][(task, ds)]
    db = find_sub_db(db, dataset=ds, task=task)
    scatter = make_subplots(rows=1, cols=2, horizontal_spacing=.02, subplot_titles=[r'$\text{Real-values properties }\mu$', r'$\text{Index-scaled properties }\tilde{\mu}$'])
    for idx, scale in enumerate(['value', 'index']):
        plot_data, axis_names = assemble_scatter_data(ENV_SEL, db, scale, xaxis, yaxis, meta, UNIT_FMT)
        traces = create_scatter_graph(plot_data, axis_names, dark_mode=False, display_text=False, marker_width=8, return_traces=True)
        scatter.add_traces(traces, rows=[1]*len(traces), cols=[idx+1]*len(traces))
        min_x, max_x = np.min([min(data['x']) for data in plot_data.values()]), np.max([max(data['x']) for data in plot_data.values()])
        min_y, max_y = np.min([min(data['y']) for data in plot_data.values()]), np.max([max(data['y']) for data in plot_data.values()])
        diff_x, diff_y = max_x - min_x, max_y - min_y
        scatter.update_xaxes(range=[min_x-0.1*diff_x, max_x+0.1*diff_x], showgrid=False, title=tex(axis_names[0]), row=1, col=idx+1)
        scatter.update_yaxes(range=[min_y-0.1*diff_y, max_y+0.1*diff_y], showgrid=False, title=tex(axis_names[1]), row=1, col=idx+1)
        ax_bounds = [bounds[scale][(task, ds, env)][xaxis].tolist(), bounds[scale][(task, ds, env)][yaxis].tolist()]
        add_rating_background(scatter, ax_bounds, True, 'mean', dark_mode=False, rowcol=(1, idx+1, idx))
    for idx in [1, 2]:
        scatter.data[idx]['showlegend'] = False
    scatter.update_yaxes(side='right', row=1, col=2)
    scatter.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 25},
                            legend=dict(x=.5, y=0.05, orientation="h", xanchor="center", yanchor="bottom"))
    finalize(scatter, fname, show)


def chapter4(show):
    pass


def chapter5(show):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", type=int, default=-1)
    parser.add_argument("--show", default=True)
    args = parser.parse_args()

    ####### DUMMY OUTPUT - for setting up pdf export of plotly
    fig = px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")

    chapters = [chapter1, chapter2, chapter3, chapter4, chapter5]
    if args.chapter == -1:
        for i in range(5):
            chapters[i](args.show)

    if args.chapter < 1 or args.chapter > 5:
        raise ValueError("Chapter number must be between 1 and 5")
    
    ####### print chapter figures
    chapters[args.chapter-1](args.show)
    os.remove("dummy.pdf")