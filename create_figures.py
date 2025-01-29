import argparse
import importlib
import os
import time
from itertools import product

# STREP imports
from main import DATABASES
from strep.index_scale import load_database, scale_and_rate, _extract_weights
from strep.correlations import calc_correlation, calc_all_correlations
from strep.util import lookup_meta, find_sub_db, fill_meta, loopup_task_ds_metrics, prop_dict_to_val
from strep.elex.util import RATING_COLORS, RATING_COLOR_SCALE, RATING_COLOR_SCALE_REV
from strep.elex.graphs import assemble_scatter_data, create_scatter_graph, add_rating_background, create_star_plot
from strep.unit_reformatting import CustomUnitReformater
from strep.labels.label_generation import PropertyLabel

# external libraries
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale, make_colorscale

PLOT_WIDTH = 800
PLOT_HEIGHT = PLOT_WIDTH // 3

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


def finalize_tex(data, fname):
    print(1)


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

    DBS = {}

    ############################### ImageNet Results (Section 3.3.1) #######################################################
    DBS['ImageNetEff22'] = load_db(DATABASES['ImageNetEff22'])
    db, meta, defaults, idx_bounds, val_bounds, _ = DBS['ImageNetEff22']
    ds, task, bounds = 'imagenet', 'infer', {'value': val_bounds, 'index': idx_bounds}
    MOD_SEL = 'EfficientNetB2', 'VGG16', 'MobileNetV3Small'
    ENV_SEL = pd.unique(db['environment'])[:2]
    task_props = {prop: meta for prop, meta in meta['properties'].items() if prop in idx_bounds[task, ds, ENV_SEL[0]]}
    weights = _extract_weights(task_props)
    for prop, weight in zip(task_props, weights):
        task_props[prop]['weight'] = weight

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

    ############################## EdgeAcc Results (Section 3.3.2) #######################################################
    DBS['EdgeAccUSB'] = load_db(DATABASES['EdgeAccUSB'])
    full_db, meta, defaults, idx_bounds, val_bounds, _ = DBS['EdgeAccUSB']
    envs, env_cols, env_symb, models,  = sorted(pd.unique(full_db['environment']).tolist()), {}, {}, {}
    MOD_SEL = {
        'MobileNetV2': 'Laptop',
        'yolov8s-seg': 'Desktop',
        'NASNetMobile': 'RasPi'
    }
    eni_metr, rti_metr = 'approx_USB_power_draw', 'running_time'
    for env in envs:
        if 'TPU' in env:
            env_cols[env] = LAM_COL_FIVE[4]
        else:
            env_cols[env] = LAM_COL_FIVE[0] if 'NCS' in env else LAM_COL_FIVE[2]
        if 'Desktop' in env:
            env_symb[env] = "circle"
        else:
            env_symb[env] = "cross" if 'Laptop' in env else "x"
    host_envs = { host: [env for env in envs if host in env] for host in ['Desktop', 'Laptop', 'RasPi'] }
    for ds in pd.unique(full_db['dataset']):
        subdb = full_db[full_db['dataset'] == ds]
        models[ds] = sorted(pd.unique(subdb['model']).tolist()) # [mod for mod in pd.unique(subdb['model']) if subdb[subdb['model'] == mod].shape[0] > 3])
    models_cls, models_seg = models['imagenet'], models['coco']
    models = models_cls + [None] + models_seg
    model_names = [f'{mod[:4]}..{mod[-6:]}' if mod is not None and len(mod) > 12 else mod for mod in models]

    ############# Collect results (comparison table not used)
    all_improvements = {'NCS': {rti_metr: [], eni_metr: []}, 'TPU': {rti_metr: [], eni_metr: []}}
    db = full_db[full_db['dataset'] == 'imagenet']
    for model, short in zip(models, model_names):
        if model is None:
            db = full_db[full_db['dataset'] == 'coco']
        else:
            row = [short]
            for host in host_envs:
                subdb = db[(db['model'] == model) & (db['architecture'] == host)]
                results = {eni_metr: {}, rti_metr: {}}
                for proc, metric in product(['CPU', 'NCS', 'TPU'], results.keys()):
                    if subdb[subdb['backend'] == proc][metric].shape[0] > 0:
                        results[metric][proc] = subdb[subdb['backend'] == proc][metric].iloc[0]
                to_add = ['---', '---']
                if 'CPU' in results[eni_metr]:
                    to_add[0] = f"{results[eni_metr]['CPU']:4.2f}"
                    for metric, proc in product(results.keys(), ['TPU', 'NCS']):
                        value = results[metric][proc] / results[metric]['CPU'] * 100 if proc in results[metric] else np.inf
                        all_improvements[proc][metric].append(value)

    fname = print_init('ch3_edge_summary2') ###############################################################################
    fig = go.Figure()
    fig.add_hline(y=100, line_dash="dot", annotation_text="CPU-only")
    for proc, proc_vals in all_improvements.items():
        x, y = [], []
        for metric, metric_vals in proc_vals.items():
            dropped = [val for val in metric_vals if not np.isinf(val)]
            y = y + dropped
            x = x + [lookup_meta(meta, metric, key='shortname', subdict='properties')] * len(dropped)
        fig.add_trace(go.Box(x=x, y=y, name=proc, marker_color=env_cols[f'Laptop {proc}']))
    fig.update_layout(
        yaxis_title=tex('Relative consumption [\%]'), boxmode='group',
        width=PLOT_WIDTH*0.5, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        legend=dict(title='Acceleration via', yanchor="top", y=1, xanchor="right", x=0.975, orientation="h", )
    )
    finalize(fig, fname, show)

    fname = print_init('ch3_edge_stars') ###############################################################################
    # fig = make_subplots(rows=len(host_envs), cols=len(MOD_SEL), specs=[[{'type': 'polar'}] * len(MOD_SEL)] * len(host_envs), subplot_titles=MOD_SEL)
    fig = make_subplots(rows=1, cols=len(host_envs), specs=[[{'type': 'polar'}] * len(MOD_SEL)], subplot_titles=[tex(f'{mod} on {e}') for mod, e in MOD_SEL.items()])
    for idx, (mod, host) in enumerate(MOD_SEL.items()):
        for e_idx, env in enumerate(host_envs[host]):
            model = find_sub_db(full_db, environment=env, model=mod).iloc[0].to_dict()
            summary = fill_meta(model, meta)
            trace = create_star_plot(summary, meta['properties'], name=env.split()[1], color=env_cols[env], showlegend=idx==0, return_trace=True)
            fig.add_trace(trace, row=1, col=idx+1)
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)), width=PLOT_WIDTH, height=PLOT_HEIGHT,
        legend=dict( yanchor="bottom", y=-0.25, xanchor="center", x=0.5, orientation='h'), margin={'l': 50, 'r': 50, 'b': 0, 't': 40}
    )
    fig.update_annotations(yshift=20)
    finalize(fig, fname, show)

    fname = print_init('ch3_edge_compound') ###############################################################################
    fig = go.Figure(layout={'width': PLOT_WIDTH, 'height': PLOT_HEIGHT*1.5, 'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
                            'yaxis':{'title': r'$\text{Compound score } S(m, E)$'}, 'xaxis': {'range': (-1, len(models)-1)}})
    for env in envs:
        subdb, m_size = find_sub_db(full_db, environment=env), 6 if 'Desktop' in env else 10
        vals = [subdb[subdb['model'] == mod]['compound_index'].iloc[0] if subdb[subdb['model'] == mod].shape[0] > 0 else None for mod in models]
        fig.add_trace(go.Scatter(x=model_names, y=vals, name=env, mode='markers', marker=dict(color=env_cols[env], size=m_size, opacity=.7, symbol=env_symb[env])))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5,
                                   entrywidth=0.3, entrywidthmode='fraction'))
    fig.update_xaxes(tickangle=90)
    finalize(fig, fname, show)

    fname = print_init('ch3_edge_scatter_trades') ###############################################################################
    # scatter plots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    xaxis, yaxis, col = 'Resources', 'Quality', 1
    ax_bounds = [list(idx_bounds.values())[0][f'{ax}_index'].tolist() for ax in [xaxis, yaxis]]
    model_to_display = ['MobileNetV2', 'DenseNet169', 'MobileNet', 'InceptionV3', 'ResNet50V2', 'MobileNetV3Small', 'VGG16', 'Xception']
    data_to_display = full_db[full_db['model'].isin(model_to_display)]
    for p_idx, (host, host_envs) in enumerate(host_envs.items()):
        row = p_idx + 1
        plot_data, axis_names = assemble_scatter_data(host_envs, data_to_display, 'index', xaxis, yaxis, meta, UNIT_FMT)
        traces = create_scatter_graph(plot_data, axis_names, dark_mode=False, display_text=True, marker_width=6, return_traces=True)
        for idx, trace in enumerate(traces):
            if p_idx == 0 and idx > 0:
                traces[idx].name = f'Inference on {trace.legendgroup.split()[1]}'
            else:
                traces[idx].showlegend = False
        fig.add_traces(traces, rows=[row]*len(traces), cols=[col]*len(traces))
        min_x, max_x = np.min([min(data['x']) for data in plot_data.values()]), np.max([max(data['x']) for data in plot_data.values()])
        min_y, max_y = np.min([min(data['y']) for data in plot_data.values()]), np.max([max(data['y']) for data in plot_data.values()])
        diff_x, diff_y = max_x - min_x, max_y - min_y
        fig.update_xaxes(range=[min_x-0.1*diff_x, max_x+0.1*diff_x], showgrid=False, title=tex(axis_names[0]), row=row, col=col)
        fig.update_yaxes(range=[min_y-0.1*diff_y, max_y+0.1*diff_y], showgrid=False, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title=tex(axis_names[1]), row=row, col=col)
        add_rating_background(fig, ax_bounds, True, rowcol=(row, col, p_idx))
        fig.update_yaxes(title_text=r'$\text{{H} - Quality score } S_Q(m, E)$'.replace('{H}', host), range=[0.82, 1.02], row=p_idx+1, col=1)
        x_title = r'$\text{Resource score } S_R(m, E)$' if p_idx == len(host_envs) - 1 else ''
        fig.update_xaxes(title_text=x_title, range=[0.00, 1.02], row=p_idx+1, col=1)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*3, margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5,
                                  entrywidth=0.3, entrywidthmode='fraction'))
    finalize(fig, fname, show)

    ############################### Biases Results (Section 3.3.3) #######################################################

    fname = print_init('ch3_bias_correlation_imagenet') ###############################################################################
    correlations = {}
    task_ds_env_sel = {
        'ImageNetEff22': ('ImageNet Models on A100 x8 [Fis+22]', 'infer', 'imagenet', 'A100 x8 - PyTorch 1.10.2+cu113'),
        'EdgeAccUSB': ('ImageNet Models on Laptop NCS [SFB24]', 'infer', 'imagenet', 'Laptop NCS')
    }
    titles = [title for title, _, _, _ in task_ds_env_sel.values()]
    fig = make_subplots(rows=2, cols=len(DBS), subplot_titles=titles, horizontal_spacing=0.14, vertical_spacing=0.05, row_heights=[0.6, 0.4])
    for idx, (name, (db, meta, _, _, _, _)) in enumerate(DBS.items()):
        correlations[name] = calc_all_correlations(db, 'index')
        corr = correlations[name][task_ds_env_sel[name][1:]]
        prop_names = [lookup_meta(meta, prop, 'shortname', 'properties') for prop in corr.columns]
        fig.add_trace(go.Heatmap(z=corr, x=prop_names, y=prop_names, coloraxis="coloraxis"), row=1, col=1+idx)
        above_diag = corr.values[np.triu_indices(corr.shape[0], k=1)]
        fig.add_trace(go.Violin(x=above_diag, y=[1]*above_diag.size, orientation='h', spanmode='hard',
                                showlegend=False, line={'color': LAMARR_COLORS[1]}), row=2, col=1+idx)
        fig.add_trace(go.Box(x=above_diag, y=[0]*above_diag.size, orientation='h',
                             showlegend=False, marker_color=LAMARR_COLORS[1], boxmean='sd'), row=2, col=1+idx)
        fig.add_annotation(x=np.mean(above_diag), y=0, ay=-40, text=r"$\bar{R}$", row=2, col=1+idx)
        for f in [1, -1]:
            arr_head_x = np.mean(above_diag) + np.std(above_diag) * f
            fig.add_annotation(x=arr_head_x, y=0, ay=40, text=r"$\text{std}(R)$", row=2, col=1+idx)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*2, margin={'l': 0, 'r': 0, 'b': 0, 't': 20},
                      coloraxis={'colorscale': LAM_COL_SCALE, 'colorbar': {'title': 'Correlation'}},
                      xaxis3={'title': r'$\text{Correlation } r(\tilde{\mu}_i, \tilde{\mu}_j, E)$'}, xaxis4={'title': r'$\text{Correlation } r(\tilde{\mu}_i, \tilde{\mu}_j, E)$'}, yaxis3={'visible': False}, yaxis4={'visible': False})
    finalize(fig, fname, show)

    fname = print_init('ch3_bias_correlation_others') ###############################################################################
    other_dbs = {
        'XPCR': [("Train and Test", "hospital_dataset", "Intel i9-13900K"), ("Train and Test", "car_parts_dataset_without_missing_values", "Intel i9-13900K")],
        'MetaQuRe': [("train", "parkinsons", "Intel i7-6700 - Scikit-learn 1.4.0"), ("train", "breast_cancer", "Intel i7-6700 - Scikit-learn 1.4.0")],
        'PWC': [("semi-supervised-video-object-segmentation", "davis-2017-val", "unknown"), ("image-classification", "imagenet", "unknown")],
    }
    subplot_titles = [f'{"_".join(ds.split("_")[:2])} ({db})' for db, sel in other_dbs.items() for (_, ds, _) in sel]
    subplot_titles = np.array(subplot_titles).reshape((3, 2)).transpose().flatten() # bring in correct order
    fig = make_subplots(rows=2, cols=len(other_dbs), horizontal_spacing=0.08, vertical_spacing=0.18, subplot_titles=subplot_titles)
    for col, (db_name, to_display_keys) in enumerate(other_dbs.items()):
        db, meta, _, _, _, _ = load_db(DATABASES[db_name])
        # correlations[db_name] = identify_all_correlations(db, 'index')
        # for index, (key, corr) in enumerate(correlations[db_name].items()):
        #     if index > 50:
        #         break
        #     prop_names = [lookup_meta(meta, prop, 'shortname', 'properties') for prop in corr.columns]
        #     fig = go.Figure(go.Heatmap(z=corr, x=prop_names, y=prop_names, coloraxis="coloraxis"))
        #     fig.update_layout(title="   ".join(key))
        #     fig.write_image(os.path.join(DISS_FIGURES, f'test_{db_name.replace(" ", "_")}_{str(index).zfill(3)}_{"__".join(key).replace(" ", "_")}.pdf'))
        for row, (task, ds, env) in enumerate(to_display_keys):
            corr = calc_correlation(find_sub_db(db, task=task, dataset=ds, environment=env))
            prop_names = [lookup_meta(meta, prop, 'shortname', 'properties') for prop in corr.columns]
            fig.add_trace(go.Heatmap(z=corr, x=prop_names, y=prop_names, coloraxis="coloraxis"), row=1+row, col=1+col)
    fig.update_xaxes(tickangle=90)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*2, margin={'l': 0, 'r': 0, 'b': 0, 't': 20},
                      coloraxis={'colorscale': LAM_COL_SCALE, 'colorbar': {'title': 'Correlation'}})
    finalize(fig, fname, show)

    print(1)


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