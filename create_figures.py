import argparse
import importlib
import os
import time
from itertools import product

# STREP imports
from main import DATABASES
from strep.index_scale import load_database, scale_and_rate, _extract_weights, calc_correlation, calc_all_correlations
from strep.util import lookup_meta, find_sub_db, fill_meta, loopup_task_ds_metrics, prop_dict_to_val, read_json
from strep.elex.util import RATING_COLORS, RATING_COLOR_SCALE, RATING_COLOR_SCALE_REV, rgb_to_rgba, hex_to_alpha
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
LAM_COL_SCALE_REV = make_colorscale([LAMARR_COLORS[4], LAMARR_COLORS[2], LAMARR_COLORS[0]])
LAM_COL_FIVE = sample_colorscale(LAM_COL_SCALE, np.linspace(0, 1, 5))
LAM_COL_TEN = sample_colorscale(LAM_COL_SCALE, np.linspace(1, 0, 10))
LAM_SPEC, LAM_SPEC_TRANSP = LAMARR_COLORS[1], hex_to_alpha(LAMARR_COLORS[0], 0.3)

UNIT_FMT = CustomUnitReformater()

DISS_MATERIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'materials', 'dissertation', 'scripts_and_data')
DISS_FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'materials', 'dissertation', 'figures')

XPCR_DS, XPCR_DS_2, XPCR_PROP = 'hospital_dataset', "car_parts_dataset_without_missing_values", 'MASE'
META_DS, META_DS_2, META_PROP, META_ENV = 'parkinsons', 'breast_cancer', 'accuracy', 'Intel i7-6700 - Scikit-learn 1.4.0'

def load_db(db):
    database, meta = load_database(db)
    return scale_and_rate(database, meta)

def tex(text):
    return r'$\text{' + text + r'}$'

def format_val(value):
    if value >= 1000000:
        return f'{np.round(value/1000000, 1)}e6'
    if value >= 100000:
        return f'{np.round(value/100000, 1)}e5'
    if value >= 10000:
        return f'{np.round(value/10000, 1)}e4'
    if value >= 100:
        return str(np.round(value))[:-2]
    return f'{value:4.2f}'[:4]

def print_init(fname):
    print(f'                 - -- ---  {fname:<20}  --- -- -                 ')
    return fname

def finalize(fig, fname, show):
    fig.update_layout(font_family='Open-Sherif')
    fig.update_annotations(yshift=2) # to adapt tex titles
    if show:
        fig.show()
    fig.write_image(os.path.join(DISS_FIGURES, f"{fname}.pdf"))

def finalize_tex(fname, rows, align):
    TEX_TABLE_GENERAL = r'''
    \begin{tabular}$ALIGN
        \toprule 
        $DATA
        \bottomrule
    \end{tabular}'''
    final_text = TEX_TABLE_GENERAL.replace('$DATA', '\n        '.join(rows))
    final_text = final_text.replace('$ALIGN', align)
    with open(os.path.join(DISS_FIGURES, f"tab_{fname}.tex"), 'w') as outf:
        outf.write(final_text)


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
        correlations[name] = calc_all_correlations(db)
        corr = correlations[name][task_ds_env_sel[name][1:]]
        prop_names = [lookup_meta(meta, prop, 'shortname', 'properties') for prop in corr.columns]
        fig.add_trace(go.Heatmap(z=corr, x=prop_names, y=prop_names, coloraxis="coloraxis"), row=1, col=1+idx)
        above_diag = corr.values[np.triu_indices(corr.shape[0], k=1)]
        fig.add_trace(go.Violin(x=above_diag, y=[1]*above_diag.size, orientation='h', spanmode='hard',
                                showlegend=False, line={'color': LAM_SPEC}), row=2, col=1+idx)
        fig.add_trace(go.Box(x=above_diag, y=[0]*above_diag.size, orientation='h',
                             showlegend=False, marker_color=LAM_SPEC, boxmean='sd'), row=2, col=1+idx)
        fig.add_annotation(x=np.mean(above_diag), y=0, ay=-40, text=r"$\bar{R}$", row=2, col=1+idx)
        for f in [1, -1]:
            arr_head_x = np.mean(above_diag) + np.std(above_diag) * f
            fig.add_annotation(x=arr_head_x, y=0, ay=40, text=r"$\text{std}(R)$", row=2, col=1+idx)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*2, margin={'l': 0, 'r': 0, 'b': 0, 't': 24},
                      coloraxis={'colorscale': LAM_COL_SCALE, 'colorbar': {'title': 'Correlation'}},
                      xaxis3={'title': r'$\text{Correlation } r(\tilde{\mu}_i, \tilde{\mu}_j, E)$'}, xaxis4={'title': r'$\text{Correlation } r(\tilde{\mu}_i, \tilde{\mu}_j, E)$'}, yaxis3={'visible': False}, yaxis4={'visible': False})
    finalize(fig, fname, show)

    fname = print_init('ch3_bias_correlation_others') ###############################################################################
    other_dbs = {
        'XPCR': [("Train and Test", XPCR_DS, "Intel i9-13900K"), ("Train and Test", XPCR_DS_2, "Intel i9-13900K"), ("Train and Test", "m1_monthly_dataset", "Intel i9-13900K")],
        'MetaQuRe': [("train", META_DS, META_ENV), ("train", META_DS_2, META_ENV), ("train", "Titanic", META_ENV)],
        'PWC': [("image-classification", "imagenet", "unknown"), ("3d-human-pose-estimation", "mpi-inf-3dhp", "unknown"), ("3d-human-pose-estimation", "mpi-inf-3dhp", "unknown")] # ("semi-supervised-video-object-segmentation", "davis-2017-val", "unknown"), 
    }
    subplot_titles = [f'{"_".join(ds.split("_")[:2]).replace("_dataset", "")} ({db})' for db, sel in other_dbs.items() for (_, ds, _) in sel]
    subplot_titles[-1] = "CIFAR-100 (RobBench)" # TODO improve titles
    fig = make_subplots(rows=3, cols=len(other_dbs), horizontal_spacing=0.07, vertical_spacing=0.09, subplot_titles=subplot_titles)
    for row, (db_name, to_display_keys) in enumerate(other_dbs.items()):
        db, meta, _, _, _, _ = load_db(DATABASES[db_name])
        correlations[db_name] = calc_all_correlations(db)
        # correlations[db_name] = identify_all_correlations(db, 'index')
        # for index, (key, corr) in enumerate(correlations[db_name].items()):
        #     if index > 50:
        #         break
        #     prop_names = [lookup_meta(meta, prop, 'shortname', 'properties') for prop in corr.columns]
        #     fig = go.Figure(go.Heatmap(z=corr, x=prop_names, y=prop_names, coloraxis="coloraxis"))
        #     fig.update_layout(title="   ".join(key))
        #     fig.write_image(os.path.join(DISS_FIGURES, f'test_{db_name.replace(" ", "_")}_{str(index).zfill(3)}_{"__".join(key).replace(" ", "_")}.pdf'))
        for col, task_ds_env in enumerate(to_display_keys):
            if (row+1) * (col+1) < len(subplot_titles):
                corr = correlations[db_name][task_ds_env]
            else: # use RobustBench for last plot
                db, meta, _, _, _, _ = load_db(DATABASES["RobBench"])
                correlations["RobBench"] = calc_all_correlations(db)
                corr = correlations["RobBench"][("Robustness Test", "cifar100", "Tesla V100 - PyTorch 1.7.1")]
            prop_names = [lookup_meta(meta, prop, 'shortname', 'properties') for prop in corr.columns]
            fig.add_trace(go.Heatmap(z=corr, x=prop_names, y=prop_names, coloraxis="coloraxis"), row=1+row, col=1+col)
    # finalize
    fig.update_xaxes(tickangle=90)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*2.5, margin={'l': 0, 'r': 0, 'b': 0, 't': 24},
                      coloraxis={'colorscale': LAM_COL_SCALE, 'colorbar': {'title': 'Correlation'}})
    finalize(fig, fname, show)

    fname = print_init('ch3_bias_correlation_violins') ###############################################################################
    # assess the correlations for PWC_FULL, based on the pre-calculated index values 
    db = pd.read_pickle("databases/paperswithcode/database_complete_index.pkl")
    rename = {col: f"{col}_index" for col in db.select_dtypes("number").columns}
    db = db.rename(rename, axis=1)
    place_holder = np.full((db.shape[0], len(rename)), fill_value="abc")
    db = pd.concat([db, pd.DataFrame(place_holder, index=db.index, columns=list(rename.keys()))], axis=1)
    correlations["PWC_FULL"] = calc_all_correlations(db, progress_bar=True)
    # plot violins (in correct order)
    fig = go.Figure()
    for db_name in DATABASES.keys():
        if db_name in correlations:
            all_corr_vals = np.concat([ matrix.values[np.triu_indices(matrix.shape[0], k=1)] for matrix in correlations[db_name].values() ])
            fig.add_trace(go.Violin(x=[db_name]*all_corr_vals.size, y=all_corr_vals, spanmode='hard',
                                    box_visible=True, meanline_visible=True,
                                    showlegend=False, line={'color': LAM_SPEC}))
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 24},
                      yaxis={'title': r'$\text{Correlation } r(\tilde{\mu}_i, \tilde{\mu}_j, E)$'})
    finalize(fig, fname, show)


def chapter5(show):

    ######################################## MetaQuRe
    meta_features = pd.read_pickle(os.path.join(os.path.dirname(DATABASES['MetaQuRe']), "meta_features.pkl"))
    ds_embeddings = pd.read_pickle(os.path.join(os.path.dirname(DATABASES['MetaQuRe']), "meta_features_embeddings.pkl"))
    db, meta_info, _, idx_bounds, _, _ = load_db(DATABASES['MetaQuRe'])
    baselines, _ = load_database(os.path.join(os.path.dirname(DATABASES['MetaQuRe']), "baselines.pkl"))
    db['environment'] = db['environment'].map(lambda v: v.split(' - ')[0].replace(" rev 1 (v8l)", ""))
    baselines['environment'] = baselines['environment'].map(lambda v: v.split(' - ')[0].replace(" rev 1 (v8l)", ""))
    meta_results = pd.read_pickle(os.path.join(os.path.dirname(DATABASES['MetaQuRe']), "meta_learned.pkl"))
    meta_errors = pd.read_pickle(os.path.join(os.path.dirname(DATABASES['MetaQuRe']), "meta_learned_errors.pkl"))
    env_cols = {env: LAM_COL_FIVE[4-idx] for idx, env in enumerate(['Intel i9-13900K', 'Intel i7-6700', 'Intel i7-10610U', 'ARMv8'])}
    model_colors = {mod:col for mod, col in zip(pd.unique(db['model']), LAM_COL_TEN)}
    objectives = list(zip(['accuracy', 'train_power_draw', 'compound_index'], [r'$\Omega_\text{ACC}$', r'$\Omega_\text{ENT}$', r'$\Omega_\text{QR}$']))
    meta_results[['dataset', 'environment', 'model']] = db[['dataset', 'environment', 'model']]
    META_PNAME = lookup_meta(meta_info, META_PROP, "shortname", "properties")

    fname = print_init('ch5_metaqure_scatter_with_stars') ###############################################################################
    xaxis, yaxis, envs, t = "train_power_draw", META_PROP, [META_ENV.split(' - ')[0], 'ARMv8'], db["task"].iloc[0]
    ax_bounds = [idx_bounds[(t, META_DS, META_ENV)][xaxis].tolist(), idx_bounds[(t, META_DS, META_ENV)][yaxis].tolist()]
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'scatter'}, {'type': 'polar'}], [{'type': 'scatter'}, {'type': 'polar'}]],
                        column_widths=[0.6, 0.4], horizontal_spacing=.05, vertical_spacing=.1, shared_xaxes=True,
                        subplot_titles=[r"$\text{Model performance on }\texttt{" + META_DS + r"}$", "", r"$\text{Model performance on }\texttt{" + META_DS_2 + r"}$", ""])
    for r_idx, ds in enumerate([META_DS, META_DS_2]):
        data, meta_data = db[db["dataset"] == ds], meta_results[meta_results['dataset'] == ds]
        # add scatter plot
        plot_data, axis_names = assemble_scatter_data(envs, data, 'index', xaxis, yaxis, meta_info, UNIT_FMT)
        traces = create_scatter_graph(plot_data, axis_names, dark_mode=False, display_text=True, marker_width=8, return_traces=True)
        for trace in traces:
            if r_idx > 0:
                trace["showlegend"] = False
            fig.add_trace(trace, row=r_idx+1, col=1)
        min_x, max_x = np.min([min(data['x']) for data in plot_data.values()]), np.max([max(data['x']) for data in plot_data.values()])
        min_y, max_y = np.min([min(data['y']) for data in plot_data.values()]), np.max([max(data['y']) for data in plot_data.values()])
        diff_x, diff_y = max_x - min_x, max_y - min_y
        x_title = r'$\hat{\mu}_{\text{ENT}}$' if r_idx==1 else ""
        fig.update_xaxes(range=[min_x-0.1*diff_x, max_x+0.1*diff_x], showgrid=False, title=x_title, row=r_idx+1, col=1)
        fig.update_yaxes(range=[min_y-0.1*diff_y, max_y+0.1*diff_y], showgrid=False, title=r'$\hat{\mu}_{\text{' + META_PNAME + '}}$', row=r_idx+1, col=1)
        add_rating_background(fig, ax_bounds, use_grad=True, rowcol=(r_idx+1, 1, r_idx))
        # add star plot
        for col_idx, (mod, env) in enumerate(product(['kNN', 'GNB'], envs)):
            model = find_sub_db(data, environment=env, model=mod).iloc[0].to_dict()
            summary = fill_meta(model, meta_info)
            trace = create_star_plot(summary, meta_info['properties'], name=f'{mod} on {env}', color=LAM_COL_FIVE[col_idx], showlegend=r_idx==0, return_trace=True)
            fig.add_trace(trace, row=r_idx+1, col=2)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*2, margin={'l': 0, 'r': 0, 'b': 15, 't': 24},
                      legend=dict( title="Selected models", yanchor="middle", y=0.5, xanchor="center", x=0.565))
    fig.update_traces(textposition='top center')
    finalize(fig, fname, show)

    fname = print_init('ch5_metaqure_ds_embeddings') ###############################################################################
    ft_names = {'statistical': 'Manual', 'pca': 'PCA', 'ds2vec': 'DS2VEC', 'combined': 'Joined'}
    ft_sizes = {key: meta_features.loc[:,key].shape[1] if key != "combined" else None for key in ft_names.keys()}
    ft_sizes["combined"] = ft_sizes["statistical"] + ft_sizes["ds2vec"]
    plot_titles=[r'$\text{' + name + r' }|X_D|=' + str(ft_sizes[key]) + r'$' for key, name in ft_names.items()]
    fig = make_subplots(rows=2, cols=4, subplot_titles=plot_titles, horizontal_spacing=0.01, vertical_spacing=0.1)
    for idx, (key, name) in enumerate( ft_names.items() ):
        # add scatter
        embedding = ds_embeddings.loc[:,key]
        min_x, max_x, min_y, max_y = embedding["x"].min(), embedding["x"].max(), embedding["y"].min(), embedding["y"].max()
        fig.update_xaxes(range=[min_x-0.5, max_x+0.5], showticklabels=False, row=1, col=idx+1)
        fig.update_yaxes(range=[min_y-0.5, max_y+0.5], showticklabels=False, row=1, col=idx+1)
        colors, sizes = zip(*[(meta_features.loc[ds,('statistical','n_predictors')], meta_features.loc[ds,('statistical','n_instances')]) for ds in embedding.index])
        fig.add_trace( go.Scatter(x=embedding["x"], y=embedding["y"], mode='markers', showlegend=False, marker={'color': np.log(colors), 'size': np.log(sizes), 'coloraxis': 'coloraxis', 'sizemin': 1}), row=1, col=idx+1)
        # add bars for objective errors
        pred_error_mean = [meta_errors[(key, "index", f'{col}_test_err')].abs().mean() for col, _, in objectives]
        fig.add_trace(go.Bar(x=list(zip(*objectives))[1], y=pred_error_mean, text=[f'{v:4.3f}' for v in pred_error_mean], textposition='auto', marker_color=LAM_COL_FIVE[0], showlegend=False), row=2, col=idx+1)
        fig.update_yaxes(range=[0, 0.18], showticklabels=idx==0, row=2, col=idx+1)
    fig.update_yaxes(title=r'$\text{MAE}_\mathfrak{D}(S_\Omega)$', row=2, col=1)
    # add traces for the scatter size legend
    for idx, n in enumerate([int(min(list(sizes))), 500, 5000, int(max(list(sizes)))]):
        fig.add_trace( go.Scatter(x=[-100], y=[-100], mode='markers', marker={'color': [1], 'size': [np.log(n)], 'colorscale': LAM_COL_SCALE, 'sizemin':1}, name=n), row=1, col=1)
    bar_ticks = [int(min(list(colors))), 10, 100, 1000, int(max(list(colors)))]
    fig.update_layout(
        coloraxis={'colorscale': LAM_COL_SCALE, 'colorbar': {'title': '# Features', 'len': 0.55, 'xanchor': 'right', 'x': 0.01, 'y': 0.8, 'tickvals': np.log(bar_ticks), 'ticktext': bar_ticks}},
        legend=dict(title='Number of rows in the dataset', y=0.5, x=0.5, xanchor='center', yanchor='middle', orientation='h'),
        width=PLOT_WIDTH, height=PLOT_HEIGHT*1.5, margin={'l': 0, 'r': 0, 'b': 0, 't': 24}
    )
    finalize(fig, fname, show)

    fname = print_init('ch5_metaqure_prop_errors') ###############################################################################
    traces, titles = [], []
    for idx, (prop, prop_meta) in enumerate(meta_info['properties'].items()):
        row, col = 2 if idx >= len(meta_info['properties']) / 2 else 1, int(idx % (len(meta_info['properties']) / 2)) + 1
        for e_idx, (scale, trace, color) in enumerate(zip(['recalc_value', 'value'], ['Index', 'Value'], [LAM_COL_FIVE[0], LAM_SPEC])):
            res = meta_results[(scale, f'{prop}_test_err')]
            if e_idx == 0: # use same target unit for both scales!
                _, to_unit = UNIT_FMT.reformat_value(res.iloc[0], prop_meta['unit'])
            reformatted = res.abs().map(lambda val: UNIT_FMT.reformat_value(val, prop_meta['unit'], unit_to=to_unit, as_str=False))
            traces.append( (row, col, go.Box(name=trace, y=reformatted, legendgroup=trace, showlegend=idx==0, marker_color=color)) )
            if e_idx == 0:
                titles.append(f"{prop_meta['shortname']} {to_unit}")
    fig = make_subplots(rows=2, cols=int(len(meta_info['properties']) / 2), y_title=r'$\text{MAE}_\mathfrak{D}(\mu)$', subplot_titles=titles, vertical_spacing=0.12, horizontal_spacing=0.05)
    for row, col, trace in traces:
        fig.add_trace(trace, row=row, col=col)
        fig.update_xaxes(visible=False, showticklabels=False, row=row, col=col)
        if row==2:
            fig.update_yaxes(type="log", row=row, col=col) 
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*1.3, margin={'l': 57, 'r': 0, 'b': 0, 't': 24},
                      legend=dict(title='Meta-learning from values on scale:', orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5))
    finalize(fig, fname, show)

    fname = print_init('ch5_metaqure_optimal_model_choice') ###############################################################################
    fig = make_subplots(rows=len(objectives), cols=len(env_cols), shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.01, vertical_spacing=0.01, subplot_titles=list(env_cols.keys()))
    for row_idx, (sort_col, text) in enumerate(objectives):
        for col_idx, env in enumerate( env_cols.keys() ):
            pred_col, gt_col = ('index', f'{sort_col}_test_pred'), sort_col if "index" in sort_col else f'{sort_col}_index'
            groundtruth = db[db['environment'] == env][['dataset','model', gt_col]]
            predicted = meta_results.loc[groundtruth.index, pred_col]
            gt_and_pred = pd.concat([groundtruth, predicted], axis=1)
            true_best = gt_and_pred.sort_values(['dataset', gt_col], ascending=False).groupby('dataset').first()['model'].values
            pred_best = gt_and_pred.sort_values(['dataset', pred_col], ascending=False).groupby('dataset').first()['model'].values
            for bar_idx, (models, name) in enumerate(zip([true_best, pred_best], ['True best (exhaustive search)', 'Estimated best (via compositional meta-learning)'])):
                col = LAM_COL_FIVE[1] if bar_idx == 0 else LAM_COL_FIVE[0]
                mods, counts = np.unique(models, return_counts=True)
                all_mod_counts = {mod: 0 for mod in model_colors.keys()}
                for mod, cnt in zip(mods, counts):
                    all_mod_counts[mod] = cnt
                fig.add_trace(go.Bar(x=list(all_mod_counts.keys()), y=list(all_mod_counts.values()), marker_color=col, name=name, showlegend=row_idx+col_idx==0), row=row_idx+1, col=col_idx+1)
        fig.update_yaxes(title=text, row=row_idx+1, col=1)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*1.8, margin={'l': 0, 'r': 0, 'b': 0, 't': 24},
                      legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5))
    finalize(fig, fname, show)

    fname = print_init('ch5_metaqure_baseline_comparisons') ###############################################################################
    pfn_ds = pd.unique(baselines[baselines['model'] == 'PFN']['dataset'])
    meta_results[['dataset', 'environment', 'model']] = db[['dataset', 'environment', 'model']]
    # instead of per batch inference energy, use the inference energy for the complete test split for this comparison
    dataset_split_sizes = read_json(os.path.join(os.path.dirname(DATABASES['MetaQuRe']), "dataset_split_sizes.json"))
    for db_ in [baselines, db]:
        db_["test_size"] = db_["dataset"].map(lambda ds: int(dataset_split_sizes[ds]["test"][0]))
        db_["power_draw"] = db_["power_draw"] * db_["test_size"]
    comparison_data = {
        'Small datasets (71)':  ( meta_results[meta_results['dataset'].isin(pfn_ds)], baselines[baselines['dataset'].isin(pfn_ds)], db[db['dataset'].isin(pfn_ds)] ),
        'Large datasets (129)': ( meta_results[~meta_results['dataset'].isin(pfn_ds)], baselines[~baselines['dataset'].isin(pfn_ds)], db[~db['dataset'].isin(pfn_ds)] )
    }
    fig = make_subplots(rows=2, cols=2, shared_yaxes=True, horizontal_spacing=0.01, vertical_spacing=0.01, shared_xaxes=True, row_titles=list(comparison_data.keys()))
    for row_idx, (meta_res, bl_res, exhau_res) in enumerate(comparison_data.values()):
        baseline_results = {mod: {'ene': [], 'acc': [], 'env': []} for mod in ['CML', 'EXH'] + list(pd.unique(bl_res['model']))}
        for env, mod in product(reversed(env_cols.keys()), baseline_results.keys()):
            if mod == 'CML':
                # access results of our method
                sub_pred = meta_res[meta_res['environment'] == env]
                rec_models = sub_pred.sort_values(['dataset', ('index', 'accuracy_test_pred')], ascending=False).groupby('dataset').first()['model']
                data = pd.concat([db[(db['environment'] == env) & (db['dataset'] == ds) & (db['model'] == mod)] for ds, mod in rec_models.items()])
            elif mod == 'EXH':
                # access results of via exhaustive search
                sub_db = exhau_res[(exhau_res['environment'] == env)]
                data = sub_db.sort_values(['dataset','accuracy'], ascending=False).groupby('dataset').first()
                data['power_draw'] = sub_db.groupby('dataset')['power_draw'].sum()
                data['train_power_draw'] = sub_db.groupby('dataset')['train_power_draw'].sum()
            else:
                data = bl_res.loc[(bl_res['model'] == mod) & (bl_res['environment'] == env),['train_power_draw', 'power_draw', 'accuracy']].dropna()
                if data.size < 1:
                    continue
            baseline_results[mod]['ene'] = baseline_results[mod]['ene'] + data[['train_power_draw', 'power_draw']].sum(axis=1).values.tolist()
            baseline_results[mod]['acc'] = baseline_results[mod]['acc'] + data['accuracy'].values.tolist()
            baseline_results[mod]['env'] = baseline_results[mod]['env'] + [env] * data.shape[0]
        for idx, (mod, results) in enumerate( baseline_results.items() ):
            fig.add_trace(go.Box(x=results['acc'], y=results['env'], offsetgroup=f'{mod}{mod}', name=mod, legendgroup=mod, marker_color=LAM_COL_FIVE[idx], showlegend=False), row=1+row_idx, col=1)
            fig.add_trace(go.Box(x=results['ene'], y=results['env'], offsetgroup=f'{mod}{mod}', name=mod, legendgroup=mod, marker_color=LAM_COL_FIVE[idx], showlegend=row_idx==0), row=1+row_idx, col=2)
    fig.update_layout(boxmode='group', width=PLOT_WIDTH, height=PLOT_HEIGHT*2.5, margin={'l': 0, 'r': 15, 'b': 46, 't': 0},
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5))
    fig.update_traces(orientation='h')
    fig.update_xaxes(type="log", title='', row=1, col=2)
    fig.update_xaxes(title='Accuracy [%]', row=2, col=1)
    fig.update_xaxes(type="log", title='Energy draw [Ws]', row=2, col=2)
    finalize(fig, fname, show)

    ######################################## XPCR ########################################
    database, meta, _, idx_bounds, _, _ = load_db(DATABASES['XPCR_FULL'])
    meta_learned_db = pd.read_pickle(DATABASES['XPCR_FULL'].replace(".pkl", "_meta.pkl"))
    get_ds_short = lambda ds_name: ds_name[:4] + '..' + ds_name[-3:] if len(ds_name) > 9 else ds_name

    monash = pd.read_csv(os.path.join(DISS_MATERIAL, 'ch5_monash.csv'), delimiter=';', index_col='Dataset').replace('-', np.nan).astype(float)
    dnns = sorted(pd.unique(meta_learned_db['model']).tolist())
    pred_cols = list(meta['properties'].keys())

    dnn_map = {meta['model'][mod]['name']: (mod, meta['model'][mod]['short']) for mod in dnns}
    mod_map = {meta['model'][mod]['name']: (mod, meta['model'][mod]['short']) for mod in pd.unique(database['model'])}
    ds_overlap = list(reversed([ds for ds in pd.unique(database['dataset']) if ds in monash.index]))
    ds_short = [get_ds_short(meta['dataset'][ds]['name']) for ds in ds_overlap]

    fname = print_init('ch5_xpcr_method_comparison') ###############################################################################
    rows = [
        ' & '.join(['Dataset'] + [r'\multicolumn{3}{c}{$' + mod + r'$}' for mod in [r'\text{CML + } \Omega_\text{PCR}', r'\text{CML + } \Omega_\text{MASE}', r'\texttt{AutoGluonTS}', r'\texttt{AutoKeras}', r'\texttt{AutoSklearn}']]) + r' \\',
        ' & '.join([ ' ' ] + [r'$S$', r'$\tilde{\mu}_{\text{MASE}}$', r'$\mu_{\text{ENT}}$'] * 5) + r' \\',
        r'\midrule',
    ]
    for ds, data in meta_learned_db.groupby('dataset'):
        if data['dataset_orig'].iloc[0] == ds:
            row = [ get_ds_short(meta['dataset'][ds]['name']) ]

            # best XPCR
            xpcr_rec_config = data.sort_values(('index', 'compound_index_test_pred'), ascending=False).iloc[0]
            xpcr = database[(database['dataset'] == xpcr_rec_config['dataset']) & (database['model'] == xpcr_rec_config['model'])].iloc[0]
            values = [ (xpcr['compound_index'], xpcr[f"{XPCR_PROP}_index"], xpcr['train_power_draw'] / 3.6e3) ]

            # AutoForecast / simulates metsa-learned selection based on best estimated error
            aufo_rec_config = data.sort_values(('index', f'{XPCR_PROP}_test_pred'), ascending=False).iloc[0]
            aufo = database[(database['dataset'] == aufo_rec_config['dataset']) & (database['model'] == aufo_rec_config['model'])].iloc[0]
            values.append( (aufo['compound_index'], aufo[f"{XPCR_PROP}_index"], aufo['train_power_draw'] / 3.6e3) )
    
            # autokeras & autosklearn
            for auto in ['autogluon', 'autokeras', 'autosklearn']:
                try:
                    auto_res = database[(database['dataset'] == ds) & (database['model'] == auto)].iloc[0]
                    qual = auto_res[f"{XPCR_PROP}_index"]
                    powr = auto_res['train_power_draw'] / 3.6e3
                except:
                    qual, powr = 0, 0
                values.append( (auto_res['compound_index'], qual, powr) )
            
            # bold print best error
            best_idx = np.max([val[0] for val in values])
            best_err = np.max([val[1] for val in values])
            best_ene = np.min([val[2] for val in values])
            for idx, results in enumerate(values):
                format_results = [ format_val(res) for res in results ]
                format_results[0] = f'{results[0]:3.2f}'
                format_results[1] = f'{results[1]:3.2f}'
                for val_idx, best in enumerate([best_idx, best_err, best_ene]):
                    if (val_idx < 2 and results[val_idx] == best) or (val_idx == 2 and results[val_idx] == best):
                        format_results[val_idx] = r'\textbf{' + format_results[val_idx] + r'}'
                row += format_results
            rows.append(' & '.join(row) + r' \\')
    finalize_tex(fname, rows, r'{l|ccc|ccc|ccc|ccc|ccc}')

    fname = print_init('ch5_xpcr_model_perf_monash') ###############################################################################
    for name, (code_name, short) in mod_map.items():
        for ds in ds_overlap:
            subdb = database[(database['model'] == code_name) & (database['dataset'] == ds)]
            monash.loc[ds,f'{name}_mase'] = subdb['MASE'].iloc[0]
            monash.loc[ds,f'{name}_compound'] = subdb['compound_index'].iloc[0]
            if name in monash.columns:
                monash.loc[ds, f'{name}_mase_diff'] = np.abs(monash.loc[ds,f'{name}_mase'] - monash.loc[ds,name])
    min_max = {}
    def custom_scaling(x): # Sigmoid-like scaling to compress larger values for MASE
        return x / (1 + x)
    all_rel_cols = [[col for col in monash.columns if '_mase' in col and 'diff' not in col], [col for col in monash.columns if '_compound' in col], [col for col in monash.columns if '_mase_diff' in col]]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.07,
                        subplot_titles=([r'$\mu_{\text{MASE}}(m, C)$', r'$S_{\Omega_\text{PCR}}(m, C)$', tex('Diff to Monash MASE')]))
    for idx, rel_cols in enumerate(all_rel_cols):
        sub_monash = monash.loc[ds_overlap,rel_cols]
        colorbar = {'x': 0.356*(idx+1)-0.078, "thickness": 15}
        if idx==0:
            sub_monash = custom_scaling(sub_monash)
            colorbar.update( dict(tick0=0, tickmode= 'array', tickvals=custom_scaling(np.array([0.6, 1, 2, 10, 1000])), ticktext=["0.6", "1", "2", "10", ">1e3"]) )
        min_max[idx] = sub_monash.values.min(), sub_monash.values.max()
        fig.add_trace(go.Heatmap(z=sub_monash.values, x=[mod_map[mod.split('_')[0]][1] for mod in sub_monash], y=ds_short, colorscale=LAM_COL_SCALE, reversescale=idx==1, colorbar=colorbar, zmin=min_max[idx][0], zmax=min_max[idx][1]), row=1, col=idx+1)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*1.2, margin={'l': 0, 'r': 0, 'b': 0, 't': 24})
    finalize(fig, fname, show)

    fname = print_init('ch5_xpcr_cml_performance') ###############################################################################
    for name, (code_name, short) in dnn_map.items():
        for ds in ds_overlap:
            sub_meta = meta_learned_db[(meta_learned_db['model'] == code_name) & (meta_learned_db['dataset'] == ds)]
            monash.loc[ds,f'{name}_est_mase'] = sub_meta[('value', 'MASE_test_pred')].iloc[0]
            monash.loc[ds,f'{name}_est_compound'] = sub_meta[('index', 'compound_index_test_pred')].iloc[0]
            monash.loc[ds,f'{name}_est_compound_err'] = sub_meta[('index', 'compound_index_test_err')].iloc[0]
    all_rel_cols2 = [[c for c in monash.columns if c.endswith('_est_mase')], [c for c in monash.columns if c.endswith('_est_compound')], [c for c in monash.columns if c.endswith('_est_compound_err')]]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.07, 
                        subplot_titles=([r'$\hat{\mu}_{\text{' + XPCR_PROP + r'}}(m, X_C)$', r'$\hat{S}_{\Omega_\text{PCR}}(m, X_C)$', r'$\text{MAE}_\mathfrak{D}(\tilde{\mu}_{\text{' + XPCR_PROP + '}})$']))
    for idx, rel_cols in enumerate(all_rel_cols2):
        sub_monash = monash.loc[ds_overlap,rel_cols]
        colorbar = {'x': 0.356*(idx+1)-0.078, "thickness": 15}
        if idx==0:
            sub_monash = custom_scaling(sub_monash)
            colorbar.update( dict(tick0=0, tickmode='array', tickvals=custom_scaling(np.array([0.6, 1, 2, 10, 1000])), ticktext=["0.6", "1", "2", "10", ">1e3"]) )
            zmin, zmax = min_max[idx]
        else:
            zmin, zmax = sub_monash.values.min(), sub_monash.values.max()
        fig.add_trace(go.Heatmap(z=sub_monash.values, x=[mod_map[mod.split('_')[0]][1] for mod in sub_monash], y=ds_short, colorscale=LAM_COL_SCALE, reversescale=idx>0, colorbar=colorbar, zmin=zmin, zmax=zmax), row=1, col=idx+1)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*1.2, margin={'l': 0, 'r': 0, 'b': 0, 't': 24})
    finalize(fig, fname, show)

    # access statistics per data set
    fname = print_init('ch5_xpcr_scatter_with_stars') ###############################################################################
    xaxis, yaxis, env, t = "train_power_draw", XPCR_PROP, database["environment"].iloc[0], database["task"].iloc[0]
    ax_bounds = [idx_bounds[(t, ds, env)][xaxis].tolist(), idx_bounds[(t, ds, env)][yaxis].tolist()]
    top_increasing_k_stats = {}
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'scatter'}, {'type': 'polar'}], [{'type': 'scatter'}, {'type': 'polar'}]],
                        column_widths=[0.6, 0.4], horizontal_spacing=.05, vertical_spacing=.1, shared_xaxes=True,
                        subplot_titles=[r"$\text{Model performance on }\texttt{" + lookup_meta(meta, XPCR_DS, subdict='dataset') + r"}$", "", r"$\text{Model performance on }\texttt{" + lookup_meta(meta, XPCR_DS_2, subdict='dataset') + r"}$", ""])
    for ds, data in database.groupby('dataset'):
        meta_data = meta_learned_db[meta_learned_db['dataset'] == ds]
        sorted_by_pred_err = meta_data.sort_values(('index', f'{XPCR_PROP}_test_pred'), ascending=False)
        only_model_pool = data[data["model"].isin(sorted_by_pred_err["model"])]
        lowest_err = max(only_model_pool['compound_index'])
        lowest_ene = sum(only_model_pool[xaxis])
        # save ene and err for increasing k later
        if not np.isinf(lowest_err) and not np.isinf(lowest_ene):
            top_increasing_k_stats[ds] = {'err': [], 'ene': []}
            for k in range(1, meta_data.shape[0] + 1):
                model_results = data[data["model"].isin(sorted_by_pred_err.iloc[:k]['model'].values)]
                top_increasing_k_stats[ds]['err'].append( max(model_results["compound_index"] / lowest_err))
                top_increasing_k_stats[ds]['ene'].append( sum(model_results[xaxis]) / lowest_ene )
        if ds in [XPCR_DS, XPCR_DS_2]:
            r_idx = 1 if ds == XPCR_DS else 2
            pred_afo = meta_data.sort_values(('index', f'{XPCR_PROP}_test_pred'), ascending=False).iloc[0]['model']
            pred_xpcr = meta_data.sort_values(('index', 'compound_index_test_pred'), ascending=False).iloc[0]['model']
            pred_afo_short, pred_xpcr_short = lookup_meta(meta, pred_afo, "short", "model"), lookup_meta(meta, pred_xpcr, "short", "model")
            # add scatter plot
            plot_data, axis_names = assemble_scatter_data([env], data, 'index', xaxis, yaxis, meta, UNIT_FMT)
            for mod, add_annot in [(pred_afo_short, r"({\Omega_\text{MASE}})$"), (pred_xpcr_short, r"(\Omega_P)$")]:
                to_change = plot_data[env]["names"].index(mod)
                plot_data[env]["names"][to_change] = r'$\text{' + plot_data[env]["names"][to_change] + r' }' + add_annot
            trace = create_scatter_graph(plot_data, axis_names, dark_mode=False, display_text=True, marker_width=8, return_traces=True)[0]
            trace["showlegend"] = False
            fig.add_trace(trace, row=r_idx, col=1)
            min_x, max_x = np.min([min(data['x']) for data in plot_data.values()]), np.max([max(data['x']) for data in plot_data.values()])
            min_y, max_y = np.min([min(data['y']) for data in plot_data.values()]), np.max([max(data['y']) for data in plot_data.values()])
            diff_x, diff_y = max_x - min_x, max_y - min_y
            x_title = r'$\hat{\mu}_{\text{ENT}}$' if r_idx==2 else ""
            fig.update_xaxes(range=[min_x-0.1*diff_x, max_x+0.1*diff_x], showgrid=False, title=x_title, row=r_idx, col=1)
            fig.update_yaxes(range=[min_y-0.1*diff_y, max_y+0.1*diff_y], showgrid=False, title=r'$\hat{\mu}_{\text{' + XPCR_PROP + '}}$', row=r_idx, col=1)
            add_rating_background(fig, ax_bounds, use_grad=True, rowcol=(r_idx, 1, 0 if r_idx==1 else 1))
            # add star plot
            for model, m_str, col_idx in zip([pred_xpcr, pred_afo, 'autogluon'], [r'$\texttt{CML}\text{ with }\Omega_\text{PCR}$', r'$\texttt{CML}\text{ with }\Omega_\text{MASE}$', r'$\texttt{AGl}$'], [0, 2, 4]):
                model = find_sub_db(data, model=model).iloc[0].to_dict()
                summary = fill_meta(model, meta)
                trace = create_star_plot(summary, meta['properties'], name=m_str, color=LAM_COL_FIVE[col_idx], showlegend=r_idx==1, return_trace=True)
                fig.add_trace(trace, row=r_idx, col=2)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT*2, margin={'l': 0, 'r': 0, 'b': 15, 't': 24},
                      legend=dict( title="Selected models", yanchor="middle", y=0.5, xanchor="center", x=0.6))
    fig.update_traces(textposition='top center')
    finalize(fig, fname, show)

    fname = print_init('ch5_xpcr_top_k') ###############################################################################
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=[r'$\text{Best }S_{\Omega_\text{PCR}}(m, C)$', tex('Total energy draw')], horizontal_spacing=0.05)
    for ds, values in top_increasing_k_stats.items():
        err = np.array(values['err']) * 100
        ene = np.array(values['ene']) * 100
        k = np.arange(1, len(err) + 1)
        fig.add_trace(go.Scatter(x=k, y=err, mode='lines', line=dict(color=LAM_SPEC_TRANSP)), row=1, col=1)
        fig.add_trace(go.Scatter(x=k, y=ene, mode='lines', line=dict(color=LAM_SPEC_TRANSP)), row=1, col=2)
    avg_err = np.array([np.array(val['err']) for val in top_increasing_k_stats.values()]).mean(axis=0) * 100
    avg_ene = np.array([np.array(val['ene']) for val in top_increasing_k_stats.values()]).mean(axis=0) * 100
    fig.add_trace( go.Scatter(x=k, y=avg_err, mode='lines', line=dict(color='rgba(0,0,0,1.0)')), row=1, col=1)
    fig.add_trace( go.Scatter(x=k, y=avg_ene, mode='lines', line=dict(color='rgba(0,0,0,1.0)')), row=1, col=2)
    fig.update_layout( width=PLOT_WIDTH, height=PLOT_HEIGHT, showlegend=False, margin={'l': 0, 'r': 0, 'b': 20, 't': 24} )
    fig.update_yaxes(title='Relative value [%]', row=1, col=1)
    fig.update_xaxes(title='k (testing top-k recommendations)')
    finalize(fig, fname, show)

    fname = print_init('ch5_xpcr_error_estimates') ###############################################################################
    weights = {p: w for p, w in zip(meta["properties"].keys(), _extract_weights(meta["properties"]))}
    pred_cols = ['compound_index', 'compound_index_direct'] + pred_cols
    db_raw, meta = load_database(DATABASES['XPCR_FULL'])
    db_raw = db_raw[~db_raw["model"].isin(["autokeras", "autogluon", "autosklearn"])]
    db_no_comp, meta, _, _, _, _ = scale_and_rate(db_raw, meta)
    stat_str = [r'$\text{MAE}_\mathfrak{D}(\tilde{\mu})$', r'$\text{ACC}_\mathfrak{D}(\tilde{\mu})$', r'$\text{TOP-3 ACC}_\mathfrak{D}(\tilde{\mu})$']
    k_best, result_scores = 3, {s_str: {} for s_str in stat_str}
    # assemble the error stats
    for col in pred_cols:
        for key in result_scores.keys():
            result_scores[key][col] = []
        for _, sub_data in iter(meta_learned_db.groupby(('split_index', ''))):
            top_1, top_k, error = [], [], []
            for ds, data in sub_data.groupby('dataset'):
                sorted_by_true = db_no_comp.loc[data.index].sort_values('compound_index' if 'compound' in col else f'{col}_index', ascending=False)
                sorted_by_pred = data.sort_values(('index', f'{col}_test_pred'), ascending=False)
                top_k_pred = sorted_by_pred.iloc[:k_best]['model'].values
                best_model = sorted_by_true.iloc[0]['model']
                err = np.abs(data[('index', f'{col}_test_err')]).mean()
                top_1.append(best_model == sorted_by_pred.iloc[0]['model'])
                top_k.append(best_model in top_k_pred)
                error.append(err)
            for s_str, val in zip(stat_str, [np.mean(error), np.mean(top_1) * 100, np.mean(top_k) * 100]):
                result_scores[s_str][col].append(val)
    # plot data
    fig = make_subplots(rows=1, cols=len(stat_str), shared_yaxes=True, horizontal_spacing=0.02)
    max_x = []
    meta['properties']['compound_index'] = {'shortname': r'$S_{\Omega_\text{PCR}}\text{ (CML)}$'}
    meta['properties']['compound_index_direct'] = {'shortname': r'$S_{\Omega_\text{PCR}}\text{ (DML)}$'}
    for plot_idx, results in enumerate(result_scores.values()):
        x, y, e, w = zip(*reversed([(np.mean(vals), meta['properties'][key]['shortname'], np.std(vals), weights[key] if key in weights else 0) for key, vals in results.items()]))
        c = sample_colorscale(LAM_COL_SCALE, np.array(w)*3)
        trace = go.Bar(x=x, y=y, error_x=dict(type='data', array=e), orientation='h', marker_color=c)
        fig.add_trace(trace, row=1, col=plot_idx+1)
        max_x.append(max(x) + (max(x) / 10))
        fig.update_xaxes(title=stat_str[plot_idx], row=1, col=plot_idx+1)
    fig.update_layout(width=PLOT_WIDTH, title_y=0.99, title_x=0.5, height=PLOT_HEIGHT, 
                      showlegend=False, margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    finalize(fig, fname, show)

    # why recommendation?
    fname = print_init('ch5_xpcr_explanations') ###############################################################################
    why_data = read_json(os.path.join(DISS_MATERIAL, "ch5_xpcr_why.json"))
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=list(why_data.keys()), horizontal_spacing=0.05)
    for col, data in enumerate(why_data.values()):
        vals = np.array(list(data.values()))
        if col == 1:
            vals *= 100
        fig.add_trace(go.Bar(x=list(data.keys()), y=vals, marker={"color": vals, "colorscale": LAM_COL_SCALE_REV}, showlegend=False), row=1, col=col+1)
    fig.update_xaxes(title='Property', tickangle=90, row=1, col=1)
    fig.update_xaxes(title='Meta-feature', tickangle=90, row=1, col=2)
    fig.update_yaxes(title="Importance [%]", row=1, col=1)
    fig.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 24})
    finalize(fig, fname, show)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", type=int, default=-1)
    parser.add_argument("--show", default=True)
    args = parser.parse_args()

    ####### DUMMY OUTPUT - for setting up pdf export of plotly
    fig = px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")

    chapters = [chapter1, chapter2, chapter3, None, chapter5]
    if args.chapter == -1:
        for i in range(5):
            if chapters[i]:
                chapters[i](args.show)

    if args.chapter < 1 or args.chapter > 5:
        raise ValueError("Chapter number must be between 1 and 5")
    
    ####### print chapter figures
    chapters[args.chapter-1](args.show)
    os.remove("dummy.pdf")