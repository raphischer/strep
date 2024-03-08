import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from strep.util import read_json, lookup_meta, identify_all_correlations, fill_meta
from strep.index_and_rate import prop_dict_to_val
from strep.load_experiment_logs import find_sub_db
from strep.elex.util import ENV_SYMBOLS, RATING_COLORS, RATING_COLOR_SCALE, RATING_COLOR_SCALE_REV
from strep.elex.graphs import assemble_scatter_data, create_scatter_graph, add_rating_background
from strep.labels.label_generation import PropertyLabel


PLOT_WIDTH = 900
PLOT_HEIGHT = PLOT_WIDTH // 3
COLORS = ['#009ee3', '#983082', '#ffbc29', '#35cdb4', '#e82e82', '#59bdf7', '#ec6469', '#706f6f', '#4a4ad8', '#0c122b', '#ffffff']


SEL_DS_TASK = {
    'ImageNetEff': ('imagenet', 'infer'),
    'Forecasting': ('electricity_weekly_dataset', 'Train and Test'),
    'Papers With Code': ('KITTI', 'depth-completion'),
    'RobustBench': ('cifar100', 'Robustness Test'),
}

TEX_TABLE_GENERAL = r'''
    \begin{tabular}$ALIGN
        \toprule 
        $DATA
        \bottomrule
    \end{tabular}'''


# import spacy
# nlp = spacy.load('en_core_web_lg')
# words = ' '.join(database.select_dtypes('number').columns)
# tokens = nlp(words)
# sim_matr = np.ones((database.select_dtypes('number').shape[0], database.select_dtypes('number').shape[0]))
# for x, token1 in enumerate(tokens):
#     for y, token2 in enumerate(tokens):
#         sim_matr[x,y] = token1.similarity(token2)



def create_all(databases):
    databases['ImageNetEff']
    filterstats = read_json('databases/paperswithcode/filterstats.json')
    pwc_stats = read_json('databases/paperswithcode/other_stats.json')
    os.chdir('paper_results')

    # COMPUTE CORRELATIONS
    correlations = {}
    for name, (database, _, metrics, _, _, _, _, _) in databases.items():
        correlations[name] = {  scale: identify_all_correlations(database, metrics, scale) for scale in ['index', 'value'] }

    # database stat table
    rows = [np.array(['Database', 'Data sets', 'Tasks', 'Methods', 'Environments', 'Properties (resources)', 'Evaluations', 'Incompleteness'])] # \\' + '\n' + r'        \midrule']
    for name, (db, meta, metrics, _, _, _, _, _) in databases.items():
        # identify properties (and the props that describe resources)
        prop_names = list( set().union(*[set(vals) for vals in metrics.values()]) )
        res_prop_names = [prop for prop in prop_names if lookup_meta(meta, prop, 'group', subdict='properties') == 'Resources']
        # assess the nan amounts for each ds / task combo
        nan_amounts = []
        for key, subdb in db.groupby(['dataset', 'task']):
            props = prop_dict_to_val(subdb[metrics[key]])
            nan_amounts.append(np.count_nonzero(props.isna()) / props.size)
        row = [name] + [str(pd.unique(db[col]).size) for col in ['dataset', 'task', 'model', 'environment']]
        row = row + [f'{len(prop_names)} ({len(res_prop_names)})', str(db.shape[0]), f'{np.mean(nan_amounts) * 100:5.2f} ' + r'\%']
        rows.append(np.array(row))
    tex_rows = []
    for row in np.array(rows).transpose():
        tex_rows.append(' & '.join(row) + r' \\')
    tex_rows.insert(1, r'\midrule')
    final_text = TEX_TABLE_GENERAL.replace('$DATA', '\n        '.join(tex_rows))
    final_text = final_text.replace('$ALIGN', r'{lccccccc}')
    with open('databases.tex', 'w') as outf:
        outf.write(final_text)

    ####### DUMMY OUTPUT ####### for setting up pdf export of plotly
    fig=px.scatter(x=[0, 1, 2], y=[0, 1, 4])
    fig.write_image("dummy.pdf")
    time.sleep(0.5)
    os.remove("dummy.pdf")

    # labels
    labels_to_generate = {
        'Forecasting': lambda db: db[(db['model'] == 'feedforward') & (db['dataset'] == 'electricity_weekly_dataset')],
        'ImageNetEff': lambda db: db[(db['model'] == 'MobileNetV3Small') & (db['environment'] == 'A100 x8 - TensorFlow 2.8.0') & (db['task'] == 'infer')],
        'RobustBench': lambda db: db[(db['model'] == 'Addepalli2022Efficient_WRN_34_10')],
        'Papers With Code': lambda db: db[(db['task'] == 'depth-completion') & (db['methodology'] == 'KBNet')]
    }
    for name, mod_extraction in labels_to_generate.items():
        db, meta, _, _, _, _, _, _ = databases[name]
        model = fill_meta(mod_extraction(db).iloc[0].to_dict(), meta)
        # brush up the aesthetic
        if name == 'ImageNetEff':
            model['task'] = 'Inference'
        if name == 'RobustBench':
            model['model']['name'] = model['model']['name'].replace('NeurIPS 2022', 'NeurIPS 22')
        if name == 'Papers With Code':
            model['model'] = {'name': 'KBNet (Wong & Soatto 21)', 'url': 'https://arxiv.org/abs/2108.10531'}
            model['task'] = 'Depth Completion'
        label = PropertyLabel(model, custom=meta['meta_dir'])
        label.save(f'label_{name.replace(" ", "_")}.pdf')

    # PWC filtering
    fig = go.Figure(layout={
        'width': PLOT_WIDTH / 2, 'height': PLOT_HEIGHT, 'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
        'xaxis':{'title': 'Number of evaluations'}, 'yaxis':{'title': 'Number of properties'}}
    )
    pos = [ 'bottom left', 'middle right', 'middle right', 'middle right' ]
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

    # PWC stats
    pwc_stats = pd.DataFrame(pwc_stats).transpose()
    for _, data in pwc_stats.groupby(['n_results', 'n_metrics']):
        pwc_stats.loc[data.index,'count'] = data.shape[0]
    pwc_stats['log_count'] = np.log(pwc_stats['count'])
    pwc_stats['log_n_results'] = np.log(pwc_stats['n_results'])
    pwc_stats.loc[(pwc_stats['n_results'] == 0),'log_n_results'] = -0.5
    fig = px.scatter(data_frame=pwc_stats, x='log_n_results', y='n_metrics', color='log_count', color_continuous_scale=RATING_COLOR_SCALE)
    fig.update_layout(
        width=PLOT_WIDTH / 2, height=PLOT_HEIGHT, coloraxis_colorbar_title='Cases',
        coloraxis_colorbar_tickvals=[0, np.log(10), np.log(100), np.log(1000)], coloraxis_colorbar_ticktext=[1, 10, 100, 1000],
        xaxis=dict(title='Number of evaluations', tickmode='array', tickvals=[-0.5, 0, np.log(10), np.log(100), np.log(1000)], ticktext=[0, 1, 10, 100, 1000]),
        yaxis={'title': 'Number of properties'}, margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )
    fig.write_image(f'pwc_stats.pdf')



    # PLOTS FOR ALL DATABASES
    for name, (db, meta, metrics, xdef, ydef, bounds, real_bounds, _) in databases.items():
        # scatter plots
        ds, task = SEL_DS_TASK[name]
        ds_name = lookup_meta(meta, ds, subdict='dataset')
        xaxis, yaxis = xdef[(ds, task)], ydef[(ds, task)]
        db = find_sub_db(db, dataset=ds, task=task)
        if name == 'ImageNetEff': # big figure mit value and index scales side by side
            scatter = make_subplots(rows=1, cols=2, horizontal_spacing=.05, subplot_titles=['Real measurements', 'Index scaled values'])
            for idx, (scale, x_title, y_title, plot_bounds) in enumerate(zip(['value', 'index'], ['Power Draw per Inference [Ws]', 'Power Draw per Inference Index'], ['Accuracy [%]', 'Accuracy Index'], [real_bounds, bounds])):
                plot_data, axis_names, rating_pos = assemble_scatter_data(pd.unique(db['environment'])[:2], db, scale, xaxis, yaxis, meta, plot_bounds)
                traces = create_scatter_graph(plot_data, axis_names, dark_mode=False, display_text=False, marker_width=8, return_traces=True)
                scatter.add_traces(traces, rows=[1]*len(traces), cols=[idx+1]*len(traces))
                min_x, max_x = np.min([min(data['x']) for data in plot_data.values()]), np.max([max(data['x']) for data in plot_data.values()])
                min_y, max_y = np.min([min(data['y']) for data in plot_data.values()]), np.max([max(data['y']) for data in plot_data.values()])
                diff_x, diff_y = max_x - min_x, max_y - min_y
                scatter.update_xaxes(range=[min_x-0.1*diff_x, max_x+0.1*diff_x], showgrid=False, title=x_title, row=1, col=idx+1)
                scatter.update_yaxes(range=[min_y-0.1*diff_y, max_y+0.1*diff_y], showgrid=False, title=y_title, row=1, col=idx+1)
                add_rating_background(scatter, rating_pos, 'optimistic mean', dark_mode=False, col=(idx+1))
            for idx in [1, 2]:
                scatter.data[idx]['showlegend'] = False
            scatter.update_yaxes(side='right', row=1, col=2)
            scatter.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 25},
                                  legend=dict(x=.5, y=0.05, orientation="h", xanchor="center", yanchor="bottom"))
            scatter.show()
            scatter.write_image(f"scatter_{name}.pdf")
        else:
            plot_data, axis_names, rating_pos = assemble_scatter_data([db['environment'].iloc[0]], db, 'index', xaxis, yaxis, meta, bounds)
            scatter = create_scatter_graph(plot_data, axis_names, dark_mode=False, display_text=False, marker_width=8)
            rating_pos[0][0][0] = scatter.layout.xaxis.range[1]
            rating_pos[1][0][0] = scatter.layout.yaxis.range[1]
            add_rating_background(scatter, rating_pos, 'optimistic mean', dark_mode=False)
            scatter.update_layout(width=PLOT_WIDTH / 3, height=PLOT_HEIGHT, margin={'l': 0, 'r': 0, 'b': 0, 't': 25}, title_y=0.99, title_x=0.5, title_text=f'{name} - {ds_name}')
            scatter.write_image(f"scatter_{name}.pdf")

        # star plot
        db = prop_dict_to_val(db, 'index')
        worst = db.sort_values('compound_index').iloc[0]
        best = db.sort_values('compound_index').iloc[-1]
        # np.linalg.norm(best[metrics[(ds, task)]] - worst[metrics[(ds, task)]].values)
        fig = go.Figure()
        for model, col, m_str in zip([best, worst], [RATING_COLORS[0], RATING_COLORS[4]], ['Best', 'Worst']):
            mod_name = lookup_meta(meta, model['model'], 'short', 'model')[:18]
            metr_names = [lookup_meta(meta, metr, 'shortname', 'properties') for metr in metrics[(ds, task)]]
            fig.add_trace(go.Scatterpolar(
                r=[model[col] for col in metrics[(ds, task)]], line={'color': col},
                theta=metr_names, fill='toself', name=f'{mod_name} ({m_str}): {model["compound_index"]:4.2f}'
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)), width=PLOT_WIDTH*0.25, height=PLOT_HEIGHT, title_y=0.98, title_x=0.5, title_text=ds_name,
            legend=dict( yanchor="bottom", y=0.78, xanchor="center", x=0.5), margin={'l': 50, 'r': 50, 'b': 0, 't': 0}
        )
        fig.write_image(f'true_best_{name}.pdf')

    # property correlations
    fig_prop_corr = go.Figure()
    fig_pwc_comp_impact = go.Figure()
    for idx, (key, (db, meta, metrics, xdef, ydef, bounds, _, _)) in enumerate(databases.items()):
        # assess property correlation
        for scale, corrs in correlations[key].items():
            all_corr = []
            for _, corr in corrs.items():
                all_corr = all_corr + corr[0].flatten().tolist()
            fig_prop_corr.add_trace( go.Violin(y=all_corr, x=[f'{idx}_{scale.capitalize()}'] * len(all_corr), spanmode='hard', name=f'{key} (N={len(all_corr)})', box_visible=True, meanline_visible=True, legendgroup=key, showlegend=scale=='index', line={'color': RATING_COLORS[idx]}) )
            
        # assess difference of using weighted compound or single property
        corr_spearmen = []
        for ds_task, data in db.groupby(['dataset', 'task']):
            data_val = prop_dict_to_val(data[metrics[ds_task]])
            data_ind = prop_dict_to_val(data[metrics[ds_task]], 'index')
            if np.all(data_ind.min() >= 0) and np.all(data_ind.max() <= 1): # TODO remove this hotfix after final fix of index scaling
                prop_pop = [data_val[col].dropna().size for col in metrics[ds_task]]
                most_pop = data_ind.iloc[:,np.argmax(prop_pop)]
                equally = data_ind.mean(axis=1)
                # corr_ken.append(kendalltau(most_pop.values, equally.values)[0])
                corr_spearmen.append(spearmanr(most_pop.values, equally.values)[0])
        fig_pwc_comp_impact.add_trace( go.Violin(y=corr_spearmen, spanmode='hard', name=f'{key} (N={len(corr_spearmen)})', line={'color': RATING_COLORS[idx]}, box_visible=True, meanline_visible=True) )
    # write images
    fig_prop_corr.update_layout(width=PLOT_WIDTH, height=PLOT_HEIGHT, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), margin={'l': 0, 'r': 0, 'b': 0, 't': 0}, xaxis={'tickmode': 'array', 'tickvals': [0, 1, 2, 3, 4, 5, 6, 7], 'ticktext': ['Index-scaled', 'Original values'] * 4})
    fig_prop_corr.write_image('prop_corr.pdf')
    fig_pwc_comp_impact.update_layout(width=PLOT_WIDTH / 2, height=PLOT_HEIGHT, xaxis={'visible': False, 'showticklabels': False},
                      legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="center", x=0.5), margin={'l': 0, 'r': 0, 'b': 0, 't': 0} )
    fig_pwc_comp_impact.write_image('pwc_comp_impact.pdf')

    # forecasting & imagenet metric correlation
    corr_res = []
    for db_name in ['ImageNetEff', 'Forecasting']:
        db, meta, metrics, xdef, ydef, bounds, _, _ = databases[db_name]
        corr, props = correlations[db_name]['index'][SEL_DS_TASK[db_name]]
        prop_names = [lookup_meta(meta, prop, 'shortname', 'properties') for prop in props]
        corr_res.append( (db_name, corr, prop_names) )
    fig = make_subplots(rows=1, cols=2, subplot_titles=([res[0] for res in corr_res]))
    for col, (_, corr, prop_names) in enumerate(corr_res):
        fig.add_trace(
            go.Heatmap(z=corr, x=prop_names, y=prop_names, coloraxis="coloraxis"),
            row=1, col=1+col
        )
    fig.update_layout(coloraxis = {'colorscale': RATING_COLOR_SCALE_REV, 'colorbar': {'title': 'Pearson Corr'}})
    fig.update_layout({'width': PLOT_WIDTH, 'height': PLOT_HEIGHT, 'margin': {'l': 0, 'r': 0, 'b': 0, 't': 20}})
    # colorbar={"title": 'Pearson Corr'}
    fig.show()
    fig.write_image(f'correlation_imagenet_fc.pdf')

    # imagenet env trades
    db, meta, metrics, xdef, ydef, bounds, _, _ = databases['ImageNetEff']
    envs = sorted([env for env in pd.unique(db['environment']) if 'Xeon' not in env])
    models = sorted(pd.unique(db['model']).tolist())
    traces = {}
    for env in envs:
        subdb = db[(db['environment'] == env) & (db['task'] == 'infer')]
        avail_models = set(subdb['model'].tolist())
        traces[env] = [subdb[subdb['model'] == mod]['compound_index'].iloc[0] if mod in avail_models else None for mod in models]
    # model_names = [f'{mod[:3]}..{mod[-5:]}' if len(mod) > 10 else mod for mod in models]
    fig = go.Figure(
        layout={'width': PLOT_WIDTH, 'height': PLOT_HEIGHT, 'margin': {'l': 0, 'r': 0, 'b': 0, 't': 0},
                'yaxis':{'title': 'Compound score'}},
        data=[
            go.Scatter(x=models, y=vals, name=env.replace('+cu113', ''), mode='markers',
            marker=dict(
                color=RATING_COLORS[i],
                symbol=ENV_SYMBOLS[i]
            ),) for i, (env, vals) in enumerate(traces.items())
        ]
    )
    fig.update_layout( legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5) )
    fig.write_image(f'environment_changes.pdf')

    # PWC correlation violins with and without resources
    db, meta, metrics, xdef, ydef, bounds, _, _ = databases['Papers With Code']
    vals_with_res, vals_wo_res = [], []
    fig = go.Figure()
    for _, (corr, props) in correlations['Papers With Code']['index'].items():
        has_res = False
        for prop in props:
            if lookup_meta(meta, prop, 'group', 'properties') == 'Resources':
                has_res = True
                break
        if has_res:
            vals_with_res = vals_with_res + [c for c in corr[0].flatten() if not np.isnan(c)]
        else:
            vals_wo_res = vals_wo_res + [c for c in corr[0].flatten() if not np.isnan(c)]
    print(len(vals_wo_res), len(vals_with_res))
    fig.add_trace( go.Violin(y=vals_with_res, spanmode='hard', name=f'With resources (N={len(vals_with_res)})', line={'color': RATING_COLORS[0]}, box_visible=True, meanline_visible=True) )
    fig.add_trace( go.Violin(y=vals_wo_res, spanmode='hard', name=f'Without resources (N={len(vals_wo_res)})', line={'color': RATING_COLORS[4]}, box_visible=True, meanline_visible=True) )
    fig.update_layout(width=PLOT_WIDTH / 2, height=PLOT_HEIGHT, xaxis={'visible': False, 'showticklabels': False}, 
                      legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="center", x=0.5), margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
    fig.write_image(f'pwc_corr.pdf')


if __name__ == '__main__':
    pass