import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fname = 'imagenet_120_2024-10-23_07-50-44'

db_orig = pd.read_csv(f'{fname}.csv').sort_values('start_time')
properties = [col.replace('metrics.', '') for col in db_orig.columns if 'metrics' in col]
db = db_orig.drop(columns=[col for col in db_orig.columns if 'params' not in col and 'metrics' not in col]) # drop everything that was not logged
db = db.rename(columns=lambda col: col.replace('params.', '').replace('metrics.', ''))
db = db.dropna()

video_results = pd.read_csv(os.path.join(fname, 'video_data.csv'))
assert video_results.shape[0] == db.shape[0] * 2 # should have twice as many entries

db['externally_measured'] = video_results.iloc[1::2]['val_diff'].values * 3.6e6 / db['n_samples']
db['diff'] = db['externally_measured'] - db['power_draw']
db['rel_diff'] = db['diff'] / db['externally_measured']

grouped = db.groupby(['nogpu', 'model']).first().sort_values('parameters')
texts = grouped.loc[0].index.map(lambda v: '' if v in ['DenseNet121', 'EfficientNetB1', 'EfficientNetV2B0', 'EfficientNetV2B3', 'ResNet50V2', 'ResNet152V2'] else v)

# power draw per model for all setups
fig = go.Figure([
    go.Scatter(x=grouped.loc[0].index, y=grouped.loc[0,'power_draw'], mode='markers+lines', marker={'color': '#71c1e3'}, name='GPU / CodeCarbon'),
    go.Scatter(x=grouped.loc[0].index, y=grouped.loc[0,'externally_measured'], mode='markers+lines', marker={'color': '#009ee3'}, name='GPU / extern'),
    go.Scatter(x=grouped.loc[1].index, y=grouped.loc[1,'power_draw'], mode='markers+lines', marker={'color': '#e8a2c2'}, name='CPU / CodeCarbon'),
    go.Scatter(x=grouped.loc[1].index, y=grouped.loc[1,'externally_measured'], mode='markers+lines', marker={'color': '#e82e82'}, name='CPU / extern')
])
fig.update_yaxes(type="log")
fig.update_layout(yaxis_title='Ws per inference', xaxis_title='Model', title='Profiling Comparison: CodeCarbon VS External Energy Meter (1/2)',
                  legend=dict(title='Processor / Profiling:', orientation="v", yanchor="top", y=0.95, xanchor="left", x=0.02),
                  margin={'l': 10, 'r': 10, 'b': 10, 't': 50}, width=900, height=600)
fig.show()
print(1)

# plot correlation of parameters and energy difference
# fig = go.Figure([
#     go.Scatter(x=grouped.loc[0].index, y=grouped.loc[0,'diff'], mode='markers+lines', marker={'color': '#009ee3'}, name='GPU'),
#     go.Scatter(x=grouped.loc[1].index, y=grouped.loc[1,'diff'], mode='markers+lines', marker={'color': '#e82e82'}, name='CPU')
# ])
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=grouped.loc[0].index, y=grouped.loc[0,'diff'], mode='markers+lines', marker={'color': '#009ee3'}, name='GPU'), secondary_y=False)
fig.add_trace(go.Scatter(x=grouped.loc[1].index, y=grouped.loc[1,'diff'], mode='markers+lines', marker={'color': '#e82e82'}, name='CPU'), secondary_y=False)
fig.add_trace(go.Scatter(x=grouped.loc[1].index, y=grouped.loc[1,'parameters'], mode='markers+lines', marker={'color': '#35cdb4'}, name='Model Size'), secondary_y=True)

fig.update_yaxes(type="log", title_text=r'$\Delta \text{Ws per inference}$', secondary_y=False)
fig.update_yaxes(type="log", title_text=r'Number of Parameters', secondary_y=True)

fig.update_layout(xaxis_title='Number of Parameters', title='Profiling Comparison: CodeCarbon VS External Energy Meter (2/2)',
                  legend=dict(orientation="h", yanchor="top", y=0.95, xanchor="left", x=0.02),
                  margin={'l': 10, 'r': 10, 'b': 10, 't': 50}, width=900, height=600)
fig.show()

# for gpu, s in zip ([0, 1], ['GPU', 'CPU']):
#     sub_db = db[db['nogpu'] == gpu]
#     fig.add_traces([
#         # go.Scatter(x=db['parameters'], y=db['diff'], mode='markers', name='extern'),
#         go.Scatter(x=sub_db['model'], y=sub_db['externally_measured'], mode='markers', name=f'{s} (extern)'),
#         go.Scatter(x=sub_db['model'], y=sub_db['power_draw'], mode='markers', name=f'{s} (CodeCarbon)')
#     ])